"""
Neural Logic Layer for XRL
===========================
Learnable logic network that operates on SAE features to produce interpretable rules.

Architecture:
    SAE Features → Binarization → Logic Rules (DNF) → Action Scores

Each action is represented as a Disjunctive Normal Form (DNF):
    action_i ← (c1 ∧ c2 ∧ ¬c3) ∨ (c5 ∧ ¬c7) ∨ ...

Uses soft relaxations during training and hard logic at inference for full interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class LearnableNeuralLogicLayer(nn.Module):
    """
    Learnable Neural Logic Network with interpretable rules.

    Each action has multiple clauses (conjunctions).
    Each clause is a conjunction of literals (positive/negative features).
    Action = OR of all its clauses (Disjunctive Normal Form).

    Uses temperature annealing to transition from soft to hard logic.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        n_clauses_per_action: int = 10,
        initial_temp: float = 5.0,
        l0_penalty: float = 1e-4,
        device: str = "cpu"
    ):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_clauses_per_action = n_clauses_per_action
        self.n_clauses = n_actions * n_clauses_per_action
        self.l0_penalty = l0_penalty
        self.device = device

        # Temperature for annealing (starts high, decreases to make logic harder)
        self.register_buffer('temperature', torch.tensor(initial_temp))

        # Learnable continuous weights (will be quantized to {-1, 0, 1})
        # Shape: (n_features, n_clauses)
        # clause_weights[:, j] defines clause j
        # +1 = feature must be active, -1 = feature must be inactive, 0 = don't care
        # Initialize with larger values so sign() gives meaningful ternary weights
        self.clause_weights = nn.Parameter(
            torch.randn(n_features, self.n_clauses, device=device) * 2.0
        )

        # Learnable feature selection mask (which features to use in each clause)
        # Uses L0 regularization to encourage sparsity
        # Initialize with positive bias so features are initially included
        self.mask_logits = nn.Parameter(
            torch.randn(n_features, self.n_clauses, device=device) * 0.5 + 2.0
        )

        # Clause to action assignment
        # clause i belongs to action (i // n_clauses_per_action)
        self.register_buffer(
            'action_assignment',
            torch.arange(self.n_clauses, device=device) // n_clauses_per_action
        )

    def update_temperature(self, new_temp: float):
        """Update temperature for annealing schedule"""
        self.temperature.fill_(new_temp)

    def sample_mask(self) -> torch.Tensor:
        """
        Sample binary mask indicating which features are used in each clause.
        Uses Concrete/Gumbel-Softmax distribution for differentiability.
        """
        if self.training:
            # Soft mask during training (differentiable)
            # Concrete distribution: Gumbel-Softmax reparameterization
            u = torch.rand_like(self.mask_logits)
            s = torch.sigmoid(
                (torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + self.mask_logits) /
                (self.temperature + 1e-8)
            )
            return s
        else:
            # Hard mask at inference
            return (torch.sigmoid(self.mask_logits) > 0.5).float()

    def get_clause_weights(self) -> torch.Tensor:
        """
        Get ternary clause weights {-1, 0, 1} masked by feature selection.

        Returns:
            weights: (n_features, n_clauses) in [-1, 1] (soft) or {-1, 0, 1} (hard)
        """
        mask = self.sample_mask()

        if self.training:
            # Soft ternary during training using tanh with temperature
            ternary_soft = torch.tanh(self.clause_weights / (self.temperature + 1e-8))
            return ternary_soft * mask
        else:
            # Hard ternary at inference
            ternary_hard = torch.sign(self.clause_weights)
            return ternary_hard * mask

    def evaluate_clause_soft(
        self,
        features: torch.Tensor,
        clause_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft evaluation of a clause using smooth minimum (LogSumExp trick).

        A clause is a conjunction: c1 ∧ c2 ∧ ¬c3 ∧ ...
        We use smooth-min as soft AND: AND(x) ≈ smooth_min(x)

        This is more stable than product t-norm for many literals.

        Args:
            features: (batch, n_features) in [0, 1]
            clause_weight: (n_features,) in [-1, 1]

        Returns:
            satisfaction: (batch,) in [0, 1]
        """
        batch_size = features.shape[0]

        # Only consider non-zero weighted features
        active_mask = (clause_weight.abs() > 1e-6)

        if active_mask.sum() == 0:
            # Empty clause = always satisfied
            return torch.ones(batch_size, device=features.device)

        # Get literal values for active features
        # For positive weight: use feature value
        # For negative weight: use (1 - feature)
        literal_values = torch.where(
            clause_weight > 0,
            features,  # positive literal
            1 - features  # negative literal
        )

        # Extract only active literals
        # Shape: (batch, n_features) -> (batch, n_active)
        active_literals = literal_values[:, active_mask]

        # Soft AND using smooth minimum via LogSumExp
        # smooth_min(x) = -temp * logsumexp(-x / temp)
        # For temp → 0, this → min(x) (hard AND)
        # For temp → ∞, this → mean(x)
        temp = self.temperature.clamp(min=0.1)

        # Compute soft minimum
        # satisfaction = -temp * torch.logsumexp(-active_literals / temp, dim=1)
        # Simplified: Use weighted mean (more stable, still differentiable)
        satisfaction = active_literals.mean(dim=1)

        return satisfaction

    def evaluate_clause_hard(
        self,
        features: torch.Tensor,
        clause_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Hard evaluation (exact logic) for inference.

        Args:
            features: (batch, n_features) binary {0, 1}
            clause_weight: (n_features,) in {-1, 0, 1}

        Returns:
            satisfaction: (batch,) binary {0, 1}
        """
        batch_size = features.shape[0]
        satisfaction = torch.ones(batch_size, device=features.device)

        # Positive literals must all be 1
        pos_mask = (clause_weight > 0)
        if pos_mask.any():
            satisfaction *= features[:, pos_mask].min(dim=1)[0]

        # Negative literals must all be 0 (i.e., 1 - feature = 1)
        neg_mask = (clause_weight < 0)
        if neg_mask.any():
            satisfaction *= (1 - features[:, neg_mask]).min(dim=1)[0]

        return satisfaction

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features → clause satisfaction → action scores

        Uses soft logic during training for gradients, hard logic at inference

        Args:
            features: (batch, n_features) binary or continuous SAE features

        Returns:
            action_logits: (batch, n_actions)
        """
        batch_size = features.shape[0]
        clause_weights = self.get_clause_weights()

        # Evaluate all clauses
        clause_sat = torch.zeros(batch_size, self.n_clauses, device=features.device)

        if self.training:
            # Use SOFT logic during training for gradients
            for j in range(self.n_clauses):
                clause_sat[:, j] = self.evaluate_clause_soft(features, clause_weights[:, j])
        else:
            # Use HARD logic at inference for interpretability
            for j in range(self.n_clauses):
                clause_sat[:, j] = self.evaluate_clause_hard(features, clause_weights[:, j])

        # Aggregate clauses per action using OR (sum of satisfied clauses)
        action_scores = torch.zeros(batch_size, self.n_actions, device=features.device)

        for a in range(self.n_actions):
            action_mask = (self.action_assignment == a)
            action_clauses = clause_sat[:, action_mask]

            if action_clauses.numel() == 0:
                continue

            # Sum clause satisfaction scores (soft OR during training, count at inference)
            action_scores[:, a] = action_clauses.sum(dim=1)

        return action_scores

    def complexity_penalty(self) -> torch.Tensor:
        """
        Regularization to encourage sparse, simple rules.

        Returns:
            L0 penalty on the number of features used
        """
        # L0: expected number of active features (based on mask probabilities)
        mask_prob = torch.sigmoid(self.mask_logits)
        l0_loss = self.l0_penalty * mask_prob.sum()

        return l0_loss

    def extract_rules(
        self,
        feature_names: Optional[List[str]] = None,
        action_names: Optional[List[str]] = None,
        min_weight_threshold: float = 0.05
    ) -> Dict[str, List[str]]:
        """
        Extract human-readable logic rules in DNF form.

        Args:
            feature_names: List of feature names
            action_names: List of action names
            min_weight_threshold: Minimum |weight| to include literal

        Returns:
            Dictionary mapping action_name -> list of clause strings
        """
        self.eval()
        clause_weights = self.get_clause_weights().detach().cpu()

        if action_names is None:
            action_names = [f"action_{a}" for a in range(self.n_actions)]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(self.n_features)]

        rules = {}

        for a in range(self.n_actions):
            action_name = action_names[a]
            action_mask = (self.action_assignment == a).cpu()
            clauses = []

            for clause_idx in torch.where(action_mask)[0]:
                clause = clause_weights[:, clause_idx]

                # Extract literals from this clause
                literals = []
                for feat_idx in range(self.n_features):
                    w = clause[feat_idx].item()
                    if abs(w) < min_weight_threshold:
                        continue

                    feat_name = feature_names[feat_idx]
                    if w > 0:
                        literals.append(feat_name)
                    else:
                        literals.append(f"¬{feat_name}")

                if literals:
                    clause_str = " ∧ ".join(literals)
                    clauses.append(f"({clause_str})")

            if clauses:
                rules[action_name] = clauses
            else:
                rules[action_name] = ["⊥"]  # False (never taken)

        return rules

    def get_dnf_string(
        self,
        action_idx: int,
        feature_names: Optional[List[str]] = None,
        action_names: Optional[List[str]] = None
    ) -> str:
        """Get DNF formula for a specific action"""
        rules = self.extract_rules(feature_names, action_names)
        action_name = action_names[action_idx] if action_names else f"action_{action_idx}"
        clauses = rules.get(action_name, ["⊥"])
        return f"{action_name} ← " + " ∨ ".join(clauses)

    def count_active_rules(self, threshold: float = 0.1) -> Dict[str, int]:
        """Count statistics about the learned rules"""
        self.eval()
        clause_weights = self.get_clause_weights().detach().cpu()

        stats = {
            'total_clauses': self.n_clauses,
            'non_empty_clauses': 0,
            'total_literals': 0,
            'avg_literals_per_clause': 0.0,
            'clauses_per_action': {}
        }

        non_empty = 0
        total_lits = 0

        for j in range(self.n_clauses):
            clause = clause_weights[:, j]
            n_literals = (clause.abs() > threshold).sum().item()

            if n_literals > 0:
                non_empty += 1
                total_lits += n_literals

        stats['non_empty_clauses'] = non_empty
        stats['total_literals'] = total_lits
        stats['avg_literals_per_clause'] = total_lits / max(non_empty, 1)

        for a in range(self.n_actions):
            action_mask = (self.action_assignment == a).cpu()
            action_clauses = clause_weights[:, action_mask]
            n_active = sum((action_clauses.abs() > threshold).any(dim=0)).item()
            stats['clauses_per_action'][f'action_{a}'] = n_active

        return stats


def binarize_sae_features(
    features: torch.Tensor,
    method: str = "threshold",
    threshold: float = 0.1,
    top_k: Optional[int] = None
) -> torch.Tensor:
    """
    Convert continuous SAE features to binary for logic operations.

    Args:
        features: (batch, n_features) continuous activations
        method: "threshold" or "topk"
        threshold: threshold value for binarization
        top_k: number of top features to activate (for topk method)

    Returns:
        binary_features: (batch, n_features) in {0, 1}
    """
    if method == "threshold":
        return (features > threshold).float()
    elif method == "topk":
        if top_k is None:
            raise ValueError("top_k must be specified for topk method")

        # Get top-k indices
        _, topk_idx = torch.topk(features, top_k, dim=-1)

        # Create binary mask
        binary = torch.zeros_like(features)
        binary.scatter_(-1, topk_idx, 1.0)

        return binary
    else:
        raise ValueError(f"Unknown binarization method: {method}")
