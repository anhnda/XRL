"""
SAE + Product T-Norm Neural Logic Training (V2)
=================================================
Fully differentiable neuro-symbolic pipeline:
    Input → Normalize → SAE → Sigmoid Bottleneck → Product T-Norm Logic → Actions

Key changes from V1:
    1. Input normalization using Stage 1 mean/std
    2. Sigmoid bottleneck: maps SAE activations to [0,1] with learnable sharpness
    3. Bimodality loss: pushes activations toward {0,1} during training
    4. Product t-norm logic: fully differentiable AND/OR/NOT, no hard binarization
    5. No train-eval distribution shift — same computation in both modes

Usage:
    python train_sae_logic_v2.py \
        --features_path ./stage1_outputs/collected_data.pt \
        --stage1_path ./stage1_outputs/stage1_outputs.pt \
        --mode joint \
        --hidden_dim 256 \
        --k 10 \
        --n_clauses_per_action 10 \
        --n_epochs 200 \
        --save_dir ./sae_logic_v2_outputs
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import SAE from existing codebase
from sparse_concept_autoencoder import OvercompleteSAE, SAEConfig, init_from_stage1


# ============================================================================
# Sigmoid Bottleneck
# ============================================================================

class SigmoidBottleneck(nn.Module):
    """
    Maps continuous SAE activations to [0,1] with learnable sharpness.

    Each feature i has:
        - alpha_i: sharpness (how steep the sigmoid is)
        - beta_i:  threshold (where the transition happens)

    output_i = sigmoid(alpha_i * (z_i - beta_i))

    Early training: alpha ~ 1, soft sigmoid, gradients flow everywhere
    Late training:  bimodality loss pushes alpha higher, activations become near-binary
    """

    def __init__(self, n_features: int, initial_alpha: float = 1.0):
        super().__init__()
        # log_alpha so alpha is always positive via softplus
        self.log_alpha = nn.Parameter(
            torch.full((n_features,), np.log(np.exp(initial_alpha) - 1.0))
        )
        self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, n_features) sparse SAE activations

        Returns:
            (batch, n_features) values in [0, 1]
        """
        alpha = F.softplus(self.log_alpha) + 0.5  # minimum sharpness 0.5
        return torch.sigmoid(alpha * (z - self.beta))

    def get_sharpness(self) -> torch.Tensor:
        """Return current alpha values for logging."""
        return F.softplus(self.log_alpha) + 0.5


# ============================================================================
# Product T-Norm Logic Layer
# ============================================================================

class ProductTNormLogicLayer(nn.Module):
    """
    Fully differentiable logic layer using product t-norm.

    For each action a, learns n_clauses clauses in DNF form:
        action_a = OR(clause_1, clause_2, ..., clause_n)

    Each clause is a conjunction (AND) over literals:
        clause_j = AND(literal_j1, literal_j2, ..., literal_jm)

    Each literal is a soft selection:
        literal_ji = p_ji * f_i + n_ji * (1 - f_i) + (1 - p_ji - n_ji) * 1.0

    Where:
        p_ji = probability of including f_i positively
        n_ji = probability of including f_i negated
        (1 - p_ji - n_ji) = probability of ignoring f_i (contributes 1 to product)

    Product t-norm AND: clause = product of all literals
    Probabilistic sum OR: action = 1 - product(1 - clause_j)
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        n_clauses_per_action: int = 10,
        l0_penalty_weight: float = 1e-4,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_clauses_per_action = n_clauses_per_action
        self.l0_penalty_weight = l0_penalty_weight

        total_clauses = n_actions * n_clauses_per_action

        # Learnable selection logits
        # w_pos[c, i]: logit for including feature i positively in clause c
        # w_neg[c, i]: logit for including feature i negated in clause c
        # Initialize NEGATIVE so softmax strongly favors "absent" (third category).
        # With w_pos=w_neg=-3, absent_logit=0: softmax gives p≈0.02, n≈0.02, absent≈0.95
        # This means literal ≈ 1.0 for all features → log(literal) ≈ 0
        # → log_clause_sum ≈ 0 → sigmoid(0 + bias) ≈ sigmoid(2) ≈ 0.88
        # Clauses start active with good gradients, and learn which features to select.
        self.w_pos = nn.Parameter(torch.randn(total_clauses, n_features) * 0.01 - 3.0)
        self.w_neg = nn.Parameter(torch.randn(total_clauses, n_features) * 0.01 - 3.0)

        # Per-clause bias (shifts the sigmoid activation threshold)
        # With log_clause_sum ≈ 0 at init, bias=2 gives sigmoid(2)≈0.88
        self.clause_weight = nn.Parameter(torch.ones(total_clauses) * 2.0)

    def _get_selection_probs(self):
        """
        Compute selection probabilities ensuring p + n <= 1.

        Uses a 3-way softmax over [pos, neg, absent] for each (clause, feature).
        """
        # Stack into (total_clauses, n_features, 3)
        absent_logit = torch.zeros_like(self.w_pos)
        logits = torch.stack([self.w_pos, self.w_neg, absent_logit], dim=-1)

        probs = F.softmax(logits, dim=-1)
        p = probs[..., 0]  # (total_clauses, n_features)
        n = probs[..., 1]  # (total_clauses, n_features)
        return p, n

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Log-sum-sigmoid formulation for stable gradients.

        Instead of clause = product(literals) which vanishes over 256 features,
        we compute clause = sigmoid(sum(log(literals)) + bias).

        This is monotonically related to the product t-norm but the outer sigmoid
        prevents gradient vanishing by rescaling the output to [0, 1] with
        guaranteed non-zero gradients.

        Args:
            features: (batch, n_features) values in [0, 1] from sigmoid bottleneck

        Returns:
            action_logits: (batch, n_actions) unnormalized scores
        """
        batch_size = features.shape[0]
        p, n = self._get_selection_probs()  # each (total_clauses, n_features)

        f = features.unsqueeze(1)  # (batch, 1, n_features)
        p_expanded = p.unsqueeze(0)  # (1, total_clauses, n_features)
        n_expanded = n.unsqueeze(0)  # (1, total_clauses, n_features)

        # literal = p * f + n * (1 - f) + (1 - p - n) * 1
        literals = (
            p_expanded * f +
            n_expanded * (1.0 - f) +
            (1.0 - p_expanded - n_expanded)
        )
        # literals: (batch, total_clauses, n_features), each in [0, 1]

        # Log-sum: equivalent to log(product) but we apply sigmoid after
        log_literals = torch.log(literals + 1e-8)
        log_clause_sum = log_literals.sum(dim=-1)  # (batch, total_clauses)

        # Sigmoid with learnable bias — prevents vanishing gradient
        # clause_weight acts as bias controlling activation threshold
        clauses = torch.sigmoid(log_clause_sum + self.clause_weight.unsqueeze(0))
        # clauses: (batch, total_clauses), each in [0, 1]

        # Reshape to (batch, n_actions, n_clauses_per_action)
        clauses = clauses.view(batch_size, self.n_actions, self.n_clauses_per_action)

        # OR over clauses: sum gives action logits directly
        # Each clause in [0,1], sum gives "how many clauses fire"
        action_logits = clauses.sum(dim=-1)  # (batch, n_actions)

        return action_logits

    def complexity_penalty(self) -> torch.Tensor:
        """
        L0-style penalty encouraging sparse clauses.
        Penalizes the total selection probability (p + n) across all clauses and features.
        """
        p, n = self._get_selection_probs()
        # p + n = probability that feature i is used (positively or negated) in clause j
        usage = p + n  # (total_clauses, n_features)
        # Mean usage across all clause-feature pairs
        penalty = self.l0_penalty_weight * usage.mean()
        return penalty

    def extract_rules(
        self,
        feature_names: Optional[List[str]] = None,
        action_names: Optional[List[str]] = None,
        threshold: float = 0.3,
    ) -> Dict[str, List[str]]:
        """
        Extract human-readable rules by thresholding selection probabilities.

        Args:
            feature_names: names for each feature dimension
            action_names: names for each action
            threshold: minimum selection probability to include a literal

        Returns:
            dict mapping action names to lists of clause strings
        """
        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(self.n_features)]
        if action_names is None:
            action_names = [f"action_{a}" for a in range(self.n_actions)]

        p, n = self._get_selection_probs()
        p = p.detach().cpu().numpy()
        n = n.detach().cpu().numpy()
        clause_bias = self.clause_weight.detach().cpu().numpy()

        rules = {}
        for a in range(self.n_actions):
            clauses = []
            for c in range(self.n_clauses_per_action):
                idx = a * self.n_clauses_per_action + c

                literals = []
                for i in range(self.n_features):
                    if p[idx, i] > threshold:
                        literals.append(f"{feature_names[i]}")
                    elif n[idx, i] > threshold:
                        literals.append(f"¬{feature_names[i]}")

                if literals:
                    clause_str = " ∧ ".join(literals)
                    clauses.append(f"({clause_str}) [bias={clause_bias[idx]:.2f}]")

            rules[action_names[a]] = clauses if clauses else ["(no active clauses)"]

        return rules

    def count_active_rules(
        self,
        threshold: float = 0.3,
        action_names: Optional[List[str]] = None,
    ) -> Dict:
        """Count statistics about active rules."""
        if action_names is None:
            action_names = [f"action_{a}" for a in range(self.n_actions)]

        p, n = self._get_selection_probs()
        p = p.detach().cpu().numpy()
        n = n.detach().cpu().numpy()

        total_clauses = 0
        non_empty = 0
        total_literals = 0
        per_action = {}

        for a in range(self.n_actions):
            action_clauses = 0
            for c in range(self.n_clauses_per_action):
                idx = a * self.n_clauses_per_action + c

                n_literals = ((p[idx] > threshold) | (n[idx] > threshold)).sum()
                total_clauses += 1
                if n_literals > 0:
                    non_empty += 1
                    total_literals += n_literals
                    action_clauses += 1

            per_action[action_names[a]] = action_clauses

        return {
            'total_clauses': total_clauses,
            'non_empty_clauses': non_empty,
            'avg_literals_per_clause': total_literals / max(non_empty, 1),
            'clauses_per_action': per_action,
        }


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SAELogicConfig:
    """Configuration for SAE + Product T-Norm Logic training"""

    # SAE parameters
    input_dim: int = 128
    hidden_dim: int = 256
    k: int = 10
    n_actions: int = 7

    # Sigmoid bottleneck
    initial_alpha: float = 1.0  # initial sharpness (increases via bimodality loss)

    # Logic layer parameters
    n_clauses_per_action: int = 10
    l0_penalty_weight: float = 1e-4

    # Loss weights
    alpha_recon: float = 0.1       # SAE reconstruction
    beta_action: float = 5.0       # Action prediction
    lambda_sparsity: float = 5e-3  # SAE L1 sparsity
    lambda_bimodal: float = 0.0    # Bimodality loss (annealed during training)
    bimodal_max: float = 1.0       # Maximum bimodality weight
    bimodal_warmup: int = 30       # Epochs before bimodality loss starts
    bimodal_ramp: int = 50         # Epochs to ramp bimodality to max

    # Per-class loss weights. Boost Done (index 6) to force rule learning.
    action_class_weights: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # Training mode
    training_mode: str = "joint"  # "two_stage" or "joint"
    sae_freeze_epoch: int = 80    # For joint: when to freeze SAE

    # Training
    n_epochs: int = 200
    batch_size: int = 256
    sae_lr: float = 1e-3
    logic_lr: float = 3e-3
    bottleneck_lr: float = 1e-3
    seed: int = 42

    # SAE initialization
    use_ica_init: bool = True

    # Logging
    log_every: int = 10
    save_dir: str = "./sae_logic_v2_outputs"


# ============================================================================
# Integrated Model
# ============================================================================

class SAELogicAgentV2(nn.Module):
    """
    Fully differentiable neuro-symbolic agent.

    Forward:
        normalized_features → SAE encode → sigmoid bottleneck → product t-norm logic → action scores

    No hard binarization anywhere. Bimodality loss pushes bottleneck outputs toward {0,1}.
    """

    def __init__(self, config: SAELogicConfig, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device

        # SAE for feature extraction
        self.sae = OvercompleteSAE(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            k=config.k,
        ).to(device)

        # Sigmoid bottleneck: maps SAE activations to [0, 1]
        self.bottleneck = SigmoidBottleneck(
            n_features=config.hidden_dim,
            initial_alpha=config.initial_alpha,
        ).to(device)

        # Product t-norm logic layer
        self.logic_layer = ProductTNormLogicLayer(
            n_features=config.hidden_dim,
            n_actions=config.n_actions,
            n_clauses_per_action=config.n_clauses_per_action,
            l0_penalty_weight=config.l0_penalty_weight,
        ).to(device)

        # Normalization stats (registered as buffers — not learned)
        self.register_buffer('feature_mean', torch.zeros(config.input_dim))
        self.register_buffer('feature_std', torch.ones(config.input_dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Set normalization statistics from Stage 1."""
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean.to(x.device)) / self.feature_std.to(x.device)

    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = False,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, input_dim) features (pre-normalized or raw)
            normalize: if True, apply normalization (for inference with raw CNN output)
            return_features: if True, return intermediate activations

        Returns:
            action_logits: (batch, n_actions)
            (optional) features dict
        """
        if normalize:
            x = self.normalize(x)

        # SAE encoding
        z_sparse, z_pre = self.sae.encode(x)

        # Sigmoid bottleneck → [0, 1]
        z_binary = self.bottleneck(z_sparse)

        # Product t-norm logic
        action_logits = self.logic_layer(z_binary)

        if return_features:
            x_recon = self.sae.decode(z_sparse)
            return action_logits, {
                'z_sparse': z_sparse,
                'z_pre': z_pre,
                'z_binary': z_binary,
                'x_recon': x_recon,
            }

        return action_logits

    def compute_loss(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        epoch: int = 0,
    ) -> tuple:
        """
        Compute combined loss.

        Args:
            x: (batch, input_dim) normalized features
            actions: (batch,) ground truth actions
            epoch: current epoch (for annealing)

        Returns:
            total_loss, info_dict
        """
        action_logits, features = self.forward(x, return_features=True)

        # --- Action prediction loss ---
        # Build weight tensor on the correct device (cached via buffer would be
        # cleaner, but constructing here is fine — it's tiny and fast).
        class_weights = torch.tensor(
            self.config.action_class_weights,
            dtype=torch.float32,
            device=x.device,
        )
        action_loss = F.cross_entropy(action_logits, actions, weight=class_weights)
        # --- SAE reconstruction loss ---
        # Only penalize reconstruction where SAE activated (z_sparse > 0 means that feature fired)
        # Decode back to input space: active features already determine which input dims are explained
        # active_mask = (features['z_sparse'] > 0).float()  # (batch, hidden_dim) — which concepts fired

        # x_recon is already sparse-decoded. Just don't penalize dims where recon is zero.
        #active_input_mask = (features['x_recon'].abs() > 1e-6).float()  # (batch, input_dim)

        #recon_loss = (active_input_mask * (features['x_recon'] - x) ** 2).sum() / (active_input_mask.sum() + 1e-8)
        recon_loss = F.mse_loss(features['x_recon'], x)

        # --- SAE sparsity loss ---
        sparsity_loss = self.config.lambda_sparsity * features['z_pre'].abs().mean()

        # --- Bimodality loss on bottleneck output ---
        # L_bimodal = mean(z * (1 - z)), minimized when z ∈ {0, 1}
        z_bin = features['z_binary']
        bimodal_raw = (z_bin * (1.0 - z_bin)).mean()

        # Anneal bimodality weight
        if epoch < self.config.bimodal_warmup:
            bimodal_weight = 0.0
        else:
            progress = min(
                1.0,
                (epoch - self.config.bimodal_warmup) / max(self.config.bimodal_ramp, 1)
            )
            bimodal_weight = self.config.bimodal_max * progress
        bimodal_loss = bimodal_weight * bimodal_raw

        # --- Logic complexity penalty ---
        logic_complexity = self.logic_layer.complexity_penalty()
        recon_weight = max(
            0.0001,
            0.1 * (1.0 - epoch / max(self.config.sae_freeze_epoch, 1))
        )
        act_weight = 1# min (10, (1+epoch * 0.1))
        # --- Total loss ---
        if self.config.training_mode == "joint" and epoch >= self.config.sae_freeze_epoch:
            # SAE frozen: only action + bimodality + logic
            total_loss = (
                self.config.beta_action * action_loss +
                bimodal_loss +
                logic_complexity
            )
        else:
            total_loss = (
                self.config.alpha_recon * recon_loss  +
                self.config.beta_action * action_loss  +
                sparsity_loss +
                bimodal_loss +
                logic_complexity
            )

        # --- Metrics ---
        acc = (action_logits.argmax(1) == actions).float().mean()

        # Measure how binary the bottleneck outputs are
        # Fraction of values within 0.05 of {0, 1}
        near_binary = ((z_bin < 0.05) | (z_bin > 0.95)).float().mean()

        info = {
            'total_loss': total_loss.item(),
            'action_loss': action_loss.item(),
            'recon_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'bimodal_loss': bimodal_loss.item(),
            'bimodal_weight': bimodal_weight,
            'bimodal_raw': bimodal_raw.item(),
            'logic_complexity': logic_complexity.item(),
            'accuracy': acc.item(),
            'near_binary_frac': near_binary.item(),
            'feature_density': (features['z_sparse'] > 0).float().mean().item(),
            'bottleneck_mean': z_bin.mean().item(),
            'bottleneck_std': z_bin.std().item(),
            'alpha_mean': self.bottleneck.get_sharpness().mean().item(),
        }

        return total_loss, info

    def extract_rules(
        self,
        concept_labels: Optional[List[str]] = None,
        action_names: Optional[List[str]] = None,
    ) -> Dict:
        """Extract interpretable rules from the logic layer."""
        return self.logic_layer.extract_rules(
            feature_names=concept_labels,
            action_names=action_names,
        )


# ============================================================================
# Training Functions
# ============================================================================

def train_two_stage(
    model: SAELogicAgentV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SAELogicConfig,
    device: str,
):
    """
    Two-stage training:
        Stage A: Pre-train SAE + bottleneck (freeze logic)
        Stage B: Train logic layer + fine-tune bottleneck (freeze SAE)
    """
    print("\n" + "=" * 70)
    print("TWO-STAGE TRAINING")
    print("=" * 70)

    history = []

    # --- Stage A: Pre-train SAE ---
    print("\n[Stage A] Pre-training SAE + bottleneck...")
    sae_params = list(model.sae.parameters()) + list(model.bottleneck.parameters())
    sae_optimizer = torch.optim.Adam(sae_params, lr=config.sae_lr)

    n_sae_epochs = config.n_epochs // 2

    for epoch in range(n_sae_epochs):
        model.sae.train()
        model.bottleneck.train()
        epoch_losses = []

        for batch_x, batch_a in train_loader:
            batch_x = batch_x.to(device)

            z_sparse, z_pre = model.sae.encode(batch_x)
            x_recon = model.sae.decode(z_sparse)
            z_binary = model.bottleneck(z_sparse)

            recon_loss = F.mse_loss(x_recon, batch_x)
            sparsity_loss = config.lambda_sparsity * z_pre.abs().mean()

            # Bimodality loss (annealed)
            bimodal_raw = (z_binary * (1.0 - z_binary)).mean()
            if epoch < config.bimodal_warmup:
                bimodal_w = 0.0
            else:
                progress = min(1.0, (epoch - config.bimodal_warmup) / max(config.bimodal_ramp, 1))
                bimodal_w = config.bimodal_max * progress
            bimodal_loss = bimodal_w * bimodal_raw

            loss = config.alpha_recon * recon_loss + sparsity_loss + bimodal_loss

            sae_optimizer.zero_grad()
            loss.backward()
        
            with torch.no_grad():
                model.sae._normalize_decoder()

            sae_optimizer.step()
            epoch_losses.append(loss.item())

        if (epoch + 1) % config.log_every == 0:
            near_binary = ((z_binary < 0.05) | (z_binary > 0.95)).float().mean().item()
            print(
                f"  Epoch {epoch+1}/{n_sae_epochs} | "
                f"Loss: {np.mean(epoch_losses):.4f} | "
                f"Recon: {recon_loss.item():.4f} | "
                f"NearBinary: {near_binary:.3f} | "
                f"BimodalW: {bimodal_w:.3f}"
            )

    # --- Stage B: Train logic layer ---
    print("\n[Stage B] Training logic layer...")
    for param in model.sae.parameters():
        param.requires_grad = False

    logic_params = list(model.logic_layer.parameters()) + list(model.bottleneck.parameters())
    logic_optimizer = torch.optim.Adam([
        {'params': model.logic_layer.parameters(), 'lr': config.logic_lr},
        {'params': model.bottleneck.parameters(), 'lr': config.bottleneck_lr * 0.1},  # fine-tune
    ])

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(n_sae_epochs, config.n_epochs):
        model.logic_layer.train()
        model.bottleneck.train()
        train_info = []

        for batch_x, batch_a in train_loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)

            loss, info = model.compute_loss(batch_x, batch_a, epoch)

            logic_optimizer.zero_grad()
            loss.backward()
            logic_optimizer.step()

            train_info.append(info)

        val_acc = evaluate(model, val_loader, device)
        history.append({**avg_dict(train_info), 'val_acc': val_acc, 'epoch': epoch})

        if (epoch + 1) % config.log_every == 0:
            avg = avg_dict(train_info)
            print(
                f"  Epoch {epoch+1}/{config.n_epochs} | "
                f"Loss: {avg['total_loss']:.4f} | "
                f"TrainAcc: {avg['accuracy']:.3f} | "
                f"ValAcc: {val_acc:.3f} | "
                f"NearBin: {avg['near_binary_frac']:.3f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }

    return best_model_state, best_val_acc, history


def train_joint(
    model: SAELogicAgentV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SAELogicConfig,
    device: str,
):
    """
    Joint training: SAE + bottleneck + logic layer together.
    SAE optionally frozen after sae_freeze_epoch.
    """
    print("\n" + "=" * 70)
    print("JOINT TRAINING")
    print("=" * 70)

    optimizer = torch.optim.Adam([
        {'params': model.sae.parameters(), 'lr': config.sae_lr},
        {'params': model.bottleneck.parameters(), 'lr': config.bottleneck_lr},
        {'params': model.logic_layer.parameters(), 'lr': config.logic_lr},
    ])

    best_val_acc = 0.0
    best_model_state = None
    history = []

    for epoch in range(config.n_epochs):
        # Debug: print feature statistics at epoch 0
        if epoch == 0:
            with torch.no_grad():
                batch_x, batch_a = next(iter(train_loader))
                batch_x = batch_x.to(device)
                z_sparse, _ = model.sae.encode(batch_x)
                z_binary = model.bottleneck(z_sparse)
                print(f"\n[Debug] Feature stats (first batch):")
                print(f"  Input (normalized): min={batch_x.min():.4f}, max={batch_x.max():.4f}, mean={batch_x.mean():.4f}")
                print(f"  SAE activations: min={z_sparse.min():.4f}, max={z_sparse.max():.4f}, mean={z_sparse.mean():.4f}")
                print(f"  Bottleneck output: min={z_binary.min():.4f}, max={z_binary.max():.4f}, mean={z_binary.mean():.4f}")
                print(f"  Bottleneck near-binary: {((z_binary < 0.05) | (z_binary > 0.95)).float().mean():.3f}")
                print(f"  SAE feature density: {(z_sparse > 0).float().mean():.4f}")
                print(f"  Bottleneck sharpness (alpha): mean={model.bottleneck.get_sharpness().mean():.3f}\n")

        # Freeze SAE after warmup
        if epoch == config.sae_freeze_epoch:
            print(f"\n[Info] Freezing SAE at epoch {epoch}")
            for param in model.sae.parameters():
                param.requires_grad = False

        # Training
        model.train()
        train_info = []

        for batch_x, batch_a in train_loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)

            loss, info = model.compute_loss(batch_x, batch_a, epoch)

            optimizer.zero_grad()
            loss.backward()

            # # Normalize decoder columns
            # if epoch < config.sae_freeze_epoch:
            #     with torch.no_grad():
            #         model.sae._normalize_decoder()
            # if (epoch + 1) % config.log_every == 0:
            #     log_gradient_norms(model, epoch)
            torch.nn.utils.clip_grad_norm_(model.sae.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.bottleneck.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.logic_layer.parameters(), max_norm=1.0)
            optimizer.step()
            train_info.append(info)

        # Validation
        val_acc = evaluate(model, val_loader, device)
        history.append({**avg_dict(train_info), 'val_acc': val_acc, 'epoch': epoch})

        if (epoch + 1) % config.log_every == 0:
            avg = avg_dict(train_info)
            print(
                f"  Epoch {epoch+1}/{config.n_epochs} | "
                f"Loss: {avg['total_loss']:.4f} | "
                f"Act: {avg['action_loss']:.4f} | "
                f"Rec: {avg['recon_loss']:.4f} | "
                f"TrainAcc: {avg['accuracy']:.3f} | "
                f"ValAcc: {val_acc:.3f} | "
                f"NearBin: {avg['near_binary_frac']:.3f} | "
                f"α: {avg['alpha_mean']:.1f} | "
                f"BimW: {avg['bimodal_weight']:.2f}"
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }

    return best_model_state, best_val_acc, history


def evaluate(model: SAELogicAgentV2, loader: DataLoader, device: str) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_a in loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)
            logits = model(batch_x)
            preds = logits.argmax(1)
            correct += (preds == batch_a).sum().item()
            total += batch_a.size(0)

    return correct / total
def log_gradient_norms(model, epoch):
    def _norm(params):
        grads = [p.grad for p in params if p.grad is not None]
        return torch.stack([g.norm(2) for g in grads]).norm(2).item() if grads else 0.0
    print(
        f"  [GradNorm] epoch={epoch} | "
        f"SAE={_norm(model.sae.parameters()):.4f} | "
        f"Bottleneck={_norm(model.bottleneck.parameters()):.4f} | "
        f"Logic={_norm(model.logic_layer.parameters()):.4f}"
    )

def linear_probe(
    model: SAELogicAgentV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    n_epochs: int = 50,
    lr: float = 1e-3,
):
    """
    Train a linear probe on bottleneck features to measure information content.
    If this gets high accuracy, the SAE+bottleneck preserves action-relevant info
    and the logic layer is the bottleneck.
    """
    print(f"\n{'='*70}")
    print("LINEAR PROBE (information ceiling test)")
    print(f"{'='*70}")

    model.eval()
    n_features = model.config.hidden_dim
    n_actions = model.config.n_actions

    # Extract bottleneck features
    train_z, train_a = [], []
    val_z, val_a = [], []

    with torch.no_grad():
        for batch_x, batch_a in train_loader:
            batch_x = batch_x.to(device)
            z_sparse, _ = model.sae.encode(batch_x)
            z_binary = model.bottleneck(z_sparse)
            train_z.append(z_binary.cpu())
            train_a.append(batch_a)

        for batch_x, batch_a in val_loader:
            batch_x = batch_x.to(device)
            z_sparse, _ = model.sae.encode(batch_x)
            z_binary = model.bottleneck(z_sparse)
            val_z.append(z_binary.cpu())
            val_a.append(batch_a)

    train_z = torch.cat(train_z, 0)
    train_a = torch.cat(train_a, 0)
    val_z = torch.cat(val_z, 0)
    val_a = torch.cat(val_a, 0)

    print(f"  Bottleneck features: {train_z.shape[1]}-d, "
          f"near-binary: {((train_z < 0.05) | (train_z > 0.95)).float().mean():.3f}")

    # Train linear probe
    probe = nn.Linear(n_features, n_actions).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    probe_train = DataLoader(
        TensorDataset(train_z, train_a), batch_size=256, shuffle=True
    )

    for epoch in range(n_epochs):
        probe.train()
        for bz, ba in probe_train:
            bz, ba = bz.to(device), ba.to(device)
            loss = F.cross_entropy(probe(bz), ba)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        train_preds = probe(train_z.to(device)).argmax(1).cpu()
        val_preds = probe(val_z.to(device)).argmax(1).cpu()
        train_acc = (train_preds == train_a).float().mean().item()
        val_acc = (val_preds == val_a).float().mean().item()

    print(f"  Linear probe train accuracy: {train_acc:.3f}")
    print(f"  Linear probe val accuracy:   {val_acc:.3f}")

    if val_acc > 0.7:
        print(f"  → Bottleneck features contain enough info. Logic layer is the bottleneck.")
    else:
        print(f"  → Bottleneck features lost too much info. SAE/bottleneck needs improvement.")

    return train_acc, val_acc


# ============================================================================
# Utilities
# ============================================================================

def avg_dict(dicts):
    """Average a list of dicts with numeric values."""
    keys = dicts[0].keys()
    return {k: np.mean([d[k] for d in dicts]) for k in keys}


def plot_training_history(history, save_dir):
    """Plot training curves."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SAE + Product T-Norm Logic Training", fontsize=14, fontweight="bold")

    # 1. Loss components
    ax = axes[0, 0]
    ax.plot(epochs, [h['total_loss'] for h in history], label='Total', linewidth=2)
    ax.plot(epochs, [h['action_loss'] for h in history], label='Action', alpha=0.7)
    ax.plot(epochs, [h['recon_loss'] for h in history], label='Recon', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [h['accuracy'] for h in history], label='Train', linewidth=2)
    ax.plot(epochs, [h['val_acc'] for h in history], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Near-binary fraction
    ax = axes[0, 2]
    ax.plot(epochs, [h['near_binary_frac'] for h in history], linewidth=2, color='teal')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Fraction near {0,1}')
    ax.set_title('Bottleneck Binarization Progress')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 4. Bimodality loss
    ax = axes[1, 0]
    ax.plot(epochs, [h['bimodal_loss'] for h in history], label='Weighted', linewidth=2)
    ax.plot(epochs, [h['bimodal_raw'] for h in history], label='Raw', alpha=0.7)
    ax.plot(epochs, [h['bimodal_weight'] for h in history], label='Weight', linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Bimodality')
    ax.set_title('Bimodality Loss & Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Bottleneck sharpness
    ax = axes[1, 1]
    ax.plot(epochs, [h['alpha_mean'] for h in history], linewidth=2, color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean α (sharpness)')
    ax.set_title('Sigmoid Bottleneck Sharpness')
    ax.grid(True, alpha=0.3)

    # 6. Feature density
    ax = axes[1, 2]
    ax.plot(epochs, [h['feature_density'] for h in history], label='SAE density', linewidth=2)
    ax.plot(epochs, [h['bottleneck_mean'] for h in history], label='Bottleneck mean', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Density')
    ax.set_title('Feature Sparsity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {plot_path}")


# ============================================================================
# Main
# ============================================================================

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load data ---
    print("\nLoading data...")
    data = torch.load(args.features_path, weights_only=False)
    features = data['features']  # (N, input_dim) raw CNN outputs
    actions = data['actions']    # (N,)
    print(f"  Raw features: shape={features.shape}, range=[{features.min():.2f}, {features.max():.2f}]")

    # --- Load Stage 1 and normalize ---
    stage1_data = None
    if args.stage1_path and os.path.exists(args.stage1_path):
        print(f"  Loading Stage 1 from {args.stage1_path}...")
        stage1_data = torch.load(args.stage1_path, weights_only=False)
        feat_mean = stage1_data['feature_mean']
        feat_std = stage1_data['feature_std']

        # Normalize features
        features = (features - feat_mean) / feat_std
        print(f"  Normalized features: range=[{features.min():.2f}, {features.max():.2f}], mean={features.mean():.4f}")
    else:
        print("  WARNING: No stage1_path provided. Using raw features (not recommended).")
        feat_mean = features.mean(dim=0)
        feat_std = features.std(dim=0).clamp(min=1e-6)
        features = (features - feat_mean) / feat_std

    # --- Train/val split ---
    n_train = int(0.9 * len(features))
    indices = torch.randperm(len(features), generator=torch.Generator().manual_seed(args.seed))

    train_dataset = TensorDataset(features[indices[:n_train]], actions[indices[:n_train]])
    val_dataset = TensorDataset(features[indices[n_train:]], actions[indices[n_train:]])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"  Actions: {actions.max().item() + 1} classes")

    # --- Create config ---
    config = SAELogicConfig(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        k=args.k,
        n_actions=int(actions.max().item() + 1),
        n_clauses_per_action=args.n_clauses_per_action,
        training_mode=args.mode,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        seed=args.seed,
        # Bimodality schedule
        bimodal_max=args.bimodal_max,
        bimodal_warmup=args.bimodal_warmup,
        bimodal_ramp=args.bimodal_ramp,
        # Penalty
        l0_penalty_weight=args.l0_penalty,
        lambda_sparsity=args.lambda_sparsity,
        # Learning rates
        sae_lr=args.sae_lr,
        logic_lr=args.logic_lr,
        bottleneck_lr=args.bottleneck_lr,
        # Freeze
        sae_freeze_epoch=args.sae_freeze_epoch,
        # Class weights: Done (index 6) boosted
        action_class_weights=tuple(args.action_class_weights),

    )

    # --- Create model ---
    print("\nInitializing model...")
    model = SAELogicAgentV2(config, device=device)

    # Store normalization stats in model (for inference)
    model.set_normalization(feat_mean, feat_std)

    # Initialize SAE from Stage 1 (ICA directions, etc.)
    if stage1_data is not None and config.use_ica_init:
        print("Initializing SAE from Stage 1...")
        sae_config = SAEConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            k=config.k,
            n_actions=config.n_actions,
            use_ica_init=config.use_ica_init,
        )
        init_from_stage1(model.sae, stage1_data, sae_config)

    # Print config
    print(f"\nConfig:")
    print(f"  Architecture: {config.input_dim} → SAE({config.hidden_dim}, k={config.k}) → "
          f"Sigmoid → Logic({config.n_clauses_per_action} clauses/action) → {config.n_actions} actions")
    print(f"  Mode: {config.training_mode}")
    print(f"  Bimodality: warmup={config.bimodal_warmup}, ramp={config.bimodal_ramp}, max={config.bimodal_max}")
    print(f"  SAE freeze epoch: {config.sae_freeze_epoch}")

    # --- Train ---
    if args.mode == "two_stage":
        best_state, best_acc, history = train_two_stage(
            model, train_loader, val_loader, config, device
        )
    elif args.mode == "joint":
        best_state, best_acc, history = train_joint(
            model, train_loader, val_loader, config, device
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"\n{'='*70}")
    print(f"Training complete! Best validation accuracy: {best_acc:.3f}")
    print(f"{'='*70}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state['model'])
    else:
        print("WARNING: No improvement during training.")
        best_state = {
            'model': model.state_dict(),
            'epoch': config.n_epochs - 1,
            'val_acc': best_acc,
        }

    # --- Linear probe to test information content ---
    linear_probe(model, train_loader, val_loader, device)

    # --- Extract rules ---
    print("\nExtracting rules...")
    action_names = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]
    rules = model.extract_rules(action_names=action_names)

    print(f"\n{'='*70}")
    print("LEARNED RULES (DNF Form)")
    print(f"{'='*70}")
    # for action_name, clauses in rules.items():
    #     print(f"\n{action_name} ←")
    #     for i, clause in enumerate(clauses):
    #         print(f"    {clause}")
    #         if i < len(clauses) - 1:
    #             print("  ∨")
    for action_name, clauses in rules.items():
        print(f"\n{action_name} ←")
        unique_clauses = list(dict.fromkeys(clauses))  # preserve order, remove dupes
        for i, clause in enumerate(unique_clauses):
            print(f"    {clause}")
            if i < len(unique_clauses) - 1:
                print("  ∨")
    # --- Check binarization quality ---
    print(f"\n{'='*70}")
    print("BINARIZATION QUALITY CHECK")
    print(f"{'='*70}")
    model.eval()
    with torch.no_grad():
        all_z = []
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            z_sparse, _ = model.sae.encode(batch_x)
            z_binary = model.bottleneck(z_sparse)
            all_z.append(z_binary.cpu())
        all_z = torch.cat(all_z, dim=0)

        near_01 = ((all_z < 0.05) | (all_z > 0.95)).float().mean()
        near_05 = ((all_z > 0.4) & (all_z < 0.6)).float().mean()
        print(f"  Values near {{0,1}} (within 0.05): {near_01:.3f}")
        print(f"  Values near 0.5 (ambiguous):     {near_05:.3f}")
        print(f"  Mean bottleneck value:           {all_z.mean():.4f}")
        print(f"  Mean sharpness (alpha):          {model.bottleneck.get_sharpness().mean():.1f}")

    # --- Save ---
    save_path = os.path.join(args.save_dir, "sae_logic_v2_model.pt")
    torch.save({
        'model_state': best_state['model'],
        'config': asdict(config),
        'rules': rules,
        'best_val_acc': best_acc,
        'feature_mean': feat_mean,
        'feature_std': feat_std,
    }, save_path)

    rules_path = os.path.join(args.save_dir, "learned_rules.json")
    with open(rules_path, 'w') as f:
        json.dump(rules, f, indent=2)

    # Plot training history
    plot_training_history(history, args.save_dir)

    # Print rule statistics
    stats = model.logic_layer.count_active_rules(action_names=action_names)
    print(f"\n{'='*70}")
    print("RULE STATISTICS")
    print(f"{'='*70}")
    print(f"  Total clauses: {stats['total_clauses']}")
    print(f"  Non-empty clauses: {stats['non_empty_clauses']}")
    print(f"  Avg literals per clause: {stats['avg_literals_per_clause']:.2f}")
    print(f"\n  Clauses per action:")
    for action, count in stats['clauses_per_action'].items():
        print(f"    {action}: {count}")

    print(f"\n  Saved model: {save_path}")
    print(f"  Saved rules: {rules_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE + Product T-Norm Logic (V2)")

    # Data
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--stage1_path", type=str, default=None)

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_clauses_per_action", type=int, default=10)

    # Training
    parser.add_argument("--mode", type=str, default="joint", choices=["two_stage", "joint"])
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    # Learning rates
    parser.add_argument("--sae_lr", type=float, default=1e-3)
    parser.add_argument("--logic_lr", type=float, default=3e-3)
    parser.add_argument("--bottleneck_lr", type=float, default=1e-3)

    # Bimodality schedule
    parser.add_argument("--bimodal_max", type=float, default=1.0)
    parser.add_argument("--bimodal_warmup", type=int, default=30)
    parser.add_argument("--bimodal_ramp", type=int, default=50)

    # Regularization
    parser.add_argument("--l0_penalty", type=float, default=1e-4)
    parser.add_argument("--lambda_sparsity", type=float, default=5e-3)

    # Freezing
    parser.add_argument("--sae_freeze_epoch", type=int, default=80)

    # Output
    parser.add_argument("--save_dir", type=str, default="./sae_logic_v2_outputs")
    parser.add_argument(
            "--action_class_weights",
            type=float,
            nargs=7,
            default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            metavar=("TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"),
            help="Per-action loss weights. Boost rare actions to force rule learning.",
        )
    args = parser.parse_args()
    main(args)