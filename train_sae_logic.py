"""
SAE + Neural Logic Network Training
====================================
Integrates Sparse Autoencoder with Learnable Neural Logic Layer.

Architecture:
    Input → Encoder → SAE Features → Binarization → Logic Rules → Actions

Can be trained in two modes:
    1. Two-stage: Pre-train SAE, then train logic layer
    2. Joint: Train SAE and logic layer together (end-to-end)

Usage:
    # Two-stage training (safer)
    python train_sae_logic.py \
        --features_path ./stage1_outputs/collected_data.pt \
        --stage1_path ./stage1_outputs/stage1_outputs.pt \
        --mode two_stage \
        --save_dir ./sae_logic_outputs

    # Joint training (better performance)
    python train_sae_logic.py \
        --features_path ./stage1_outputs/collected_data.pt \
        --stage1_path ./stage1_outputs/stage1_outputs.pt \
        --mode joint \
        --save_dir ./sae_logic_outputs
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import from existing XRL codebase
from sparse_concept_autoencoder import OvercompleteSAE, SAEConfig, init_from_stage1
from neural_logic_layer import LearnableNeuralLogicLayer, binarize_sae_features


# ============================================================================
# Differentiable Binarization
# ============================================================================

def gumbel_topk_binarization(
    features: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    training: bool = True
) -> torch.Tensor:
    """
    Differentiable top-k binarization using Gumbel-Softmax.

    During training: Uses Gumbel noise + straight-through estimator
    During inference: Uses hard top-k selection

    Args:
        features: (batch, n_features) continuous SAE activations
        k: number of features to select
        temperature: Gumbel-Softmax temperature (lower = more discrete)
        training: whether in training mode

    Returns:
        binary_features: (batch, n_features) in {0, 1} with gradients
    """
    if training:
        # Add Gumbel noise for stochastic exploration
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(features) + 1e-8) + 1e-8)
        logits = features + temperature * gumbel_noise
    else:
        logits = features

    # Hard top-k selection
    _, topk_idx = torch.topk(logits, k, dim=-1)
    binary_hard = torch.zeros_like(features)
    binary_hard.scatter_(-1, topk_idx, 1.0)

    if training:
        # Straight-through estimator: hard forward, soft backward
        # Soft distribution: sigmoid over scaled features
        soft_probs = torch.sigmoid(features / (temperature + 1e-8))

        # Keep top-k for soft as well (soft masking)
        soft_topk = torch.zeros_like(features)
        soft_topk.scatter_(-1, topk_idx, 1.0)
        soft_probs = soft_probs * soft_topk

        # Straight-through: gradient flows through soft, but forward uses hard
        binary = binary_hard.detach() - soft_probs.detach() + soft_probs
        return binary
    else:
        return binary_hard


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SAELogicConfig:
    """Configuration for SAE + Logic training"""

    # SAE parameters (from existing config)
    input_dim: int = 128
    hidden_dim: int = 256
    k: int = 10
    n_actions: int = 7

    # Logic layer parameters
    n_clauses_per_action: int = 10
    initial_temp: float = 5.0
    min_temp: float = 0.5  # Keep higher minimum for stability
    temp_decay: float = 0.99  # Slower decay (was 0.98)

    # Feature binarization
    binarization_method: str = "topk"  # "threshold" or "topk"
    binarization_threshold: float = 0.01  # Lower threshold if using threshold method
    binarization_topk: Optional[int] = None  # Auto-set to k if None

    # Loss weights
    alpha: float = 1.0          # SAE reconstruction
    beta: float = 2.0           # Action prediction
    lambda1: float = 5e-3       # SAE L1 sparsity
    lambda2: float = 1e-4       # Logic complexity (L0)

    # Training mode
    training_mode: str = "joint"  # "two_stage" or "joint"
    sae_freeze_epoch: int = 50    # For joint: when to freeze SAE

    # Training
    n_epochs: int = 200
    batch_size: int = 256
    sae_lr: float = 1e-3
    logic_lr: float = 5e-3
    seed: int = 42

    # SAE initialization
    use_ica_init: bool = True

    # Logging
    log_every: int = 10
    save_dir: str = "./sae_logic_outputs"


# ============================================================================
# Integrated Model
# ============================================================================

class SAELogicAgent(nn.Module):
    """
    Complete agent: SAE feature extraction + Neural Logic reasoning

    Forward:
        features → SAE encode → binary features → logic layer → action scores
    """

    def __init__(
        self,
        config: SAELogicConfig,
        device: str = "cpu"
    ):
        super().__init__()
        self.config = config
        self.device = device

        # SAE for feature extraction
        self.sae = OvercompleteSAE(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            k=config.k
        ).to(device)

        # Neural logic layer for action prediction
        self.logic_layer = LearnableNeuralLogicLayer(
            n_features=config.hidden_dim,
            n_actions=config.n_actions,
            n_clauses_per_action=config.n_clauses_per_action,
            initial_temp=config.initial_temp,
            l0_penalty=config.lambda2,
            device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch, input_dim) input features
            return_features: if True, return intermediate activations

        Returns:
            action_logits: (batch, n_actions)
            (optional) features: dict of intermediate values
        """
        # SAE encoding
        z_sparse, z_pre = self.sae.encode(x)

        # Determine k for top-k selection
        topk_val = self.config.binarization_topk
        if topk_val is None:
            topk_val = self.config.k

        # Binarize features using Gumbel-Softmax (differentiable)
        # Use logic layer's temperature for consistency
        gumbel_temp = self.logic_layer.temperature.item()
        logic_features = gumbel_topk_binarization(
            z_sparse,
            k=topk_val,
            temperature=gumbel_temp,
            training=self.training
        )

        # Logic layer forward
        action_logits = self.logic_layer(logic_features)

        if return_features:
            x_recon = self.sae.decode(z_sparse)

            return action_logits, {
                'z_sparse': z_sparse,
                'z_pre': z_pre,
                'logic_features': logic_features,
                'binary_features': logic_features,  # Now same as logic_features
                'x_recon': x_recon
            }

        return action_logits

    def compute_loss(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        epoch: int = 0
    ) -> tuple:
        """
        Compute combined loss

        Args:
            x: (batch, input_dim) input features
            actions: (batch,) ground truth actions
            epoch: current epoch (for freezing SAE)

        Returns:
            total_loss, info_dict
        """
        # Forward pass
        action_logits, features = self.forward(x, return_features=True)

        # Action prediction loss
        action_loss = F.cross_entropy(action_logits, actions)

        # SAE reconstruction loss
        x_recon = features['x_recon']
        recon_loss = F.mse_loss(x_recon, x)

        # SAE sparsity loss (L1 on pre-topk activations)
        z_pre = features['z_pre']
        sparsity_loss = self.config.lambda1 * z_pre.abs().mean()

        # Logic complexity penalty
        logic_complexity = self.logic_layer.complexity_penalty()

        # Total loss
        if self.config.training_mode == "joint":
            # In joint mode, optionally freeze SAE after warmup
            if epoch >= self.config.sae_freeze_epoch:
                total_loss = (
                    self.config.beta * action_loss +
                    logic_complexity
                )
            else:
                total_loss = (
                    self.config.alpha * recon_loss +
                    self.config.beta * action_loss +
                    sparsity_loss +
                    logic_complexity
                )
        else:
            # Two-stage: SAE is frozen, only train logic
            total_loss = (
                self.config.beta * action_loss +
                logic_complexity
            )

        # Compute accuracy
        acc = (action_logits.argmax(1) == actions).float().mean()

        info = {
            'total_loss': total_loss.item(),
            'action_loss': action_loss.item(),
            'recon_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'logic_complexity': logic_complexity.item(),
            'accuracy': acc.item(),
            'feature_density': (features['z_sparse'] > 0).float().mean().item(),
            'binary_density': features['binary_features'].mean().item(),
            'temperature': self.logic_layer.temperature.item()
        }

        return total_loss, info

    def extract_rules(
        self,
        concept_labels: Optional[list] = None,
        action_names: Optional[list] = None
    ) -> Dict:
        """Extract interpretable rules"""
        return self.logic_layer.extract_rules(
            feature_names=concept_labels,
            action_names=action_names
        )


# ============================================================================
# Training Functions
# ============================================================================

def train_two_stage(
    model: SAELogicAgent,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SAELogicConfig,
    device: str
):
    """
    Two-stage training:
        Stage 1: Pre-train SAE (freeze logic)
        Stage 2: Train logic layer (freeze SAE)
    """
    print("\n" + "="*70)
    print("TWO-STAGE TRAINING")
    print("="*70)

    # Stage 1: Pre-train SAE
    print("\n[Stage 1] Pre-training SAE...")
    model.sae.train()
    model.logic_layer.eval()

    sae_optimizer = torch.optim.Adam(model.sae.parameters(), lr=config.sae_lr)

    for epoch in range(config.n_epochs // 2):
        epoch_losses = []

        for batch_x, batch_a in train_loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)

            # SAE reconstruction only
            z_sparse, z_pre = model.sae.encode(batch_x)
            x_recon = model.sae.decode(z_sparse)

            recon_loss = F.mse_loss(x_recon, batch_x)
            sparsity_loss = config.lambda1 * z_pre.abs().mean()
            loss = config.alpha * recon_loss + sparsity_loss

            sae_optimizer.zero_grad()
            loss.backward()

            # Normalize decoder columns
            with torch.no_grad():
                model.sae._normalize_decoder()

            sae_optimizer.step()

            epoch_losses.append(loss.item())

        if (epoch + 1) % config.log_every == 0:
            print(f"  Epoch {epoch+1}/{config.n_epochs//2} | Loss: {np.mean(epoch_losses):.4f}")

    # Stage 2: Train logic layer
    print("\n[Stage 2] Training logic layer...")
    model.sae.eval()
    model.logic_layer.train()

    # Freeze SAE
    for param in model.sae.parameters():
        param.requires_grad = False

    logic_optimizer = torch.optim.Adam(model.logic_layer.parameters(), lr=config.logic_lr)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(config.n_epochs // 2, config.n_epochs):
        # Update temperature
        current_temp = max(
            config.min_temp,
            config.initial_temp * (config.temp_decay ** epoch)
        )
        model.logic_layer.update_temperature(current_temp)

        # Training
        model.logic_layer.train()
        train_info = []

        for batch_x, batch_a in train_loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)

            loss, info = model.compute_loss(batch_x, batch_a, epoch)

            logic_optimizer.zero_grad()
            loss.backward()
            logic_optimizer.step()

            train_info.append(info)

        # Validation
        val_acc = evaluate(model, val_loader, device)

        if (epoch + 1) % config.log_every == 0:
            avg_info = {k: np.mean([d[k] for d in train_info]) for k in train_info[0]}
            print(f"  Epoch {epoch+1}/{config.n_epochs} | "
                  f"Loss: {avg_info['total_loss']:.4f} | "
                  f"Train Acc: {avg_info['accuracy']:.3f} | "
                  f"Val Acc: {val_acc:.3f} | "
                  f"Temp: {current_temp:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }

    return best_model_state, best_val_acc


def train_joint(
    model: SAELogicAgent,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SAELogicConfig,
    device: str
):
    """
    Joint training: Train SAE and logic layer together
    """
    print("\n" + "="*70)
    print("JOINT TRAINING")
    print("="*70)

    # Optimizers
    optimizer = torch.optim.Adam([
        {'params': model.sae.parameters(), 'lr': config.sae_lr},
        {'params': model.logic_layer.parameters(), 'lr': config.logic_lr}
    ])

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(config.n_epochs):
        # Update temperature
        current_temp = max(
            config.min_temp,
            config.initial_temp * (config.temp_decay ** epoch)
        )
        model.logic_layer.update_temperature(current_temp)

        # Debug: Print feature statistics on first epoch
        if epoch == 0:
            with torch.no_grad():
                batch_x, batch_a = next(iter(train_loader))
                batch_x = batch_x.to(device)
                z_sparse, _ = model.sae.encode(batch_x)
                topk_val = config.binarization_topk if config.binarization_topk else config.k
                binary_features = binarize_sae_features(
                    z_sparse, config.binarization_method, config.binarization_threshold, topk_val
                )
                print(f"\n[Debug] Feature stats (first batch):")
                print(f"  SAE features: min={z_sparse.min():.4f}, max={z_sparse.max():.4f}, mean={z_sparse.mean():.4f}")
                print(f"  Binary features: density={binary_features.mean():.4f}, sum/sample={binary_features.sum(1).mean():.1f}")
                print(f"  Binarization: method={config.binarization_method}, threshold={config.binarization_threshold}, topk={topk_val}\n")

        # Optionally freeze SAE after warmup
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

            # Normalize decoder columns
            if epoch < config.sae_freeze_epoch:
                with torch.no_grad():
                    model.sae._normalize_decoder()

            optimizer.step()

            train_info.append(info)

        # Validation
        val_acc = evaluate(model, val_loader, device)

        if (epoch + 1) % config.log_every == 0:
            avg_info = {k: np.mean([d[k] for d in train_info]) for k in train_info[0]}
            print(f"  Epoch {epoch+1}/{config.n_epochs} | "
                  f"Loss: {avg_info['total_loss']:.4f} | "
                  f"Train Acc: {avg_info['accuracy']:.3f} | "
                  f"Val Acc: {val_acc:.3f} | "
                  f"Temp: {current_temp:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }

    return best_model_state, best_val_acc


def evaluate(model: SAELogicAgent, loader: DataLoader, device: str) -> float:
    """Evaluate model accuracy

    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation data
        device: Device to run on
    """
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


# ============================================================================
# Main Training Loop
# ============================================================================

def main(args):
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data = torch.load(args.features_path, weights_only=False)
    features = data['features']  # (N, input_dim)
    actions = data['actions']    # (N,)

    # Split train/val
    n_train = int(0.9 * len(features))
    indices = torch.randperm(len(features))

    train_dataset = TensorDataset(features[indices[:n_train]], actions[indices[:n_train]])
    val_dataset = TensorDataset(features[indices[n_train:]], actions[indices[n_train:]])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create config
    config = SAELogicConfig(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        k=args.k,
        n_actions=actions.max().item() + 1,
        n_clauses_per_action=args.n_clauses_per_action,
        training_mode=args.mode,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        seed=args.seed
    )

    # Create model
    print("\nInitializing model...")
    model = SAELogicAgent(config, device=device)

    # Optionally initialize SAE from stage1
    if args.stage1_path and os.path.exists(args.stage1_path):
        print("Initializing SAE from Stage 1...")
        stage1_data = torch.load(args.stage1_path, weights_only=False)

        # Create SAEConfig matching the actual model dimensions
        sae_config = SAEConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            k=config.k,
            n_actions=config.n_actions,
            use_ica_init=config.use_ica_init
        )
        init_from_stage1(model.sae, stage1_data, sae_config)

    # Train
    if args.mode == "two_stage":
        best_state, best_acc = train_two_stage(model, train_loader, val_loader, config, device)
    elif args.mode == "joint":
        best_state, best_acc = train_joint(model, train_loader, val_loader, config, device)
    else:
        raise ValueError(f"Unknown training mode: {args.mode}")

    print(f"\n✓ Training complete! Best validation accuracy: {best_acc:.3f}")

    # Load best model (handle case where no improvement occurred)
    if best_state is not None:
        model.load_state_dict(best_state['model'])
    else:
        print("\n⚠ Warning: No improvement during training. Using final model state.")
        best_state = {
            'model': model.state_dict(),
            'epoch': config.n_epochs - 1,
            'val_acc': best_acc
        }

    # Extract rules
    print("\nExtracting rules...")
    action_names = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]
    rules = model.extract_rules(action_names=action_names)

    print("\n" + "="*70)
    print("LEARNED RULES (DNF Form)")
    print("="*70)
    for action_name, clauses in rules.items():
        print(f"\n{action_name} ←")
        for i, clause in enumerate(clauses):
            print(f"    {clause}")
            if i < len(clauses) - 1:
                print("  ∨")

    # Save results
    save_path = os.path.join(args.save_dir, "sae_logic_model.pt")
    torch.save({
        'model_state': best_state['model'],
        'config': asdict(config),
        'rules': rules,
        'best_val_acc': best_acc
    }, save_path)

    # Save rules as JSON
    rules_path = os.path.join(args.save_dir, "learned_rules.json")
    with open(rules_path, 'w') as f:
        json.dump(rules, f, indent=2)

    print(f"\n✓ Saved model to: {save_path}")
    print(f"✓ Saved rules to: {rules_path}")

    # Print statistics
    stats = model.logic_layer.count_active_rules()
    print("\n" + "="*70)
    print("RULE STATISTICS")
    print("="*70)
    print(f"Total clauses: {stats['total_clauses']}")
    print(f"Non-empty clauses: {stats['non_empty_clauses']}")
    print(f"Average literals per clause: {stats['avg_literals_per_clause']:.2f}")
    print("\nClauses per action:")
    for action, count in stats['clauses_per_action'].items():
        print(f"  {action}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE + Neural Logic Network")

    # Data
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to collected_data.pt from stage 1")
    parser.add_argument("--stage1_path", type=str, default=None,
                        help="Path to stage1 outputs for SAE initialization")

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="SAE hidden dimension (overcomplete)")
    parser.add_argument("--k", type=int, default=10,
                        help="TopK sparsity")
    parser.add_argument("--n_clauses_per_action", type=int, default=10,
                        help="Number of logic clauses per action")

    # Training
    parser.add_argument("--mode", type=str, default="joint",
                        choices=["two_stage", "joint"],
                        help="Training mode")
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output
    parser.add_argument("--save_dir", type=str, default="./sae_logic_outputs",
                        help="Directory to save outputs")

    args = parser.parse_args()
    main(args)
