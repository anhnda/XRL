"""
SAE + Product T-Norm Neural Logic Training (V3 - Unified)
==========================================================
Supports both MiniGrid and Atari environments.
The SAE + logic pipeline is fully environment-agnostic — it operates
on collected feature vectors, so the only env-specific part is the
action names displayed in outputs.

Key insight: The original V2 worked at epoch 30 (93.4% val acc) before the
SAE drifted. The fix is dead simple:

    1. Pre-train SAE alone (recon only) until it converges
    2. FREEZE SAE completely
    3. Compute activation statistics from frozen SAE
    4. Apply FIXED normalization (not learned) between SAE and bottleneck
    5. Train bottleneck + logic on the frozen, normalized features

No learnable scaler. No gradient wars. The logic layer sees a stable feature
space from the start.

Usage (MiniGrid):
    python train_sae_logic.py \\
        --features_path ./stage1_outputs/collected_data.pt \\
        --stage1_path ./stage1_outputs/stage1_outputs.pt \\
        --hidden_dim 300 --k 50 \\
        --n_clauses_per_action 10 \\
        --sae_pretrain_epochs 50 \\
        --n_epochs 400 \\
        --save_dir ./sae_logic_v3_outputs

Usage (Atari Breakout):
    python train_sae_logic.py \\
        --features_path ./stage1_atari/collected_data.pt \\
        --stage1_path ./stage1_atari/stage1_outputs.pt \\
        --hidden_dim 1024 --k 100 \\
        --n_clauses_per_action 10 \\
        --sae_pretrain_epochs 50 \\
        --n_epochs 400 \\
        --save_dir ./sae_logic_atari
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict, field
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

from sparse_concept_autoencoder import OvercompleteSAE, SAEConfig, init_from_stage1


# ---------------------------------------------------------------------------
# Default action names per env type (fallback if not in collected data)
# ---------------------------------------------------------------------------

DEFAULT_ACTION_NAMES = {
    "minigrid": ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"],
    "atari_4":  ["NOOP", "FIRE", "RIGHT", "LEFT"],                          # Breakout
    "atari_6":  ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"], # Pong, SpaceInvaders
}


def get_action_names(n_actions: int, stored_names=None, env_type: str = None):
    """
    Resolve action names from (in priority order):
      1. Names stored in the collected_data.pt file
      2. Known defaults keyed on env_type + n_actions
      3. Generic fallback: action_0, action_1, ...
    """
    if stored_names is not None:
        if isinstance(stored_names, list) and len(stored_names) >= n_actions:
            return stored_names[:n_actions]

    if env_type == "minigrid" and n_actions <= 7:
        return DEFAULT_ACTION_NAMES["minigrid"][:n_actions]
    if env_type == "atari":
        key = f"atari_{n_actions}"
        if key in DEFAULT_ACTION_NAMES:
            return DEFAULT_ACTION_NAMES[key]

    return [f"action_{i}" for i in range(n_actions)]


# ============================================================================
# Sigmoid Bottleneck
# ============================================================================

class SigmoidBottleneck(nn.Module):
    def __init__(self, n_features: int, initial_alpha: float = 1.0):
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.full((n_features,), np.log(np.exp(initial_alpha) - 1.0))
        )
        self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.log_alpha) + 0.5
        return torch.sigmoid(alpha * (z - self.beta))

    def get_sharpness(self) -> torch.Tensor:
        return F.softplus(self.log_alpha) + 0.5


# ============================================================================
# Product T-Norm Logic Layer
# ============================================================================

class ProductTNormLogicLayer(nn.Module):
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

        self.w_pos = nn.Parameter(torch.randn(total_clauses, n_features) * 0.01 - 3.0)
        self.w_neg = nn.Parameter(torch.randn(total_clauses, n_features) * 0.01 - 3.0)
        self.clause_weight = nn.Parameter(torch.ones(total_clauses) * 2.0)

    def _get_selection_probs(self):
        absent_logit = torch.zeros_like(self.w_pos)
        logits = torch.stack([self.w_pos, self.w_neg, absent_logit], dim=-1)
        probs = F.softmax(logits, dim=-1)
        return probs[..., 0], probs[..., 1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        p, n = self._get_selection_probs()

        f = features.unsqueeze(1)
        p_ex = p.unsqueeze(0)
        n_ex = n.unsqueeze(0)

        literals = p_ex * f + n_ex * (1.0 - f) + (1.0 - p_ex - n_ex)
        log_literals = torch.log(literals + 1e-8)
        log_clause_sum = log_literals.sum(dim=-1)

        clauses = torch.sigmoid(log_clause_sum + self.clause_weight.unsqueeze(0))
        clauses = clauses.view(batch_size, self.n_actions, self.n_clauses_per_action)
        return clauses.sum(dim=-1)

    def complexity_penalty(self) -> torch.Tensor:
        p, n = self._get_selection_probs()
        return self.l0_penalty_weight * (p + n).mean()

    def extract_rules(self, feature_names=None, action_names=None, threshold=0.3):
        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(self.n_features)]
        if action_names is None:
            action_names = [f"action_{a}" for a in range(self.n_actions)]

        p, n = self._get_selection_probs()
        p, n = p.detach().cpu().numpy(), n.detach().cpu().numpy()
        cb = self.clause_weight.detach().cpu().numpy()

        rules = {}
        for a in range(self.n_actions):
            clauses = []
            for c in range(self.n_clauses_per_action):
                idx = a * self.n_clauses_per_action + c
                lits = []
                for i in range(self.n_features):
                    if p[idx, i] > threshold:
                        lits.append(f"{feature_names[i]}")
                    elif n[idx, i] > threshold:
                        lits.append(f"¬{feature_names[i]}")
                if lits:
                    clauses.append(f"({' ∧ '.join(lits)}) [bias={cb[idx]:.2f}]")
            rules[action_names[a]] = clauses if clauses else ["(no active clauses)"]
        return rules

    def count_active_rules(self, threshold=0.3, action_names=None):
        if action_names is None:
            action_names = [f"action_{a}" for a in range(self.n_actions)]
        p, n = self._get_selection_probs()
        p, n = p.detach().cpu().numpy(), n.detach().cpu().numpy()

        total_clauses = non_empty = total_literals = 0
        per_action = {}
        for a in range(self.n_actions):
            ac = 0
            for c in range(self.n_clauses_per_action):
                idx = a * self.n_clauses_per_action + c
                nl = ((p[idx] > threshold) | (n[idx] > threshold)).sum()
                total_clauses += 1
                if nl > 0:
                    non_empty += 1
                    total_literals += nl
                    ac += 1
            per_action[action_names[a]] = ac
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
    input_dim: int = 128
    hidden_dim: int = 256
    k: int = 10
    n_actions: int = 7
    initial_alpha: float = 1.0
    n_clauses_per_action: int = 10
    l0_penalty_weight: float = 1e-4

    beta_action: float = 5.0
    lambda_bimodal: float = 0.0
    bimodal_max: float = 0.3
    bimodal_warmup: int = 30
    bimodal_ramp: int = 80

    action_class_weights: tuple = None  # None = uniform; set in main()

    sae_pretrain_epochs: int = 50
    sae_lr: float = 1e-3
    lambda_sparsity: float = 5e-3
    alpha_recon: float = 1.0

    n_epochs: int = 400
    batch_size: int = 256
    logic_lr: float = 3e-3
    bottleneck_lr: float = 1e-3
    max_grad_norm: float = 5.0

    seed: int = 42
    use_ica_init: bool = True
    log_every: int = 10
    save_dir: str = "./sae_logic_v3_outputs"

    # Metadata (not used in training, kept for reproducibility)
    env_name: str = "unknown"
    env_type: str = "unknown"

    def __post_init__(self):
        if self.action_class_weights is None:
            self.action_class_weights = tuple([1.0] * self.n_actions)


# ============================================================================
# Model
# ============================================================================

class SAELogicAgentV3(nn.Module):
    def __init__(self, config: SAELogicConfig, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device

        self.sae = OvercompleteSAE(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            k=config.k,
        ).to(device)

        self.bottleneck = SigmoidBottleneck(
            n_features=config.hidden_dim,
            initial_alpha=config.initial_alpha,
        ).to(device)

        self.logic_layer = ProductTNormLogicLayer(
            n_features=config.hidden_dim,
            n_actions=config.n_actions,
            n_clauses_per_action=config.n_clauses_per_action,
            l0_penalty_weight=config.l0_penalty_weight,
        ).to(device)

        self.register_buffer('feature_mean', torch.zeros(config.input_dim))
        self.register_buffer('feature_std', torch.ones(config.input_dim))
        self.register_buffer('z_mean', torch.zeros(config.hidden_dim))
        self.register_buffer('z_std', torch.ones(config.hidden_dim))

        self._sae_frozen = False

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean) / self.feature_std

    def compute_z_normalization(self, loader: DataLoader):
        print("\n  Computing SAE activation statistics...")
        self.sae.eval()

        all_z = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                z_sparse, _ = self.sae.encode(batch_x)
                all_z.append(z_sparse.cpu())
        all_z = torch.cat(all_z, 0)

        z_mean = torch.zeros(self.config.hidden_dim)
        z_std = torch.ones(self.config.hidden_dim)

        for i in range(self.config.hidden_dim):
            active = all_z[:, i]
            active_nonzero = active[active > 0]
            if len(active_nonzero) > 10:
                z_mean[i] = active_nonzero.mean()
                z_std[i] = active_nonzero.std().clamp(min=1e-3)
            else:
                z_mean[i] = 0.0
                z_std[i] = 1.0

        self.z_mean.copy_(z_mean.to(self.z_mean.device))
        self.z_std.copy_(z_std.to(self.z_std.device))

        active_mask = all_z > 0
        z_normed = (all_z - z_mean) / z_std
        active_normed = z_normed[active_mask]
        print(f"    Raw SAE activations (active): mean={all_z[active_mask].mean():.2f}, "
              f"std={all_z[active_mask].std():.2f}, max={all_z.max():.2f}")
        print(f"    Normalized (active): mean={active_normed.mean():.2f}, "
              f"std={active_normed.std():.2f}, "
              f"range=[{active_normed.min():.2f}, {active_normed.max():.2f}]")
        print(f"    Feature activity rate: {active_mask.float().mean():.3f}")

    # def normalize_z(self, z: torch.Tensor) -> torch.Tensor:
    #     return (z - self.z_mean) / self.z_std
    def normalize_z(self, z: torch.Tensor) -> torch.Tensor:
        active_mask = z > 0
        z_normed = (z - self.z_mean) / self.z_std
        z_normed[~active_mask] = -5.0  # sigmoid(-5) ≈ 0.007 ≈ 0
        return z_normed
    def freeze_sae(self):
        for param in self.sae.parameters():
            param.requires_grad = False
        self._sae_frozen = True
        print("  SAE frozen.")

    def forward(self, x, normalize_input=False, return_features=False):
        if normalize_input:
            x = self.normalize_input(x)
        z_sparse, z_pre = self.sae.encode(x)
        z_normed = self.normalize_z(z_sparse)
        z_binary = self.bottleneck(z_normed)
        action_logits = self.logic_layer(z_binary)

        if return_features:
            x_recon = self.sae.decode(z_sparse)
            return action_logits, {
                'z_sparse': z_sparse,
                'z_pre': z_pre,
                'z_normed': z_normed,
                'z_binary': z_binary,
                'x_recon': x_recon,
            }
        return action_logits

    def extract_rules(self, concept_labels=None, action_names=None):
        return self.logic_layer.extract_rules(
            feature_names=concept_labels,
            action_names=action_names,
        )


# ============================================================================
# Stage 1: SAE Pre-training
# ============================================================================

# def pretrain_sae(model, train_loader, config, device):
#     print("\n" + "=" * 70)
#     print("STAGE 1: SAE PRE-TRAINING (reconstruction only)")
#     print("=" * 70)

#     optimizer = torch.optim.Adam(model.sae.parameters(), lr=config.sae_lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=config.sae_pretrain_epochs, eta_min=1e-5
#     )

#     for epoch in range(config.sae_pretrain_epochs):
#         model.sae.train()
#         epoch_recon, epoch_sparsity = [], []

#         for batch_x, _ in train_loader:
#             batch_x = batch_x.to(device)
#             z_sparse, z_pre = model.sae.encode(batch_x)
#             x_recon = model.sae.decode(z_sparse)

#             recon_loss = F.mse_loss(x_recon, batch_x)
#             sparsity_loss = config.lambda_sparsity * z_pre.abs().mean()
#             loss = config.alpha_recon * recon_loss + sparsity_loss

#             optimizer.zero_grad()
#             loss.backward()
#             with torch.no_grad():
#                 model.sae._normalize_decoder()
#             optimizer.step()

#             epoch_recon.append(recon_loss.item())
#             epoch_sparsity.append(sparsity_loss.item())

#         scheduler.step()

#         if (epoch + 1) % config.log_every == 0:
#             with torch.no_grad():
#                 batch_x, _ = next(iter(train_loader))
#                 batch_x = batch_x.to(device)
#                 z_sparse, _ = model.sae.encode(batch_x)
#                 density = (z_sparse > 0).float().mean().item()
#                 z_active_mean = z_sparse[z_sparse > 0].mean().item() if (z_sparse > 0).any() else 0

#             print(
#                 f"  Epoch {epoch+1}/{config.sae_pretrain_epochs} | "
#                 f"Recon: {np.mean(epoch_recon):.4f} | "
#                 f"Sparsity: {np.mean(epoch_sparsity):.4f} | "
#                 f"Density: {density:.3f} | "
#                 f"ActiveMean: {z_active_mean:.1f}"
#             )

#     model.sae.eval()
#     total_recon, n_batches = 0, 0
#     with torch.no_grad():
#         for batch_x, _ in train_loader:
#             batch_x = batch_x.to(device)
#             z_sparse, _ = model.sae.encode(batch_x)
#             x_recon = model.sae.decode(z_sparse)
#             total_recon += F.mse_loss(x_recon, batch_x).item()
#             n_batches += 1
#     print(f"\n  Final reconstruction MSE: {total_recon / n_batches:.4f}")

def pretrain_sae(model, train_loader, config, device):
    print("\n" + "=" * 70)
    print("STAGE 1: SAE PRE-TRAINING (reconstruction + action auxiliary)")
    print("=" * 70)

    # Auxiliary action predictor on SAE latents
    action_head = nn.Linear(config.hidden_dim, config.n_actions).to(device)
    
    optimizer = torch.optim.Adam(
        list(model.sae.parameters()) + list(action_head.parameters()),
        lr=config.sae_lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.sae_pretrain_epochs, eta_min=1e-5
    )

    lambda_action_aux = 0.5  # weight for action prediction loss

    for epoch in range(config.sae_pretrain_epochs):
        model.sae.train()
        action_head.train()
        epoch_recon, epoch_sparsity, epoch_act = [], [], []

        for batch_x, batch_a in train_loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)
            z_sparse, z_pre = model.sae.encode(batch_x)
            x_recon = model.sae.decode(z_sparse)

            recon_loss = F.mse_loss(x_recon, batch_x)
            sparsity_loss = config.lambda_sparsity * z_pre.abs().mean()
            action_loss = F.cross_entropy(action_head(z_sparse), batch_a)
            
            loss = (config.alpha_recon * recon_loss 
                    + sparsity_loss 
                    + lambda_action_aux * action_loss)

            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                model.sae._normalize_decoder()
            optimizer.step()

            epoch_recon.append(recon_loss.item())
            epoch_sparsity.append(sparsity_loss.item())
            epoch_act.append(action_loss.item())

        scheduler.step()

        if (epoch + 1) % config.log_every == 0:
            with torch.no_grad():
                batch_x, batch_a = next(iter(train_loader))
                batch_x = batch_x.to(device)
                z_sparse, _ = model.sae.encode(batch_x)
                density = (z_sparse > 0).float().mean().item()
                act_acc = (action_head(z_sparse).argmax(1) == batch_a.to(device)).float().mean().item()

            print(
                f"  Epoch {epoch+1}/{config.sae_pretrain_epochs} | "
                f"Recon: {np.mean(epoch_recon):.4f} | "
                f"ActLoss: {np.mean(epoch_act):.4f} | "
                f"ActAcc: {act_acc:.3f} | "
                f"Density: {density:.3f}"
            )
    # action_head is discarded — only SAE is kept
# ============================================================================
# Stage 2: Logic Training
# ============================================================================

def train_logic(model, train_loader, val_loader, config, device, action_names):
    print("\n" + "=" * 70)
    print("STAGE 2: LOGIC TRAINING (SAE frozen, fixed normalization)")
    print("=" * 70)

    with torch.no_grad():
        batch_x, batch_a = next(iter(train_loader))
        batch_x = batch_x.to(device)
        z_sparse, _ = model.sae.encode(batch_x)
        z_normed = model.normalize_z(z_sparse)
        z_binary = model.bottleneck(z_normed)
        print(f"\n  [Debug] Features entering logic layer:")
        print(f"    SAE activations: mean={z_sparse.mean():.2f}, max={z_sparse.max():.2f}")
        print(f"    After normalization: mean={z_normed.mean():.3f}, std={z_normed.std():.3f}, "
              f"range=[{z_normed.min():.2f}, {z_normed.max():.2f}]")
        print(f"    After bottleneck: mean={z_binary.mean():.3f}, "
              f"near-binary={((z_binary < 0.05) | (z_binary > 0.95)).float().mean():.3f}")
        print(f"    Feature density: {(z_sparse > 0).float().mean():.3f}\n")

    optimizer = torch.optim.Adam([
        {'params': model.bottleneck.parameters(), 'lr': config.bottleneck_lr},
        {'params': model.logic_layer.parameters(), 'lr': config.logic_lr},
    ])

    n_logic_epochs = config.n_epochs - config.sae_pretrain_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_logic_epochs, eta_min=1e-5
    )

    class_weights = torch.tensor(
        list(config.action_class_weights), dtype=torch.float32, device=device
    )

    best_val_acc = 0.0
    best_model_state = None
    history = []
    patience = 0
    max_patience = 100

    for epoch_idx in range(n_logic_epochs):
        epoch = epoch_idx + config.sae_pretrain_epochs

        model.bottleneck.train()
        model.logic_layer.train()
        train_info = []

        for batch_x, batch_a in train_loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)
            action_logits, features = model.forward(batch_x, return_features=True)

            action_loss = F.cross_entropy(action_logits, batch_a, weight=class_weights)

            z_bin = features['z_binary']
            bimodal_raw = (z_bin * (1.0 - z_bin)).mean()
            if epoch_idx < config.bimodal_warmup:
                bimodal_weight = 0.0
            else:
                progress = min(1.0, (epoch_idx - config.bimodal_warmup) / max(config.bimodal_ramp, 1))
                bimodal_weight = config.bimodal_max * progress
            bimodal_loss = bimodal_weight * bimodal_raw

            logic_complexity = model.logic_layer.complexity_penalty()

            total_loss = (
                config.beta_action * action_loss +
                bimodal_loss +
                logic_complexity
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.bottleneck.parameters()) + list(model.logic_layer.parameters()),
                config.max_grad_norm
            )
            optimizer.step()

            acc = (action_logits.argmax(1) == batch_a).float().mean()
            near_binary = ((z_bin < 0.05) | (z_bin > 0.95)).float().mean()

            train_info.append({
                'total_loss': total_loss.item(),
                'action_loss': action_loss.item(),
                'bimodal_loss': bimodal_loss.item(),
                'bimodal_weight': bimodal_weight,
                'bimodal_raw': bimodal_raw.item(),
                'logic_complexity': logic_complexity.item(),
                'accuracy': acc.item(),
                'near_binary_frac': near_binary.item(),
                'feature_density': (features['z_sparse'] > 0).float().mean().item(),
                'bottleneck_mean': z_bin.mean().item(),
                'bottleneck_std': z_bin.std().item(),
                'alpha_mean': model.bottleneck.get_sharpness().mean().item(),
            })

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)

        avg = avg_dict(train_info)
        avg['val_acc'] = val_acc
        avg['epoch'] = epoch
        history.append(avg)

        if (epoch + 1) % config.log_every == 0:
            lr_now = optimizer.param_groups[1]['lr']
            print(
                f"  Epoch {epoch+1}/{config.n_epochs} | "
                f"Loss: {avg['total_loss']:.4f} | "
                f"Act: {avg['action_loss']:.4f} | "
                f"TrainAcc: {avg['accuracy']:.3f} | "
                f"ValAcc: {val_acc:.3f} | "
                f"NearBin: {avg['near_binary_frac']:.3f} | "
                f"α: {avg['alpha_mean']:.1f} | "
                f"BimW: {avg['bimodal_weight']:.2f} | "
                f"LR: {lr_now:.1e}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }
            patience = 0
        else:
            patience += 1

        if patience >= max_patience and epoch_idx > 100:
            print(f"\n  [Info] Early stopping at epoch {epoch}")
            break

    return best_model_state, best_val_acc, history


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_a in loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)
            logits = model(batch_x)
            correct += (logits.argmax(1) == batch_a).sum().item()
            total += batch_a.size(0)
    return correct / total


def per_class_accuracy(model, loader, device, action_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_a in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(batch_a)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    print(f"\n  {'Action':<14} {'Count':>6} {'Correct':>8} {'Accuracy':>9}")
    print(f"  {'-'*40}")
    for i, name in enumerate(action_names):
        mask = all_labels == i
        count = mask.sum().item()
        if count > 0:
            correct = (all_preds[mask] == i).sum().item()
            print(f"  {name:<14} {count:>6} {correct:>8} {correct/count:>9.3f}")
        else:
            print(f"  {name:<14} {0:>6} {'N/A':>8} {'N/A':>9}")


def linear_probe(model, train_loader, val_loader, device, n_epochs=50, lr=1e-3):
    print(f"\n{'='*70}")
    print("LINEAR PROBE (information ceiling test)")
    print(f"{'='*70}")

    model.eval()

    def collect(loader):
        zs, acts = [], []
        with torch.no_grad():
            for bx, ba in loader:
                bx = bx.to(device)
                z_sparse, _ = model.sae.encode(bx)
                z_normed = model.normalize_z(z_sparse)
                z_bin = model.bottleneck(z_normed)
                zs.append(z_bin.cpu())
                acts.append(ba)
        return torch.cat(zs, 0), torch.cat(acts, 0)

    train_z, train_a = collect(train_loader)
    val_z, val_a = collect(val_loader)

    print(f"  Features: {train_z.shape[1]}-d, "
          f"near-binary: {((train_z < 0.05) | (train_z > 0.95)).float().mean():.3f}")

    probe = nn.Linear(train_z.shape[1], model.config.n_actions).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(train_z, train_a), batch_size=256, shuffle=True)

    for _ in range(n_epochs):
        probe.train()
        for bz, ba in dl:
            bz, ba = bz.to(device), ba.to(device)
            loss = F.cross_entropy(probe(bz), ba)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        tr_acc = (probe(train_z.to(device)).argmax(1).cpu() == train_a).float().mean().item()
        va_acc = (probe(val_z.to(device)).argmax(1).cpu() == val_a).float().mean().item()

    print(f"  Linear probe train acc: {tr_acc:.3f}")
    print(f"  Linear probe val acc:   {va_acc:.3f}")
    if va_acc > 0.7:
        print(f"  → Features have enough info. Logic layer is the bottleneck.")
    else:
        print(f"  → Features lost info. SAE/bottleneck needs work.")
    return tr_acc, va_acc


# ============================================================================
# Utilities
# ============================================================================

def avg_dict(dicts):
    keys = dicts[0].keys()
    return {k: np.mean([d[k] for d in dicts]) for k in keys}


def plot_training_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SAE + Product T-Norm Logic Training (V3)", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, [h['total_loss'] for h in history], label='Total', linewidth=2)
    ax.plot(epochs, [h['action_loss'] for h in history], label='Action', alpha=0.7)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Loss Components')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, [h['accuracy'] for h in history], label='Train', linewidth=2)
    ax.plot(epochs, [h['val_acc'] for h in history], label='Val', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, [h['near_binary_frac'] for h in history], linewidth=2, color='teal')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Fraction near {0,1}')
    ax.set_title('Bottleneck Binarization'); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, [h['bimodal_loss'] for h in history], label='Weighted', linewidth=2)
    ax.plot(epochs, [h['bimodal_raw'] for h in history], label='Raw', alpha=0.7)
    ax.plot(epochs, [h['bimodal_weight'] for h in history], label='Weight', linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Bimodality')
    ax.set_title('Bimodality Loss & Weight'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, [h['alpha_mean'] for h in history], linewidth=2, color='purple')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mean α'); ax.set_title('Bottleneck Sharpness')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, [h['feature_density'] for h in history], label='SAE density', linewidth=2)
    ax.plot(epochs, [h['bottleneck_mean'] for h in history], label='Bottleneck mean', alpha=0.7)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Density'); ax.set_title('Feature Sparsity')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


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
    features = data['features']
    actions = data['actions']

    # Resolve env metadata from file or CLI
    env_name = data.get('env_name', args.env_name or 'unknown')
    env_type = data.get('env_type', args.env_type or 'unknown')
    stored_action_names = data.get('action_names', None)

    n_actions = int(actions.max().item()) + 1
    action_names = get_action_names(n_actions, stored_action_names, env_type)

    print(f"  Raw features: {features.shape}, range=[{features.min():.2f}, {features.max():.2f}]")
    print(f"  Env: {env_name} ({env_type}), actions: {n_actions} → {action_names}")

    # --- Normalize ---
    stage1_data = None
    if args.stage1_path and os.path.exists(args.stage1_path):
        print(f"  Loading Stage 1 from {args.stage1_path}...")
        stage1_data = torch.load(args.stage1_path, weights_only=False)
        feat_mean = stage1_data['feature_mean']
        feat_std = stage1_data['feature_std']
        features = (features - feat_mean) / feat_std
        print(f"  Normalized: range=[{features.min():.2f}, {features.max():.2f}]")
    else:
        feat_mean = features.mean(0)
        feat_std = features.std(0).clamp(min=1e-6)
        features = (features - feat_mean) / feat_std

    # --- Split ---
    n_train = int(0.9 * len(features))
    idx = torch.randperm(len(features), generator=torch.Generator().manual_seed(args.seed))
    train_ds = TensorDataset(features[idx[:n_train]], actions[idx[:n_train]])
    val_ds = TensorDataset(features[idx[n_train:]], actions[idx[n_train:]])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    # Add after train_loader/val_loader creation
    print("\n  === RAW FEATURE LINEAR PROBE ===")
    _probe = nn.Linear(features.shape[1], n_actions).to(device)
    _opt = torch.optim.Adam(_probe.parameters(), lr=1e-3)
    for _ in range(100):
        for bx, ba in train_loader:
            bx, ba = bx.to(device), ba.to(device)
            _opt.zero_grad()
            F.cross_entropy(_probe(bx), ba).backward()
            _opt.step()
    _correct = 0
    with torch.no_grad():
        for bx, ba in val_loader:
            bx, ba = bx.to(device), ba.to(device)
            _correct += (_probe(bx).argmax(1) == ba).sum().item()
    print(f"  Raw 512-d feature linear probe val acc: {_correct/len(val_ds):.3f}")
    print("\n  === 128-d STAGE1 FEATURE LINEAR PROBE ===")
    _probe2 = nn.Linear(features.shape[1], n_actions).to(device)
    _opt2 = torch.optim.Adam(_probe2.parameters(), lr=1e-3)
    for _ in range(100):
        for bx, ba in train_loader:
            bx, ba = bx.to(device), ba.to(device)
            _opt2.zero_grad()
            F.cross_entropy(_probe2(bx), ba).backward()
            _opt2.step()
    with torch.no_grad():
        correct2 = sum(
            (_probe2(bx.to(device)).argmax(1) == ba.to(device)).sum().item()
            for bx, ba in val_loader
        )
    print(f"  128-d Stage1 feature linear probe val acc: {correct2/len(val_ds):.3f}")

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"\n  Class distribution:")
    for i, name in enumerate(action_names):
        c = (actions == i).sum().item()
        print(f"    {name}: {c} ({100*c/len(actions):.1f}%)")

    # --- Action class weights ---
    if args.action_class_weights is not None:
        if len(args.action_class_weights) != n_actions:
            raise ValueError(
                f"--action_class_weights has {len(args.action_class_weights)} values "
                f"but env has {n_actions} actions."
            )
        action_class_weights = tuple(args.action_class_weights)
    else:
        action_class_weights = tuple([1.0] * n_actions)

    # --- Config ---
    config = SAELogicConfig(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        k=args.k,
        n_actions=n_actions,
        n_clauses_per_action=args.n_clauses_per_action,
        sae_pretrain_epochs=args.sae_pretrain_epochs,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        seed=args.seed,
        bimodal_max=args.bimodal_max,
        bimodal_warmup=args.bimodal_warmup,
        bimodal_ramp=args.bimodal_ramp,
        l0_penalty_weight=args.l0_penalty,
        lambda_sparsity=args.lambda_sparsity,
        sae_lr=args.sae_lr,
        logic_lr=args.logic_lr,
        bottleneck_lr=args.bottleneck_lr,
        action_class_weights=action_class_weights,
        max_grad_norm=args.max_grad_norm,
        beta_action=args.beta_action,
        env_name=env_name,
        env_type=env_type,
    )
    if args.no_ica_init:
        config.use_ica_init = False
    # --- Model ---
    print("\nInitializing model...")
    model = SAELogicAgentV3(config, device=device)
    model.set_normalization(feat_mean, feat_std)

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

    print(f"\nArchitecture: {config.input_dim} → SAE({config.hidden_dim}, k={config.k}) → "
          f"FixedNorm → Sigmoid → Logic({config.n_clauses_per_action} clauses/action) → {n_actions}")
    print(f"  SAE pretrain : {config.sae_pretrain_epochs} epochs")
    print(f"  Logic train  : {config.n_epochs - config.sae_pretrain_epochs} epochs")
    print(f"  Bimodality   : warmup={config.bimodal_warmup}, ramp={config.bimodal_ramp}, max={config.bimodal_max}")

    # ── Stage 1: Pre-train SAE ───────────────────────────────────────────────
    pretrain_sae(model, train_loader, config, device)

    # ── Freeze SAE + compute normalization ──────────────────────────────────
    model.freeze_sae()
    model.compute_z_normalization(train_loader)
    model.to(device)

    # ── Stage 2: Train logic ─────────────────────────────────────────────────
    best_state, best_acc, history = train_logic(
        model, train_loader, val_loader, config, device, action_names
    )

    print(f"\n{'='*70}")
    print(f"Training complete! Best val accuracy: {best_acc:.3f}")
    print(f"{'='*70}")

    if best_state is not None:
        model.load_state_dict(best_state['model'])
    else:
        best_state = {'model': model.state_dict(), 'epoch': config.n_epochs - 1, 'val_acc': best_acc}

    # --- Analysis ---
    print(f"\n{'='*70}")
    print("PER-CLASS ACCURACY (val)")
    print(f"{'='*70}")
    per_class_accuracy(model, val_loader, device, action_names)

    linear_probe(model, train_loader, val_loader, device)

    # --- Rules ---
    print(f"\n{'='*70}")
    print("LEARNED RULES (DNF)")
    print(f"{'='*70}")
    rules = model.extract_rules(action_names=action_names)
    for aname, clauses in rules.items():
        print(f"\n{aname} ←")
        for i, c in enumerate(dict.fromkeys(clauses)):
            print(f"    {c}")
            if i < len(set(clauses)) - 1:
                print("  ∨")

    # --- Binarization ---
    print(f"\n{'='*70}")
    print("BINARIZATION QUALITY")
    print(f"{'='*70}")
    model.eval()
    with torch.no_grad():
        all_z = []
        for bx, _ in val_loader:
            bx = bx.to(device)
            zs, _ = model.sae.encode(bx)
            zn = model.normalize_z(zs)
            zb = model.bottleneck(zn)
            all_z.append(zb.cpu())
        all_z = torch.cat(all_z, 0)
        print(f"  Near {{0,1}} (±0.05): {((all_z < 0.05) | (all_z > 0.95)).float().mean():.3f}")
        print(f"  Near 0.5 (±0.1):    {((all_z > 0.4) & (all_z < 0.6)).float().mean():.3f}")
        print(f"  Mean: {all_z.mean():.4f}")
        print(f"  Sharpness α: {model.bottleneck.get_sharpness().mean():.1f}")

    # --- Save ---
    save_path = os.path.join(args.save_dir, "sae_logic_v3_model.pt")
    torch.save({
        'model_state': best_state['model'],
        'config': asdict(config),
        'rules': rules,
        'best_val_acc': best_acc,
        'feature_mean': feat_mean,
        'feature_std': feat_std,
        'z_mean': model.z_mean.cpu(),
        'z_std': model.z_std.cpu(),
        'action_names': action_names,
        'env_name': env_name,
        'env_type': env_type,
    }, save_path)

    with open(os.path.join(args.save_dir, "learned_rules.json"), 'w') as f:
        json.dump(rules, f, indent=2)

    plot_training_history(history, args.save_dir)

    stats = model.logic_layer.count_active_rules(action_names=action_names)
    print(f"\n{'='*70}")
    print("RULE STATISTICS")
    print(f"{'='*70}")
    print(f"  Total clauses      : {stats['total_clauses']}")
    print(f"  Non-empty          : {stats['non_empty_clauses']}")
    print(f"  Avg literals/clause: {stats['avg_literals_per_clause']:.2f}")
    for a, c in stats['clauses_per_action'].items():
        print(f"    {a}: {c}")

    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAE + Product T-Norm Logic V3 (MiniGrid + Atari)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MiniGrid
  python train_sae_logic_v3.py \\
      --features_path ./stage1_outputs/collected_data.pt \\
      --stage1_path   ./stage1_outputs/stage1_outputs.pt \\
      --hidden_dim 300 --k 50

  # Atari Breakout (4 actions)
  python train_sae_logic_v3.py \\
      --features_path ./stage1_atari/collected_data.pt \\
      --stage1_path   ./stage1_atari/stage1_outputs.pt \\
      --hidden_dim 1024 --k 100

  # Atari with explicit class weights
  python train_sae_logic_v3.py \\
      --features_path ./stage1_atari/collected_data.pt \\
      --action_class_weights 1.0 2.0 1.0 1.0
        """
    )

    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--stage1_path", type=str, default=None)

    # Env metadata (auto-detected from collected_data.pt if present)
    parser.add_argument("--env_name", type=str, default=None,
                        help="Override env name (usually auto-detected from features file)")
    parser.add_argument("--env_type", type=str, default=None, choices=["minigrid", "atari"],
                        help="Override env type (usually auto-detected from features file)")

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=300,
                        help="SAE hidden dim (MiniGrid: 300, Atari: 1024 recommended)")
    parser.add_argument("--k", type=int, default=50,
                        help="SAE top-k sparsity (MiniGrid: 50, Atari: 100 recommended)")
    parser.add_argument("--n_clauses_per_action", type=int, default=10)

    # Training
    parser.add_argument("--sae_pretrain_epochs", type=int, default=50)
    parser.add_argument("--n_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ica_init", action="store_true", default=False)
    # LRs
    parser.add_argument("--sae_lr", type=float, default=1e-3)
    parser.add_argument("--logic_lr", type=float, default=3e-3)
    parser.add_argument("--bottleneck_lr", type=float, default=1e-3)

    # Loss weights
    parser.add_argument("--beta_action", type=float, default=5.0)
    parser.add_argument("--bimodal_max", type=float, default=0.3)
    parser.add_argument("--bimodal_warmup", type=int, default=30)
    parser.add_argument("--bimodal_ramp", type=int, default=80)
    parser.add_argument("--l0_penalty", type=float, default=1e-4)
    parser.add_argument("--lambda_sparsity", type=float, default=5e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)

    # Per-action class weights — variable length, auto-infers n_actions
    parser.add_argument("--action_class_weights", type=float, nargs="+", default=None,
                        help="Per-action loss weights. Length must match env's n_actions. "
                             "Default: uniform 1.0 for all actions.")

    parser.add_argument("--save_dir", type=str, default="./sae_logic_v3_outputs")

    args = parser.parse_args()

    main(args)