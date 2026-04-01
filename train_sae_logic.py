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
    6. Clause diversity penalty: eliminates duplicate clauses per action
    7. Per-action class weights: forces rule learning for rare actions
    8. Fully vectorized penalties — no Python loops over actions/clauses
    9. DataLoader with num_workers + pin_memory for fast data throughput
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparse_concept_autoencoder import OvercompleteSAE, SAEConfig, init_from_stage1


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
        n_clauses_per_action: int = 5,
        l0_penalty_weight: float = 1e-3,
    ):
        super().__init__()
        self.n_features          = n_features
        self.n_actions           = n_actions
        self.n_clauses_per_action = n_clauses_per_action
        self.l0_penalty_weight   = l0_penalty_weight

        total_clauses = n_actions * n_clauses_per_action
        self.w_pos = nn.Parameter(torch.randn(total_clauses, n_features) * 0.01 - 3.0)
        self.w_neg = nn.Parameter(torch.randn(total_clauses, n_features) * 0.01 - 3.0)
        self.clause_weight = nn.Parameter(torch.ones(total_clauses) * 2.0)

    def _get_selection_probs(self):
        """
        Returns p, n each of shape (total_clauses, n_features).
        Mathematically identical to original — just a 3-way softmax.
        """
        absent = torch.zeros_like(self.w_pos)
        logits = torch.stack([self.w_pos, self.w_neg, absent], dim=-1)
        probs  = F.softmax(logits, dim=-1)
        return probs[..., 0], probs[..., 1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized forward — identical math to original,
        no Python loops.

        Args:
            features: (batch, n_features) in [0, 1]

        Returns:
            action_logits: (batch, n_actions)
        """
        batch_size = features.shape[0]
        p, n = self._get_selection_probs()  # (total_clauses, n_features)

        # Broadcast: (batch, total_clauses, n_features)
        f = features.unsqueeze(1)
        p_e = p.unsqueeze(0)
        n_e = n.unsqueeze(0)

        literals       = p_e * f + n_e * (1.0 - f) + (1.0 - p_e - n_e)
        log_clause_sum = torch.log(literals + 1e-8).sum(dim=-1)          # (batch, total_clauses)
        clauses        = torch.sigmoid(log_clause_sum + self.clause_weight)  # (batch, total_clauses)

        # (batch, n_actions, n_clauses_per_action) → sum → (batch, n_actions)
        action_logits = clauses.view(batch_size, self.n_actions, self.n_clauses_per_action).sum(dim=-1)
        return action_logits

    def complexity_penalty(self) -> torch.Tensor:
        """Vectorized L0 penalty — identical value to original."""
        p, n = self._get_selection_probs()
        return self.l0_penalty_weight * (p + n).mean()

    def diversity_penalty(self) -> torch.Tensor:
        """
        Vectorized clause diversity penalty — identical value to original
        Python-loop version, now computed in a single bmm call.
        """
        p, n    = self._get_selection_probs()
        usage   = p + n                                                    # (total_clauses, n_features)
        nc      = self.n_clauses_per_action

        U       = usage.view(self.n_actions, nc, self.n_features)          # (A, nc, F)
        norms   = U.norm(dim=2, keepdim=True).clamp(min=1e-8)
        U_norm  = U / norms                                                # (A, nc, F)
        sim     = torch.bmm(U_norm, U_norm.transpose(1, 2))               # (A, nc, nc)

        # Upper-triangle mask (shared across all actions)
        mask = torch.triu(
            torch.ones(nc, nc, device=usage.device, dtype=torch.bool), diagonal=1
        )
        return sim[:, mask].mean()

    def extract_rules(
        self,
        feature_names: Optional[List[str]] = None,
        action_names:  Optional[List[str]] = None,
        threshold: float = 0.3,
    ) -> Dict[str, List[str]]:
        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(self.n_features)]
        if action_names is None:
            action_names  = [f"action_{a}" for a in range(self.n_actions)]

        p, n        = self._get_selection_probs()
        p           = p.detach().cpu().numpy()
        n           = n.detach().cpu().numpy()
        clause_bias = self.clause_weight.detach().cpu().numpy()

        rules = {}
        for a in range(self.n_actions):
            clauses = []
            for c in range(self.n_clauses_per_action):
                idx      = a * self.n_clauses_per_action + c
                literals = []
                for i in range(self.n_features):
                    if p[idx, i] > threshold:
                        literals.append(feature_names[i])
                    elif n[idx, i] > threshold:
                        literals.append(f"¬{feature_names[i]}")
                if literals:
                    clauses.append(f"({' ∧ '.join(literals)}) [bias={clause_bias[idx]:.2f}]")

            rules[action_names[a]] = clauses if clauses else ["(no active clauses)"]
        return rules

    def count_active_rules(
        self,
        threshold: float = 0.3,
        action_names: Optional[List[str]] = None,
    ) -> Dict:
        if action_names is None:
            action_names = [f"action_{a}" for a in range(self.n_actions)]

        p, n = self._get_selection_probs()
        p    = p.detach().cpu().numpy()
        n    = n.detach().cpu().numpy()

        total_clauses = non_empty = total_literals = 0
        per_action = {}

        for a in range(self.n_actions):
            action_clauses = 0
            for c in range(self.n_clauses_per_action):
                idx       = a * self.n_clauses_per_action + c
                n_lits    = ((p[idx] > threshold) | (n[idx] > threshold)).sum()
                total_clauses += 1
                if n_lits > 0:
                    non_empty     += 1
                    total_literals += n_lits
                    action_clauses += 1
            per_action[action_names[a]] = action_clauses

        return {
            'total_clauses':          total_clauses,
            'non_empty_clauses':      non_empty,
            'avg_literals_per_clause': total_literals / max(non_empty, 1),
            'clauses_per_action':     per_action,
        }


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SAELogicConfig:
    input_dim: int            = 128
    hidden_dim: int           = 256
    k: int                    = 10
    n_actions: int            = 4

    initial_alpha: float      = 1.0

    n_clauses_per_action: int = 5
    l0_penalty_weight: float  = 1e-3
    lambda_diversity: float   = 0.1

    alpha_recon: float        = 0.001
    beta_action: float        = 10.0
    lambda_sparsity: float    = 1e-4
    lambda_bimodal: float     = 0.0
    bimodal_max: float        = 1.0
    bimodal_warmup: int       = 60
    bimodal_ramp: int         = 40

    action_class_weights: tuple = (1.0, 1.0, 2.0, 2.0)

    training_mode: str        = "joint"
    sae_freeze_epoch: int     = 100

    n_epochs: int             = 200
    batch_size: int           = 256
    sae_lr: float             = 1e-3
    logic_lr: float           = 3e-3
    bottleneck_lr: float      = 1e-3
    seed: int                 = 42

    use_ica_init: bool        = True
    log_every: int            = 10
    save_dir: str             = "./sae_logic_v2_outputs"


# ============================================================================
# Integrated Model
# ============================================================================

class SAELogicAgentV2(nn.Module):
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
        self.register_buffer('feature_std',  torch.ones(config.input_dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean.to(x.device)) / self.feature_std.to(x.device)

    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = False,
        return_features: bool = False,
    ):
        if normalize:
            x = self.normalize(x)

        z_sparse, z_pre = self.sae.encode(x)
        z_binary        = self.bottleneck(z_sparse)
        action_logits   = self.logic_layer(z_binary)

        if return_features:
            return action_logits, {
                'z_sparse': z_sparse,
                'z_pre':    z_pre,
                'z_binary': z_binary,
                'x_recon':  self.sae.decode(z_sparse),
            }
        return action_logits

    def compute_loss(self, x: torch.Tensor, actions: torch.Tensor, epoch: int = 0):
        action_logits, feats = self.forward(x, return_features=True)

        class_weights = torch.tensor(
            self.config.action_class_weights, dtype=torch.float32, device=x.device
        )
        action_loss   = F.cross_entropy(action_logits, actions, weight=class_weights)
        recon_loss    = F.mse_loss(feats['x_recon'], x)
        sparsity_loss = self.config.lambda_sparsity * feats['z_pre'].abs().mean()

        z_bin        = feats['z_binary']
        bimodal_raw  = (z_bin * (1.0 - z_bin)).mean()

        if epoch < self.config.bimodal_warmup:
            bimodal_weight = 0.0
        else:
            progress = min(1.0,
                (epoch - self.config.bimodal_warmup) / max(self.config.bimodal_ramp, 1))
            bimodal_weight = self.config.bimodal_max * progress
        bimodal_loss = bimodal_weight * bimodal_raw

        logic_complexity = self.logic_layer.complexity_penalty()
        diversity_loss   = self.config.lambda_diversity * self.logic_layer.diversity_penalty()

        if self.config.training_mode == "joint" and epoch >= self.config.sae_freeze_epoch:
            total_loss = (
                self.config.beta_action * action_loss +
                bimodal_loss +
                logic_complexity +
                diversity_loss
            )
        else:
            total_loss = (
                self.config.alpha_recon * recon_loss +
                self.config.beta_action * action_loss +
                sparsity_loss +
                bimodal_loss +
                logic_complexity +
                diversity_loss
            )

        acc         = (action_logits.argmax(1) == actions).float().mean()
        near_binary = ((z_bin < 0.05) | (z_bin > 0.95)).float().mean()

        info = {
            'total_loss':       total_loss.item(),
            'action_loss':      action_loss.item(),
            'recon_loss':       recon_loss.item(),
            'sparsity_loss':    sparsity_loss.item(),
            'bimodal_loss':     bimodal_loss.item(),
            'bimodal_weight':   bimodal_weight,
            'bimodal_raw':      bimodal_raw.item(),
            'logic_complexity': logic_complexity.item(),
            'diversity_loss':   diversity_loss.item(),
            'accuracy':         acc.item(),
            'near_binary_frac': near_binary.item(),
            'feature_density':  (feats['z_sparse'] > 0).float().mean().item(),
            'bottleneck_mean':  z_bin.mean().item(),
            'bottleneck_std':   z_bin.std().item(),
            'alpha_mean':       self.bottleneck.get_sharpness().mean().item(),
        }
        return total_loss, info

    def extract_rules(
        self,
        concept_labels: Optional[List[str]] = None,
        action_names:   Optional[List[str]] = None,
    ) -> Dict:
        return self.logic_layer.extract_rules(
            feature_names=concept_labels,
            action_names=action_names,
        )


# ============================================================================
# DataLoader factory
# ============================================================================

def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Fast DataLoader with pinned memory and multiple workers."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


# ============================================================================
# Training helpers
# ============================================================================

def avg_dict(dicts):
    keys = dicts[0].keys()
    return {k: np.mean([d[k] for d in dicts]) for k in keys}


def evaluate(model: SAELogicAgentV2, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_a in loader:
            batch_x, batch_a = batch_x.to(device, non_blocking=True), \
                                batch_a.to(device, non_blocking=True)
            preds    = model(batch_x).argmax(1)
            correct += (preds == batch_a).sum().item()
            total   += batch_a.size(0)
    return correct / total


# ============================================================================
# Training Functions
# ============================================================================

def train_joint(
    model: SAELogicAgentV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SAELogicConfig,
    device: str,
):
    print("\n" + "=" * 70)
    print("JOINT TRAINING")
    print("=" * 70)

    optimizer = torch.optim.Adam([
        {'params': model.sae.parameters(),         'lr': config.sae_lr},
        {'params': model.bottleneck.parameters(),   'lr': config.bottleneck_lr},
        {'params': model.logic_layer.parameters(),  'lr': config.logic_lr},
    ])

    best_val_acc    = 0.0
    best_model_state = None
    history         = []

    for epoch in range(config.n_epochs):

        if epoch == 0:
            with torch.no_grad():
                batch_x, _ = next(iter(train_loader))
                batch_x    = batch_x.to(device)
                z_sparse, _ = model.sae.encode(batch_x)
                z_binary    = model.bottleneck(z_sparse)
                print(f"\n[Debug] Epoch 0 feature stats:")
                print(f"  Input : min={batch_x.min():.3f}  max={batch_x.max():.3f}  mean={batch_x.mean():.4f}")
                print(f"  SAE z : min={z_sparse.min():.3f}  max={z_sparse.max():.3f}  mean={z_sparse.mean():.4f}")
                print(f"  Bottle: min={z_binary.min():.3f}  max={z_binary.max():.3f}  mean={z_binary.mean():.4f}")
                print(f"  NearBin={((z_binary<0.05)|(z_binary>0.95)).float().mean():.3f}  "
                      f"density={( z_sparse>0).float().mean():.4f}\n")

        if epoch == config.sae_freeze_epoch:
            print(f"\n[Info] Freezing SAE at epoch {epoch}")
            for p in model.sae.parameters():
                p.requires_grad = False

        model.train()
        train_info = []

        for batch_x, batch_a in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_a = batch_a.to(device, non_blocking=True)

            loss, info = model.compute_loss(batch_x, batch_a, epoch)

            optimizer.zero_grad()
            loss.backward()

            if epoch < config.sae_freeze_epoch:
                with torch.no_grad():
                    model.sae._normalize_decoder()

            optimizer.step()
            train_info.append(info)

        val_acc = evaluate(model, val_loader, device)
        history.append({**avg_dict(train_info), 'val_acc': val_acc, 'epoch': epoch})

        if (epoch + 1) % config.log_every == 0:
            avg = avg_dict(train_info)
            print(
                f"  Epoch {epoch+1:3d}/{config.n_epochs} | "
                f"Loss: {avg['total_loss']:.4f} | "
                f"Act: {avg['action_loss']:.4f} | "
                f"Rec: {avg['recon_loss']:.4f} | "
                f"Acc: {avg['accuracy']:.3f} | "
                f"Val: {val_acc:.3f} | "
                f"Bin: {avg['near_binary_frac']:.3f} | "
                f"α: {avg['alpha_mean']:.1f} | "
                f"Bim: {avg['bimodal_weight']:.2f} | "
                f"Div: {avg['diversity_loss']:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = {'model': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}

    return best_model_state, best_val_acc, history


def train_two_stage(
    model: SAELogicAgentV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SAELogicConfig,
    device: str,
):
    print("\n" + "=" * 70)
    print("TWO-STAGE TRAINING")
    print("=" * 70)

    history      = []
    n_sae_epochs = config.n_epochs // 2

    # --- Stage A: SAE + bottleneck ---
    print("\n[Stage A] Pre-training SAE + bottleneck...")
    sae_optimizer = torch.optim.Adam(
        list(model.sae.parameters()) + list(model.bottleneck.parameters()),
        lr=config.sae_lr
    )

    for epoch in range(n_sae_epochs):
        model.sae.train(); model.bottleneck.train()
        epoch_losses = []

        for batch_x, _ in train_loader:
            batch_x  = batch_x.to(device, non_blocking=True)
            z_sparse, z_pre = model.sae.encode(batch_x)
            x_recon  = model.sae.decode(z_sparse)
            z_binary = model.bottleneck(z_sparse)

            recon_loss    = F.mse_loss(x_recon, batch_x)
            sparsity_loss = config.lambda_sparsity * z_pre.abs().mean()

            bimodal_raw = (z_binary * (1.0 - z_binary)).mean()
            if epoch < config.bimodal_warmup:
                bimodal_w = 0.0
            else:
                progress  = min(1.0, (epoch - config.bimodal_warmup) / max(config.bimodal_ramp, 1))
                bimodal_w = config.bimodal_max * progress

            loss = config.alpha_recon * recon_loss + sparsity_loss + bimodal_w * bimodal_raw

            sae_optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                model.sae._normalize_decoder()
            sae_optimizer.step()
            epoch_losses.append(loss.item())

        if (epoch + 1) % config.log_every == 0:
            nb = ((z_binary < 0.05) | (z_binary > 0.95)).float().mean().item()
            print(f"  Epoch {epoch+1}/{n_sae_epochs} | Loss: {np.mean(epoch_losses):.4f} | "
                  f"Recon: {recon_loss.item():.4f} | NearBin: {nb:.3f} | BimodalW: {bimodal_w:.3f}")

    # --- Stage B: Logic layer ---
    print("\n[Stage B] Training logic layer...")
    for p in model.sae.parameters():
        p.requires_grad = False

    logic_optimizer = torch.optim.Adam([
        {'params': model.logic_layer.parameters(), 'lr': config.logic_lr},
        {'params': model.bottleneck.parameters(),  'lr': config.bottleneck_lr * 0.1},
    ])

    best_val_acc     = 0.0
    best_model_state = None

    for epoch in range(n_sae_epochs, config.n_epochs):
        model.logic_layer.train(); model.bottleneck.train()
        train_info = []

        for batch_x, batch_a in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_a = batch_a.to(device, non_blocking=True)
            loss, info = model.compute_loss(batch_x, batch_a, epoch)
            logic_optimizer.zero_grad()
            loss.backward()
            logic_optimizer.step()
            train_info.append(info)

        val_acc = evaluate(model, val_loader, device)
        history.append({**avg_dict(train_info), 'val_acc': val_acc, 'epoch': epoch})

        if (epoch + 1) % config.log_every == 0:
            avg = avg_dict(train_info)
            print(f"  Epoch {epoch+1}/{config.n_epochs} | Loss: {avg['total_loss']:.4f} | "
                  f"Acc: {avg['accuracy']:.3f} | Val: {val_acc:.3f} | "
                  f"Bin: {avg['near_binary_frac']:.3f} | Div: {avg['diversity_loss']:.4f}")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = {'model': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}

    return best_model_state, best_val_acc, history


# ============================================================================
# Linear probe
# ============================================================================

def linear_probe(model, train_loader, val_loader, device, n_epochs=50, lr=1e-3):
    print(f"\n{'='*70}\nLINEAR PROBE (information ceiling test)\n{'='*70}")
    model.eval()

    train_z, train_a, val_z, val_a = [], [], [], []
    with torch.no_grad():
        for bx, ba in train_loader:
            z = model.bottleneck(model.sae.encode(bx.to(device))[0]).cpu()
            train_z.append(z); train_a.append(ba)
        for bx, ba in val_loader:
            z = model.bottleneck(model.sae.encode(bx.to(device))[0]).cpu()
            val_z.append(z); val_a.append(ba)

    train_z = torch.cat(train_z); train_a = torch.cat(train_a)
    val_z   = torch.cat(val_z);   val_a   = torch.cat(val_a)

    nb = ((train_z < 0.05) | (train_z > 0.95)).float().mean()
    print(f"  Bottleneck: {train_z.shape[1]}-d, near-binary={nb:.3f}")

    probe     = nn.Linear(model.config.hidden_dim, model.config.n_actions).to(device)
    opt       = torch.optim.Adam(probe.parameters(), lr=lr)
    p_loader  = DataLoader(TensorDataset(train_z, train_a), batch_size=256, shuffle=True)

    for _ in range(n_epochs):
        probe.train()
        for bz, ba in p_loader:
            loss = F.cross_entropy(probe(bz.to(device)), ba.to(device))
            opt.zero_grad(); loss.backward(); opt.step()

    probe.eval()
    with torch.no_grad():
        tr_acc = (probe(train_z.to(device)).argmax(1).cpu() == train_a).float().mean().item()
        va_acc = (probe(val_z.to(device)).argmax(1).cpu()   == val_a  ).float().mean().item()

    print(f"  Train acc: {tr_acc:.3f}  Val acc: {va_acc:.3f}")
    if va_acc > 0.7:
        print("  → Bottleneck has enough info. Logic layer is the bottleneck.")
    else:
        print("  → Bottleneck lost too much info. SAE/bottleneck needs improvement.")
    return tr_acc, va_acc


# ============================================================================
# Plot
# ============================================================================

def plot_training_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("SAE + Product T-Norm Logic Training", fontsize=14, fontweight="bold")

    panels = [
        (axes[0,0], [('total_loss','Total'),('action_loss','Action'),('recon_loss','Recon')],
         'Loss Components', 'Loss'),
        (axes[0,1], [('accuracy','Train'),('val_acc','Val')],
         'Accuracy', 'Accuracy'),
        (axes[0,2], [('near_binary_frac',None)],
         'Bottleneck Binarization', 'Fraction near {0,1}'),
        (axes[0,3], [('diversity_loss',None)],
         'Clause Diversity Loss ↓', 'Diversity loss'),
        (axes[1,0], [('bimodal_loss','Weighted'),('bimodal_raw','Raw'),('bimodal_weight','Weight')],
         'Bimodality Loss & Weight', 'Bimodality'),
        (axes[1,1], [('alpha_mean',None)],
         'Sigmoid Sharpness', 'Mean α'),
        (axes[1,2], [('feature_density','SAE density'),('bottleneck_mean','Bottle mean')],
         'Feature Sparsity', 'Density'),
        (axes[1,3], [('logic_complexity',None)],
         'Logic Complexity (L0)', 'Complexity'),
    ]

    for ax, series, title, ylabel in panels:
        for key, label in series:
            vals = [h[key] for h in history]
            ax.plot(epochs, vals, label=label, linewidth=2) if label else \
            ax.plot(epochs, vals, linewidth=2)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        if any(label for _, label in series):
            ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {path}")


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
    data     = torch.load(args.features_path, weights_only=False)
    features = data['features']
    actions  = data['actions']
    print(f"  Raw features: {features.shape}, range=[{features.min():.2f}, {features.max():.2f}]")

    raw_counts = torch.bincount(actions, minlength=7)
    raw_names  = ["TurnLeft","TurnRight","Forward","Pickup","Drop","Toggle","Done"]
    print("  Raw action distribution:")
    for name, cnt in zip(raw_names, raw_counts):
        print(f"    {name:10s}: {cnt:6d} ({100.*cnt/len(actions):.1f}%)")

    # --- Normalize ---
    stage1_data = None
    if args.stage1_path and os.path.exists(args.stage1_path):
        print(f"  Loading Stage 1 from {args.stage1_path}...")
        stage1_data = torch.load(args.stage1_path, weights_only=False)
        feat_mean   = stage1_data['feature_mean']
        feat_std    = stage1_data['feature_std']
    else:
        print("  WARNING: No stage1_path. Computing normalization from data.")
        feat_mean = features.mean(dim=0)
        feat_std  = features.std(dim=0).clamp(min=1e-6)

    features = (features - feat_mean) / feat_std
    print(f"  Normalized: range=[{features.min():.2f}, {features.max():.2f}], mean={features.mean():.4f}")

    # --- Filter to active actions ---
    KEEP         = {1, 2, 3, 5}
    REMAP        = {1: 0, 2: 1, 3: 2, 5: 3}
    ACTION_NAMES = ["TurnRight", "Forward", "Pickup", "Toggle"]

    mask     = torch.tensor([a.item() in KEEP for a in actions])
    features = features[mask]
    actions  = torch.tensor([REMAP[a.item()] for a in actions[mask]])

    print(f"\n  After filtering: {len(features)} samples, {actions.max().item()+1} actions")
    filtered_counts = torch.bincount(actions, minlength=4)
    for name, cnt in zip(ACTION_NAMES, filtered_counts):
        bar = "█" * int(40 * cnt / len(actions))
        print(f"    {name:10s}: {cnt:6d} ({100.*cnt/len(actions):5.1f}%) {bar}")

    # --- Train/val split ---
    n_actions = int(actions.max().item() + 1)
    n_train   = int(0.9 * len(features))
    indices   = torch.randperm(len(features), generator=torch.Generator().manual_seed(args.seed))

    train_dataset = TensorDataset(features[indices[:n_train]], actions[indices[:n_train]])
    val_dataset   = TensorDataset(features[indices[n_train:]], actions[indices[n_train:]])

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader   = make_loader(val_dataset,   args.batch_size, shuffle=False)

    print(f"\n  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # --- Config ---
    config = SAELogicConfig(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        k=args.k,
        n_actions=n_actions,
        n_clauses_per_action=args.n_clauses_per_action,
        training_mode=args.mode,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        seed=args.seed,
        bimodal_max=args.bimodal_max,
        bimodal_warmup=args.bimodal_warmup,
        bimodal_ramp=args.bimodal_ramp,
        l0_penalty_weight=args.l0_penalty,
        lambda_sparsity=args.lambda_sparsity,
        lambda_diversity=args.lambda_diversity,
        alpha_recon=args.alpha_recon,
        beta_action=args.beta_action,
        sae_lr=args.sae_lr,
        logic_lr=args.logic_lr,
        bottleneck_lr=args.bottleneck_lr,
        sae_freeze_epoch=args.sae_freeze_epoch,
        action_class_weights=tuple(args.action_class_weights[:n_actions]),
    )

    # --- Model ---
    print("\nInitializing model...")
    model = SAELogicAgentV2(config, device=device)
    model.set_normalization(feat_mean.to(device), feat_std.to(device))

    if stage1_data is not None and config.use_ica_init:
        print("Initializing SAE from Stage 1 ICA...")
        init_from_stage1(model.sae, stage1_data, SAEConfig(
            input_dim=config.input_dim, hidden_dim=config.hidden_dim,
            k=config.k, n_actions=config.n_actions, use_ica_init=True,
        ))

    print(f"\n  {config.input_dim}d → SAE({config.hidden_dim}, k={config.k})"
          f" → Logic({config.n_clauses_per_action}×{n_actions}) → {n_actions} actions")
    print(f"  alpha_recon={config.alpha_recon}  beta_action={config.beta_action}")
    print(f"  class_weights={dict(zip(ACTION_NAMES, config.action_class_weights))}")
    print(f"  lambda_diversity={config.lambda_diversity}  l0={config.l0_penalty_weight}")

    # --- Train ---
    if args.mode == "two_stage":
        best_state, best_acc, history = train_two_stage(model, train_loader, val_loader, config, device)
    else:
        best_state, best_acc, history = train_joint(model, train_loader, val_loader, config, device)

    print(f"\n{'='*70}\nBest validation accuracy: {best_acc:.3f}\n{'='*70}")

    if best_state:
        model.load_state_dict(best_state['model'])
    else:
        best_state = {'model': model.state_dict(), 'epoch': config.n_epochs-1, 'val_acc': best_acc}

    # --- Linear probe ---
    linear_probe(model, train_loader, val_loader, device)

    # --- Rules ---
    print("\nExtracting rules...")
    rules = model.extract_rules(action_names=ACTION_NAMES)
    print(f"\n{'='*70}\nLEARNED RULES (DNF Form)\n{'='*70}")
    for action_name, clauses in rules.items():
        print(f"\n{action_name} ←")
        for i, clause in enumerate(list(dict.fromkeys(clauses))):
            print(f"    {clause}")
            if i < len(list(dict.fromkeys(clauses))) - 1:
                print("  ∨")

    # --- Binarization check ---
    print(f"\n{'='*70}\nBINARIZATION QUALITY\n{'='*70}")
    model.eval()
    all_z = []
    with torch.no_grad():
        for bx, _ in val_loader:
            z = model.bottleneck(model.sae.encode(bx.to(device))[0]).cpu()
            all_z.append(z)
    all_z   = torch.cat(all_z)
    near_01 = ((all_z < 0.05) | (all_z > 0.95)).float().mean()
    near_05 = ((all_z > 0.40) & (all_z < 0.60)).float().mean()
    print(f"  Near {{0,1}}: {near_01:.3f}  Ambiguous: {near_05:.3f}")
    print(f"  Mean: {all_z.mean():.4f}  Alpha: {model.bottleneck.get_sharpness().mean():.1f}")

    # --- Save ---
    save_path = os.path.join(args.save_dir, "sae_logic_v2_model.pt")
    torch.save({
        'model_state':  best_state['model'],
        'config':       asdict(config),
        'rules':        rules,
        'best_val_acc': best_acc,
        'feature_mean': feat_mean,
        'feature_std':  feat_std,
        'action_remap': REMAP,
        'action_names': ACTION_NAMES,
    }, save_path)

    with open(os.path.join(args.save_dir, "learned_rules.json"), 'w') as f:
        json.dump(rules, f, indent=2)

    plot_training_history(history, args.save_dir)

    stats = model.logic_layer.count_active_rules(action_names=ACTION_NAMES)
    print(f"\n{'='*70}\nRULE STATISTICS\n{'='*70}")
    print(f"  Total: {stats['total_clauses']}  Non-empty: {stats['non_empty_clauses']}"
          f"  Avg literals: {stats['avg_literals_per_clause']:.2f}")
    for a, c in stats['clauses_per_action'].items():
        print(f"    {a}: {c}")
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--features_path",        type=str,   required=True)
    parser.add_argument("--stage1_path",           type=str,   default=None)
    parser.add_argument("--hidden_dim",            type=int,   default=256)
    parser.add_argument("--k",                     type=int,   default=10)
    parser.add_argument("--n_clauses_per_action",  type=int,   default=5)
    parser.add_argument("--mode",                  type=str,   default="joint",
                        choices=["two_stage","joint"])
    parser.add_argument("--n_epochs",              type=int,   default=200)
    parser.add_argument("--batch_size",            type=int,   default=256)
    parser.add_argument("--seed",                  type=int,   default=42)
    parser.add_argument("--sae_lr",                type=float, default=1e-3)
    parser.add_argument("--logic_lr",              type=float, default=3e-3)
    parser.add_argument("--bottleneck_lr",         type=float, default=1e-3)
    parser.add_argument("--alpha_recon",           type=float, default=0.001)
    parser.add_argument("--beta_action",           type=float, default=10.0)
    parser.add_argument("--bimodal_max",           type=float, default=1.0)
    parser.add_argument("--bimodal_warmup",        type=int,   default=60)
    parser.add_argument("--bimodal_ramp",          type=int,   default=40)
    parser.add_argument("--l0_penalty",            type=float, default=1e-3)
    parser.add_argument("--lambda_sparsity",       type=float, default=1e-4)
    parser.add_argument("--lambda_diversity",      type=float, default=0.1)
    parser.add_argument("--sae_freeze_epoch",      type=int,   default=100)
    parser.add_argument("--action_class_weights",  type=float, nargs="+",
                        default=[1.0, 1.0, 2.0, 2.0],
                        help="TurnRight Forward Pickup Toggle")
    parser.add_argument("--save_dir",              type=str,   default="./sae_logic_v2_outputs")

    args = parser.parse_args()
    main(args)