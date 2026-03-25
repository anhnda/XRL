"""
Stage 2: Overcomplete Sparse Autoencoder Training
===================================================
Trains an overcomplete SAE with:
  - ICA-anchored decoder initialization (from Stage 1)
  - TopK sparsity with straight-through estimator
  - Decoder column unit-norm constraint
  - Interaction-aware action predictor: W_a * c + W_int * phi(c)
  - Signal subspace regularization (from Stage 1 V_k)
  - Dead unit resampling
  - Full joint loss:
      L = α·recon + β·action_CE + λ1·L1(z) + λ2·L1(W_int) + λ3·L_signal + λ4·L_dead

Usage:
    # Single run
    python sparse_concept_autoencoder.py \\
        --features_path ./stage1_outputs/collected_data.pt \\
        --stage1_path ./stage1_outputs/stage1_outputs.pt \\
        --hidden_dim 512 --k 10 --n_epochs 100

    # Multi-run for consensus (Stage 3)
    python sparse_concept_autoencoder.py \\
        --features_path ./stage1_outputs/collected_data.pt \\
        --stage1_path ./stage1_outputs/stage1_outputs.pt \\
        --n_runs 10 --save_dir ./stage2_outputs
"""

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SAEConfig:
    """All hyperparameters for Stage 2."""
    # Architecture
    input_dim: int = 128
    hidden_dim: int = 512          # Overcomplete: D >> d  (e.g. 4x-8x input_dim)
    k: int = 10                    # TopK sparsity per sample
    n_actions: int = 7

    # Interaction predictor
    max_interaction_pairs: int = 0  # 0 = auto (all pairs among active concepts)
    interaction_top_n: int = 50     # Only keep top-N concept pairs by activation frequency

    # Loss weights
    alpha: float = 1.0             # Reconstruction
    beta: float = 1.0              # Action prediction CE
    lambda1: float = 1e-3          # L1 on concept activations z
    lambda2: float = 1e-3          # L1 on interaction weights W_int
    lambda3: float = 0.1           # Signal subspace regularization
    lambda4: float = 1.0           # Dead unit auxiliary loss weight

    # Dead unit resampling
    dead_threshold: float = 1e-3   # Activation freq below this = dead
    resample_every: int = 500      # Steps between resampling checks
    resample_warmup: int = 1000    # Don't resample before this many steps

    # Training
    n_epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 42

    # ICA initialization
    use_ica_init: bool = True
    ica_stability_threshold: float = 0.8  # Only use ICA components above this stability

    # Logging
    log_every: int = 10            # Epochs between detailed logging
    save_dir: str = "./stage2_outputs"


# ============================================================================
# Model: Overcomplete Sparse Autoencoder
# ============================================================================

class OvercompleteSAE(nn.Module):
    """
    Overcomplete Sparse Autoencoder with TopK sparsity.

    Architecture:
        Encode: z = TopK(ReLU(W_e (x - b_d) + b_e))
        Decode: x_hat = W_d z + b_d

    Decoder columns are constrained to unit norm.
    """

    def __init__(self, input_dim: int, hidden_dim: int, k: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k

        # Encoder: (d -> D)
        self.W_e = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.b_e = nn.Parameter(torch.zeros(hidden_dim))

        # Decoder: (D -> d)
        self.W_d = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.b_d = nn.Parameter(torch.zeros(input_dim))

        # Default initialization (overridden by init_from_stage1 if used)
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.W_d)

        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self._normalize_decoder()

    def _normalize_decoder(self):
        """Normalize decoder columns to unit norm."""
        norms = self.W_d.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_d.data.div_(norms)

    def encode_pre_topk(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output before TopK (for auxiliary losses)."""
        return F.relu(F.linear(x - self.b_d, self.W_e, self.b_e))

    def topk(self, h: torch.Tensor) -> torch.Tensor:
        """TopK with straight-through estimator."""
        # Get top-k values and indices
        topk_vals, topk_idx = torch.topk(h, self.k, dim=-1)

        # Create sparse output
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)

        # Straight-through: forward uses mask, backward flows through h
        return h * mask

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_sparse : (batch, D) sparse concept activations after TopK
            z_pre    : (batch, D) pre-TopK activations (for L1 loss)
        """
        z_pre = self.encode_pre_topk(x)
        z_sparse = self.topk(z_pre)
        return z_sparse, z_pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # W_d: (d, D). We want z @ W_d^T which gives (batch, d).
        # But F.linear(z, W) computes z @ W^T, so we need W = W_d.
        # However W_d is (d, D) and F.linear expects weight (out_features, in_features) = (d, D).
        # F.linear(z, W_d) = z @ W_d^T = (batch, D) @ (D, d) = (batch, d). Correct.
        return F.linear(z, self.W_d, self.b_d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_hat    : (batch, d) reconstruction
            z_sparse : (batch, D) sparse concepts
            z_pre    : (batch, D) pre-TopK activations
        """
        z_sparse, z_pre = self.encode(x)
        x_hat = self.decode(z_sparse)
        return x_hat, z_sparse, z_pre


# ============================================================================
# Interaction-Aware Action Predictor
# ============================================================================

class InteractionActionPredictor(nn.Module):
    """
    Action predictor: a_hat = softmax(W_a c + W_int phi(c) + b_a)

    where phi(c) contains pairwise products c_i * c_j for selected pairs.
    Since concepts are non-negative (ReLU + TopK), c_i * c_j > 0 iff both active.

    The L1 penalty on W_int encourages sparse interactions → short rules.
    """

    def __init__(self, hidden_dim: int, n_actions: int, interaction_pairs: Optional[torch.Tensor] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        # Linear part: W_a c + b_a
        self.W_a = nn.Linear(hidden_dim, n_actions)

        # Interaction part
        if interaction_pairs is not None and len(interaction_pairs) > 0:
            self.register_buffer("interaction_pairs", interaction_pairs)  # (n_pairs, 2)
            self.n_pairs = len(interaction_pairs)
            self.W_int = nn.Linear(self.n_pairs, n_actions, bias=False)
            # Initialize interaction weights to zero (let L1 keep them sparse)
            nn.init.zeros_(self.W_int.weight)
        else:
            self.interaction_pairs = None
            self.n_pairs = 0
            self.W_int = None

    def compute_interactions(self, c: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute pairwise products for selected concept pairs."""
        if self.interaction_pairs is None:
            return None
        # c: (batch, D), pairs: (n_pairs, 2)
        i_idx = self.interaction_pairs[:, 0]  # (n_pairs,)
        j_idx = self.interaction_pairs[:, 1]  # (n_pairs,)
        phi = c[:, i_idx] * c[:, j_idx]       # (batch, n_pairs)
        return phi

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Returns action logits (batch, n_actions)."""
        logits = self.W_a(c)

        if self.W_int is not None:
            phi = self.compute_interactions(c)
            logits = logits + self.W_int(phi)

        return logits

    def get_interaction_l1(self) -> torch.Tensor:
        """L1 norm of interaction weights for regularization."""
        if self.W_int is not None:
            return self.W_int.weight.abs().sum()
        return torch.tensor(0.0, device=self.W_a.weight.device)


# ============================================================================
# Initialization from Stage 1
# ============================================================================

def init_from_stage1(model: OvercompleteSAE, stage1_data: dict, config: SAEConfig):
    """
    Initialize SAE decoder columns using ICA directions from Stage 1.

    First k_stable columns = stable ICA directions (stability > threshold)
    Remaining columns = random unit vectors in signal subspace
    Encoder = transpose of decoder (approximate inverse)
    """
    d = config.input_dim
    D = config.hidden_dim

    ica_directions = stage1_data["ica_directions"]    # (d, k_ica)
    ica_stability = stage1_data["ica_stability"]       # (k_ica,)
    ica_rank = stage1_data["ica_rank"]                 # (k_ica,)
    V_k = stage1_data["V_k"]                          # (d, k_svd)
    feature_mean = stage1_data["feature_mean"]         # (d,)

    # Select stable ICA components
    stable_mask = ica_stability > config.ica_stability_threshold
    stable_indices = torch.where(stable_mask)[0]
    n_stable = len(stable_indices)

    print(f"\n  ICA init: {n_stable} stable components (threshold={config.ica_stability_threshold})")

    # Get the device the model lives on
    model_device = model.W_d.device

    with torch.no_grad():
        W_d = model.W_d.data  # (d, D) — on model_device

        # --- Fill first n_stable columns with stable ICA directions ---
        if n_stable > 0:
            # Use ICA directions ranked by reliability
            ranked_stable = []
            for idx in ica_rank:
                if stable_mask[idx]:
                    ranked_stable.append(idx.item())
                if len(ranked_stable) >= min(n_stable, D):
                    break

            for col, ica_idx in enumerate(ranked_stable):
                direction = ica_directions[:, ica_idx].to(model_device)
                W_d[:, col] = direction / (direction.norm() + 1e-8)

        # --- Fill remaining columns with random directions in signal subspace ---
        n_remaining = D - min(n_stable, D)
        if n_remaining > 0 and V_k.shape[1] > 0:
            k_svd = V_k.shape[1]
            # Random coefficients in PCA space, project to original space
            random_coeffs = torch.randn(k_svd, n_remaining)
            V_k_dev = V_k.to(model_device)
            random_dirs = V_k_dev @ random_coeffs.to(model_device)  # (d, n_remaining)
            # Normalize
            norms = random_dirs.norm(dim=0, keepdim=True).clamp(min=1e-8)
            random_dirs = random_dirs / norms
            W_d[:, min(n_stable, D):] = random_dirs
        else:
            # Fallback: random unit vectors
            random_dirs = torch.randn(d, n_remaining, device=model_device)
            norms = random_dirs.norm(dim=0, keepdim=True).clamp(min=1e-8)
            W_d[:, min(n_stable, D):] = random_dirs / norms

        # Normalize all decoder columns
        model._normalize_decoder()

        # Initialize encoder as approximate transpose of decoder
        model.W_e.data = model.W_d.data.t().clone()

        # Initialize decoder bias to feature mean
        model.b_d.data = feature_mean.clone().to(model_device)

    print(f"  Decoder: {n_stable} ICA + {n_remaining} random signal-subspace directions")
    print(f"  Encoder: transposed decoder")
    print(f"  Decoder bias: feature mean")


# ============================================================================
# Interaction pair selection
# ============================================================================

def select_interaction_pairs(features: torch.Tensor, model: OvercompleteSAE,
                             top_n: int = 50, min_coactivation: float = 0.01,
                             device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Select concept pairs for the interaction predictor.

    Strategy: run a forward pass on a sample of data, find pairs of concepts
    that frequently co-activate (both in top-K simultaneously). These are
    candidates for meaningful conjunctions.

    Args:
        features : (N, d) features
        model    : trained or initialized SAE
        top_n    : max number of pairs to keep
        min_coactivation : minimum co-activation frequency to consider

    Returns:
        pairs : (n_pairs, 2) long tensor of concept index pairs
    """
    model.eval()
    n_sample = min(5000, len(features))
    x = features[:n_sample].to(device)

    with torch.no_grad():
        _, z_sparse, _ = model(x)
        active = (z_sparse > 0).float()  # (N, D)

    # Co-activation matrix: how often are concepts i and j both active?
    coact = (active.t() @ active) / n_sample  # (D, D)

    # Zero diagonal (no self-interactions)
    coact.fill_diagonal_(0.0)

    # Only upper triangle (avoid duplicates)
    coact = torch.triu(coact, diagonal=1)

    # Filter by minimum frequency
    valid = coact > min_coactivation
    if valid.sum() == 0:
        print("  Warning: no concept pairs above min_coactivation threshold")
        # Fallback: take top pairs regardless
        valid = coact > 0

    # Get top-N pairs by co-activation frequency
    flat_coact = coact.flatten()
    n_pairs = min(top_n, (flat_coact > 0).sum().item())

    if n_pairs == 0:
        print("  Warning: no active concept pairs found, returning empty")
        return torch.zeros(0, 2, dtype=torch.long)

    _, flat_idx = torch.topk(flat_coact, n_pairs)
    D = coact.shape[0]
    rows = flat_idx // D
    cols = flat_idx % D
    pairs = torch.stack([rows, cols], dim=1).cpu()  # (n_pairs, 2)

    model.train()

    print(f"  Selected {len(pairs)} interaction pairs (from {(coact > 0).sum().item()} active pairs)")
    if len(pairs) > 0:
        top5_coact = flat_coact[flat_idx[:5]]
        print(f"  Top-5 co-activation rates: {[f'{v:.3f}' for v in top5_coact.tolist()]}")

    return pairs


# ============================================================================
# Dead unit resampling
# ============================================================================

class DeadUnitTracker:
    """Tracks concept activation frequency and resamples dead units."""

    def __init__(self, hidden_dim: int, threshold: float = 1e-3, device: torch.device = torch.device("cpu")):
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.device = device

        # Exponential moving average of activation frequency
        self.ema_freq = torch.zeros(hidden_dim, device=device)
        self.ema_decay = 0.99
        self.n_updates = 0

    def update(self, z_sparse: torch.Tensor):
        """Update activation frequency EMA."""
        with torch.no_grad():
            batch_freq = (z_sparse > 0).float().mean(dim=0)
            if self.n_updates == 0:
                self.ema_freq = batch_freq
            else:
                self.ema_freq = self.ema_decay * self.ema_freq + (1 - self.ema_decay) * batch_freq
            self.n_updates += 1

    def get_dead_units(self) -> torch.Tensor:
        """Return indices of dead units."""
        return torch.where(self.ema_freq < self.threshold)[0]

    def resample(self, model: OvercompleteSAE, x_batch: torch.Tensor, x_recon: torch.Tensor):
        """
        Resample dead units: re-initialize their decoder column toward the
        direction of highest reconstruction error in the current batch.
        """
        dead = self.get_dead_units()
        if len(dead) == 0:
            return 0

        with torch.no_grad():
            # Find samples with highest reconstruction error
            errors = (x_batch - x_recon).pow(2).sum(dim=1)  # (batch,)
            n_resample = min(len(dead), len(x_batch))

            # Top error samples
            _, top_idx = torch.topk(errors, min(n_resample * 2, len(errors)))

            for i, dead_idx in enumerate(dead[:n_resample]):
                # Pick a high-error sample
                sample_idx = top_idx[i % len(top_idx)]
                residual = x_batch[sample_idx] - x_recon[sample_idx]

                # Set decoder column to residual direction
                direction = residual / (residual.norm() + 1e-8)
                model.W_d.data[:, dead_idx] = direction

                # Set encoder row to match
                model.W_e.data[dead_idx, :] = direction

                # Reset encoder bias to small positive value (encourage activation)
                model.b_e.data[dead_idx] = 0.1

                # Reset EMA for this unit
                self.ema_freq[dead_idx] = 0.05  # Give it a chance

        n_resampled = min(n_resample, len(dead))
        return n_resampled


# ============================================================================
# Signal subspace regularization
# ============================================================================

def signal_subspace_loss(model: OvercompleteSAE, V_k: torch.Tensor) -> torch.Tensor:
    """
    Penalize decoder columns that live in the noise subspace.

    L_signal = mean_j (1 - ||V_k^T d_j||^2 / ||d_j||^2)

    Where d_j is the j-th decoder column and V_k spans the signal subspace.
    Since decoder columns are unit-norm, ||d_j||^2 = 1.
    """
    # W_d: (d, D), V_k: (d, k)
    # Project each decoder column onto signal subspace
    projections = V_k.t() @ model.W_d  # (k, D)
    signal_energy = projections.pow(2).sum(dim=0)  # (D,) — energy in signal subspace per column
    # decoder columns are unit norm, so total energy ≈ 1
    noise_fraction = 1.0 - signal_energy  # fraction in noise subspace
    return noise_fraction.clamp(min=0).mean()


# ============================================================================
# Training loop
# ============================================================================

def train_sae(features: torch.Tensor, actions: torch.Tensor,
              stage1_data: dict, config: SAEConfig,
              run_id: int = 0) -> dict:
    """
    Train one SAE run.

    Returns:
        dict with trained model, action predictor, metrics, config
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    seed = config.seed + run_id * 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"\n{'#'*60}")
    print(f"  STAGE 2: SAE TRAINING (run {run_id}, seed {seed})")
    print(f"  Architecture: {config.input_dim} -> {config.hidden_dim} (x{config.hidden_dim/config.input_dim:.1f})")
    print(f"  TopK = {config.k}, n_epochs = {config.n_epochs}")
    print(f"  Loss: α={config.alpha} β={config.beta} λ1={config.lambda1} λ2={config.lambda2} λ3={config.lambda3}")
    print(f"{'#'*60}")

    # --- Normalize features ---
    feat_mean = stage1_data["feature_mean"].to(device)
    feat_std = stage1_data["feature_std"].to(device)
    V_k = stage1_data["V_k"].to(device)

    features_norm = (features - feat_mean.cpu()) / feat_std.cpu()

    # --- Create model ---
    model = OvercompleteSAE(config.input_dim, config.hidden_dim, config.k).to(device)

    # ICA initialization
    if config.use_ica_init:
        # Need to also normalize ICA directions to match normalized feature space
        # ICA directions are in original space; we need them in normalized space
        # If x_norm = (x - mu) / sigma, then direction v in original space
        # becomes v / sigma (elementwise) in normalized space, then re-normalize
        stage1_norm = {}
        stage1_norm.update(stage1_data)

        ica_dirs = stage1_data["ica_directions"].clone()  # (d, k_ica)
        # Transform: divide each row by the corresponding std, then normalize columns
        ica_dirs_norm = ica_dirs / feat_std.cpu().unsqueeze(1)
        col_norms = ica_dirs_norm.norm(dim=0, keepdim=True).clamp(min=1e-8)
        ica_dirs_norm = ica_dirs_norm / col_norms
        stage1_norm["ica_directions"] = ica_dirs_norm

        # Same for V_k
        V_k_raw = stage1_data["V_k"].clone()  # (d, k_svd)
        V_k_norm = V_k_raw / feat_std.cpu().unsqueeze(1)
        col_norms = V_k_norm.norm(dim=0, keepdim=True).clamp(min=1e-8)
        V_k_norm = V_k_norm / col_norms
        stage1_norm["V_k"] = V_k_norm

        # Feature mean in normalized space is zero
        stage1_norm["feature_mean"] = torch.zeros(config.input_dim)

        init_from_stage1(model, stage1_norm, config)

        # Ensure all model params are on correct device after init
        model = model.to(device)

        V_k = V_k_norm.to(device)
    else:
        print("  Skipping ICA initialization (use_ica_init=False)")

    # Defensive: ensure model is fully on device
    model = model.to(device)

    # --- Select interaction pairs ---
    print("\n  Selecting interaction pairs...")
    interaction_pairs = select_interaction_pairs(
        features_norm, model, top_n=config.interaction_top_n, device=device
    )

    # --- Create action predictor ---
    action_predictor = InteractionActionPredictor(
        config.hidden_dim, config.n_actions, interaction_pairs
    ).to(device)

    print(f"  Action predictor: linear ({config.hidden_dim} -> {config.n_actions})"
          f" + {len(interaction_pairs)} interaction pairs")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(action_predictor.parameters()),
        lr=config.lr
    )

    # --- DataLoader ---
    dataset = TensorDataset(features_norm, actions)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        generator=generator,
        worker_init_fn=lambda wid: np.random.seed(seed + wid)
    )

    # --- Dead unit tracker ---
    dead_tracker = DeadUnitTracker(config.hidden_dim, config.dead_threshold, device)

    # --- Training metrics ---
    history = {
        "recon_loss": [], "action_loss": [], "total_loss": [],
        "action_acc": [], "sparsity": [],
        "n_active": [], "n_dead": [], "n_resampled": [],
        "signal_loss": [], "interaction_l1": [],
    }

    global_step = 0

    for epoch in range(config.n_epochs):
        model.train()
        action_predictor.train()

        epoch_recon = 0.0
        epoch_action = 0.0
        epoch_total = 0.0
        epoch_correct = 0
        epoch_samples = 0
        epoch_resampled = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.n_epochs}", leave=False)

        for batch_x, batch_a in pbar:
            batch_x = batch_x.to(device)
            batch_a = batch_a.to(device)

            # --- Forward ---
            x_hat, z_sparse, z_pre = model(batch_x)

            # --- Losses ---

            # 1. Reconstruction
            recon_loss = F.mse_loss(x_hat, batch_x)

            # 2. Action prediction
            action_logits = action_predictor(z_sparse)
            action_loss = F.cross_entropy(action_logits, batch_a)

            # 3. L1 sparsity on pre-TopK activations
            l1_loss = z_pre.abs().mean()

            # 4. L1 on interaction weights
            int_l1 = action_predictor.get_interaction_l1()

            # 5. Signal subspace regularization
            sig_loss = signal_subspace_loss(model, V_k)

            # 6. Total
            total_loss = (
                config.alpha * recon_loss
                + config.beta * action_loss
                + config.lambda1 * l1_loss
                + config.lambda2 * int_l1
                + config.lambda3 * sig_loss
            )

            # --- Backward ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # --- Enforce decoder unit norm ---
            with torch.no_grad():
                model._normalize_decoder()

            # --- Dead unit tracking & resampling ---
            dead_tracker.update(z_sparse.detach())
            n_resampled = 0
            if (global_step >= config.resample_warmup
                    and global_step % config.resample_every == 0):
                n_resampled = dead_tracker.resample(model, batch_x, x_hat.detach())
                epoch_resampled += n_resampled

            global_step += 1

            # --- Tracking ---
            pred = action_logits.argmax(dim=-1)
            epoch_correct += (pred == batch_a).sum().item()
            epoch_samples += len(batch_a)
            epoch_recon += recon_loss.item()
            epoch_action += action_loss.item()
            epoch_total += total_loss.item()

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "act": f"{action_loss.item():.4f}",
                "acc": f"{100*epoch_correct/epoch_samples:.1f}%",
            })

        # --- Epoch summary ---
        n_batches = len(dataloader)
        epoch_acc = 100.0 * epoch_correct / epoch_samples

        history["recon_loss"].append(epoch_recon / n_batches)
        history["action_loss"].append(epoch_action / n_batches)
        history["total_loss"].append(epoch_total / n_batches)
        history["action_acc"].append(epoch_acc)
        history["n_resampled"].append(epoch_resampled)

        # --- Detailed logging ---
        if (epoch + 1) % config.log_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                sample_x = features_norm[:2000].to(device)
                sample_a = actions[:2000].to(device)
                _, z_sp, _ = model(sample_x)

                active_rate = (z_sp > 0).float().mean(dim=0)
                n_active = (active_rate > 0.01).sum().item()
                n_dead = (active_rate < config.dead_threshold).sum().item()
                avg_sparsity = (z_sp > 0).float().mean().item()
                n_saturated = (active_rate > 0.9).sum().item()

                # Signal loss on sample
                s_loss = signal_subspace_loss(model, V_k).item()

                # Interaction L1
                i_l1 = action_predictor.get_interaction_l1().item()

                # Non-zero interaction weights
                if action_predictor.W_int is not None:
                    n_nonzero_int = (action_predictor.W_int.weight.abs() > 0.01).sum().item()
                    total_int = action_predictor.W_int.weight.numel()
                else:
                    n_nonzero_int = 0
                    total_int = 0

            history["sparsity"].append(avg_sparsity)
            history["n_active"].append(n_active)
            history["n_dead"].append(n_dead)
            history["signal_loss"].append(s_loss)
            history["interaction_l1"].append(i_l1)

            print(f"\n  Epoch {epoch+1}/{config.n_epochs}:")
            print(f"    Recon loss   : {epoch_recon/n_batches:.5f}")
            print(f"    Action loss  : {epoch_action/n_batches:.5f}")
            print(f"    Action acc   : {epoch_acc:.2f}%")
            print(f"    Active       : {n_active}/{config.hidden_dim} "
                  f"(dead={n_dead}, saturated={n_saturated})")
            print(f"    Sparsity     : {avg_sparsity*100:.1f}% "
                  f"(target ~{config.k/config.hidden_dim*100:.1f}%)")
            print(f"    Signal loss  : {s_loss:.5f}")
            print(f"    Interact L1  : {i_l1:.5f} "
                  f"(nonzero weights: {n_nonzero_int}/{total_int})")
            if epoch_resampled > 0:
                print(f"    Resampled    : {epoch_resampled} dead units")

            # Per-action accuracy
            if (epoch + 1) % (config.log_every * 2) == 0:
                model.eval()
                with torch.no_grad():
                    all_preds, all_targets = [], []
                    for bx, ba in dataloader:
                        bx = bx.to(device)
                        _, zs, _ = model(bx)
                        logits = action_predictor(zs)
                        all_preds.append(logits.argmax(dim=-1).cpu())
                        all_targets.append(ba)
                    all_preds = torch.cat(all_preds)
                    all_targets = torch.cat(all_targets)

                    action_names = ["TurnLeft", "TurnRight", "Forward",
                                    "Pickup", "Drop", "Toggle", "Done"]
                    print("    Per-action accuracy:")
                    for a in range(config.n_actions):
                        mask = all_targets == a
                        if mask.sum() > 0:
                            acc = (all_preds[mask] == a).float().mean().item()
                            print(f"      {action_names[a]:10s}: {acc*100:5.1f}% "
                                  f"({mask.sum().item():5d} samples)")

            model.train()

    # --- Final eval ---
    model.eval()
    with torch.no_grad():
        all_x = features_norm.to(device)
        _, z_all, _ = model(all_x)
        active_rate = (z_all > 0).float().mean(dim=0)

    return {
        "model": model.cpu(),
        "action_predictor": action_predictor.cpu(),
        "interaction_pairs": interaction_pairs,
        "history": history,
        "config": asdict(config),
        "seed": seed,
        "run_id": run_id,
        "active_rate": active_rate.cpu(),
        "feat_mean": feat_mean.cpu(),
        "feat_std": feat_std.cpu(),
    }


# ============================================================================
# Save / Load
# ============================================================================

def save_run(result: dict, save_dir: str, run_id: int = 0):
    """Save a single training run."""
    run_dir = os.path.join(save_dir, f"run_{run_id:03d}")
    os.makedirs(run_dir, exist_ok=True)

    model = result["model"]
    action_pred = result["action_predictor"]

    # Model weights
    torch.save({
        "W_e": model.W_e.data,
        "b_e": model.b_e.data,
        "W_d": model.W_d.data,
        "b_d": model.b_d.data,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "k": model.k,
    }, os.path.join(run_dir, "sae_weights.pt"))

    # Action predictor
    ap_data = {
        "W_a_weight": action_pred.W_a.weight.data,
        "W_a_bias": action_pred.W_a.bias.data,
        "interaction_pairs": result["interaction_pairs"],
    }
    if action_pred.W_int is not None:
        ap_data["W_int_weight"] = action_pred.W_int.weight.data
    torch.save(ap_data, os.path.join(run_dir, "action_predictor.pt"))

    # Training history & metadata
    meta = {
        "config": result["config"],
        "seed": result["seed"],
        "run_id": result["run_id"],
        "final_recon_loss": result["history"]["recon_loss"][-1],
        "final_action_loss": result["history"]["action_loss"][-1],
        "final_action_acc": result["history"]["action_acc"][-1],
        "n_active_concepts": int((result["active_rate"] > 0.01).sum().item()),
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Active rates
    torch.save({
        "active_rate": result["active_rate"],
        "feat_mean": result["feat_mean"],
        "feat_std": result["feat_std"],
    }, os.path.join(run_dir, "analysis.pt"))

    # History as numpy for easy plotting
    history_np = {k: np.array(v) for k, v in result["history"].items()}
    np.savez(os.path.join(run_dir, "history.npz"), **history_np)

    print(f"  Run {run_id} saved to {run_dir}")
    return run_dir


def load_run(run_dir: str, device: torch.device = torch.device("cpu")) -> dict:
    """Load a saved run."""
    # SAE
    sae_data = torch.load(os.path.join(run_dir, "sae_weights.pt"),
                          map_location=device, weights_only=True)
    model = OvercompleteSAE(sae_data["input_dim"], sae_data["hidden_dim"], sae_data["k"])
    model.W_e.data = sae_data["W_e"]
    model.b_e.data = sae_data["b_e"]
    model.W_d.data = sae_data["W_d"]
    model.b_d.data = sae_data["b_d"]
    model = model.to(device)

    # Action predictor
    ap_data = torch.load(os.path.join(run_dir, "action_predictor.pt"),
                         map_location=device, weights_only=True)
    pairs = ap_data["interaction_pairs"]
    action_pred = InteractionActionPredictor(
        sae_data["hidden_dim"], 7, pairs  # TODO: get n_actions from metadata
    ).to(device)
    action_pred.W_a.weight.data = ap_data["W_a_weight"]
    action_pred.W_a.bias.data = ap_data["W_a_bias"]
    if "W_int_weight" in ap_data and action_pred.W_int is not None:
        action_pred.W_int.weight.data = ap_data["W_int_weight"]

    # Metadata
    with open(os.path.join(run_dir, "metadata.json")) as f:
        meta = json.load(f)

    # Analysis
    analysis = torch.load(os.path.join(run_dir, "analysis.pt"),
                          map_location=device, weights_only=True)

    return {
        "model": model,
        "action_predictor": action_pred,
        "interaction_pairs": pairs,
        "metadata": meta,
        "active_rate": analysis["active_rate"],
        "feat_mean": analysis["feat_mean"],
        "feat_std": analysis["feat_std"],
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_training(history: dict, config: dict, save_path: str):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Stage 2: SAE Training (D={config['hidden_dim']}, K={config['k']})",
        fontsize=14, fontweight="bold"
    )

    epochs = np.arange(1, len(history["recon_loss"]) + 1)

    # 1. Losses
    ax = axes[0, 0]
    ax.plot(epochs, history["recon_loss"], label="Reconstruction", linewidth=2)
    ax.plot(epochs, history["action_loss"], label="Action CE", linewidth=2)
    ax.plot(epochs, history["total_loss"], label="Total", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Action accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["action_acc"], color="green", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Action Prediction Accuracy")
    ax.grid(True, alpha=0.3)

    # 3. Active / dead concepts (sampled at log_every intervals)
    ax = axes[0, 2]
    if history["n_active"]:
        log_epochs = np.linspace(1, len(history["recon_loss"]),
                                 len(history["n_active"]), dtype=int)
        ax.plot(log_epochs, history["n_active"], label="Active (>1%)", linewidth=2)
        ax.plot(log_epochs, history["n_dead"], label="Dead", linewidth=2, color="red")
        ax.axhline(config["hidden_dim"], color="gray", linestyle="--", alpha=0.3,
                    label=f"Total={config['hidden_dim']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count")
    ax.set_title("Concept Utilization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Sparsity
    ax = axes[1, 0]
    if history["sparsity"]:
        log_epochs = np.linspace(1, len(history["recon_loss"]),
                                 len(history["sparsity"]), dtype=int)
        ax.plot(log_epochs, [s * 100 for s in history["sparsity"]],
                linewidth=2, color="purple")
        target = config["k"] / config["hidden_dim"] * 100
        ax.axhline(target, color="red", linestyle="--", alpha=0.5,
                    label=f"Target={target:.1f}%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active %")
    ax.set_title("Concept Sparsity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Signal subspace loss
    ax = axes[1, 1]
    if history["signal_loss"]:
        log_epochs = np.linspace(1, len(history["recon_loss"]),
                                 len(history["signal_loss"]), dtype=int)
        ax.plot(log_epochs, history["signal_loss"], linewidth=2, color="teal")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Signal Loss")
    ax.set_title("Signal Subspace Regularization")
    ax.grid(True, alpha=0.3)

    # 6. Interaction L1
    ax = axes[1, 2]
    if history["interaction_l1"]:
        log_epochs = np.linspace(1, len(history["recon_loss"]),
                                 len(history["interaction_l1"]), dtype=int)
        ax.plot(log_epochs, history["interaction_l1"], linewidth=2, color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 norm")
    ax.set_title("Interaction Weight L1")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training plot saved: {save_path}")


# ============================================================================
# Multi-run entry point (for Stage 3 consensus)
# ============================================================================

def run_stage2(features: torch.Tensor, actions: torch.Tensor,
               stage1_data: dict, config: SAEConfig,
               n_runs: int = 1) -> list:
    """
    Train SAE for n_runs with different seeds.

    Returns:
        list of result dicts from train_sae()
    """
    os.makedirs(config.save_dir, exist_ok=True)
    all_results = []

    for run_id in range(n_runs):
        result = train_sae(features, actions, stage1_data, config, run_id=run_id)

        # Save
        run_dir = save_run(result, config.save_dir, run_id)

        # Plot
        plot_training(
            result["history"], result["config"],
            os.path.join(run_dir, "training_curves.png")
        )

        all_results.append(result)

        print(f"\n  Run {run_id} complete: "
              f"recon={result['history']['recon_loss'][-1]:.5f}, "
              f"action_acc={result['history']['action_acc'][-1]:.2f}%, "
              f"active={int((result['active_rate'] > 0.01).sum().item())}/{config.hidden_dim}")

    # Save multi-run summary
    if n_runs > 1:
        summary = {
            "n_runs": n_runs,
            "runs": []
        }
        for r in all_results:
            summary["runs"].append({
                "run_id": r["run_id"],
                "seed": r["seed"],
                "final_recon": r["history"]["recon_loss"][-1],
                "final_action_acc": r["history"]["action_acc"][-1],
                "n_active": int((r["active_rate"] > 0.01).sum().item()),
            })
        with open(os.path.join(config.save_dir, "multi_run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Multi-run summary saved to {config.save_dir}/multi_run_summary.json")

    return all_results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Overcomplete SAE Training")

    # Data
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to collected_data.pt (features + actions)")
    parser.add_argument("--stage1_path", type=str, required=True,
                        help="Path to stage1_outputs.pt")

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_actions", type=int, default=7)
    parser.add_argument("--interaction_top_n", type=int, default=50)

    # Loss weights
    parser.add_argument("--alpha", type=float, default=1.0, help="Reconstruction weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Action prediction weight")
    parser.add_argument("--lambda1", type=float, default=1e-3, help="L1 sparsity weight")
    parser.add_argument("--lambda2", type=float, default=1e-3, help="Interaction L1 weight")
    parser.add_argument("--lambda3", type=float, default=0.1, help="Signal subspace weight")

    # Dead units
    parser.add_argument("--dead_threshold", type=float, default=1e-3)
    parser.add_argument("--resample_every", type=int, default=500)

    # Training
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)

    # ICA init
    parser.add_argument("--no_ica_init", action="store_true")
    parser.add_argument("--ica_stability_threshold", type=float, default=0.8)

    # Multi-run
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of runs with different seeds (for Stage 3 consensus)")

    parser.add_argument("--save_dir", type=str, default="./stage2_outputs")

    args = parser.parse_args()

    # Load data
    print(f"Loading features from {args.features_path}...")
    raw = torch.load(args.features_path, map_location="cpu", weights_only=False)
    features = raw["features"]
    actions = raw["actions"]
    print(f"  Features: {features.shape}, Actions: {actions.shape}")

    print(f"Loading Stage 1 outputs from {args.stage1_path}...")
    stage1_data = torch.load(args.stage1_path, map_location="cpu", weights_only=False)
    print(f"  Signal dim k={stage1_data['k']}, ICA directions: {stage1_data['ica_directions'].shape}")

    # Config
    config = SAEConfig(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        k=args.k,
        n_actions=args.n_actions,
        interaction_top_n=args.interaction_top_n,
        alpha=args.alpha,
        beta=args.beta,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        dead_threshold=args.dead_threshold,
        resample_every=args.resample_every,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        log_every=args.log_every,
        use_ica_init=not args.no_ica_init,
        ica_stability_threshold=args.ica_stability_threshold,
        save_dir=args.save_dir,
    )

    # Train
    results = run_stage2(features, actions, stage1_data, config, n_runs=args.n_runs)

    print(f"\n{'='*60}")
    print(f"STAGE 2 COMPLETE — {args.n_runs} run(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()