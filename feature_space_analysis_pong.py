"""
feature_analysis_pong.py
========================
Stage 1 feature-space analysis for the Pong PPO agent trained with AtariCNN.

The custom AtariCNN in train_pong.py outputs 512-dim features via:
    CNN → Flatten → Linear(3136, 512) → ReLU

This script:
  1. Loads ppo_pong.zip (or best_pong/best_model.zip) with the custom extractor
  2. Rolls out episodes on ALE/Pong-v5 and hooks the feature extractor
  3. Runs SVD + ICA + normalization stats + action-distribution analysis
  4. Saves stage1_outputs.pt + stage1_summary.json + diagnostic plots

Usage:
    python feature_analysis_pong.py                          # defaults
    python feature_analysis_pong.py --model_path best_pong/best_model.zip
    python feature_analysis_pong.py --n_episodes 300 --save_dir ./pong_stage1
    python feature_analysis_pong.py --features_path ./pong_stage1/collected_data.pt
"""

import argparse
import os
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis as scipy_kurtosis

# ── Pong constants ────────────────────────────────────────────────────────────

ENV_NAME    = "ALE/Pong-v5"
N_ACTIONS   = 6
ACTION_NAMES = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
FEATURES_DIM = 512          # must match train_pong.py
N_STACK      = 4            # frame-stack depth


# ── Replicate the custom CNN so SB3 can deserialize the zip ──────────────────

class AtariCNN(nn.Module):
    """Nature DQN-style CNN — identical to train_pong.py."""
    def __init__(self, observation_space: gym.Space, features_dim: int = FEATURES_DIM):
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        # re-import as BaseFeaturesExtractor for SB3 compatibility
        super().__init__()
        self.features_dim = features_dim
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.float()))


# SB3 requires a BaseFeaturesExtractor subclass to be in scope at load time.
# We patch it in so PPO.load() finds it by name.
try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class AtariCNN(BaseFeaturesExtractor):   # type: ignore[no-redef]
        def __init__(self, observation_space, features_dim=FEATURES_DIM):
            super().__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                sample = torch.as_tensor(observation_space.sample()[None]).float()
                n_flatten = self.cnn(sample).shape[1]
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.linear(self.cnn(observations.float()))

except ImportError:
    pass  # SB3 not available at import time — will fail later with a clear message


# ── Data collection ───────────────────────────────────────────────────────────

def collect_features(model_path: str, n_episodes: int = 200, seed: int = 42) -> tuple:
    """
    Roll out the Pong agent and collect (features, actions).

    Returns
    -------
    features : torch.Tensor  shape (N, 512)
    actions  : torch.Tensor  shape (N,)   int64
    """
    import ale_py
    gym.register_envs(ale_py)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
    from tqdm import tqdm

    print(f"\n[collect] Loading model: {model_path}")
    model = PPO.load(
        model_path,
        custom_objects={"features_extractor_class": AtariCNN},
    )
    print(f"[collect] Device: {model.device}")
    print(f"[collect] Features dim: {model.policy.features_extractor.features_dim}")

    env = make_atari_env(ENV_NAME, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecTransposeImage(env)

    features_list, actions_list = [], []

    obs = env.reset()
    episode_count = 0

    print(f"[collect] Rolling out {n_episodes} episodes …")
    with torch.no_grad():
        pbar = tqdm(total=n_episodes)
        while episode_count < n_episodes:
            action, _ = model.predict(obs, deterministic=False)
            obs_tensor = torch.as_tensor(obs).float().to(model.device)
            feats = model.policy.features_extractor(obs_tensor)
            features_list.append(feats.cpu())
            actions_list.append(torch.tensor(action, dtype=torch.long))
            obs, _, dones, _ = env.step(action)
            if dones[0]:
                episode_count += 1
                pbar.update(1)
                obs = env.reset()
        pbar.close()

    env.close()

    features = torch.cat(features_list, dim=0)
    actions  = torch.cat(actions_list,  dim=0)
    print(f"[collect] {len(features):,} steps collected  |  feature dim = {features.shape[1]}")
    return features, actions


# ── Analysis building blocks (ported from feature_space_analysis.py) ──────────

def svd_analysis(X: np.ndarray, variance_threshold: float = 0.95) -> dict:
    N, d = X.shape
    mean = X.mean(axis=0)
    Xc   = X - mean

    if N > 5 * d:
        cov = (Xc.T @ Xc) / (N - 1)
        eigenvalues, V = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, V = eigenvalues[idx], V[:, idx]
        singular_values = np.sqrt(np.maximum(eigenvalues * (N - 1), 0))
    else:
        _, singular_values, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt.T

    var_exp   = singular_values ** 2
    total_var = var_exp.sum()
    expl_ratio = var_exp / (total_var + 1e-12)
    cumulative  = np.cumsum(expl_ratio)

    k = int(np.searchsorted(cumulative, variance_threshold) + 1)
    k = min(k, d)

    if d > 3:
        log_sv       = np.log(singular_values + 1e-12)
        second_deriv = np.diff(log_sv, n=2)
        k_elbow      = int(np.argmax(second_deriv)) + 2
    else:
        k_elbow = d

    print(f"\n{'='*60}")
    print(f"SVD ANALYSIS")
    print(f"{'='*60}")
    print(f"  Feature dimension d        : {d}")
    print(f"  Number of samples N        : {N:,}")
    print(f"  Signal dim k ({variance_threshold*100:.0f}% var)   : {k}")
    print(f"  Signal dim k (elbow)       : {k_elbow}")
    print(f"  Top-1 explained variance   : {expl_ratio[0]*100:.1f}%")
    print(f"  Top-10 explained variance  : {cumulative[min(9,d-1)]*100:.1f}%")
    print(f"  Top-{k} explained variance  : {cumulative[k-1]*100:.1f}%")
    print(f"  Condition number (σ1/σk)   : {singular_values[0]/(singular_values[k-1]+1e-12):.1f}")
    if k < d:
        noise_energy = var_exp[k:].sum() / total_var * 100
        print(f"  Noise subspace energy      : {noise_energy:.2f}%")
        print(f"  Noise dimensions           : {d - k}")

    return {
        "singular_values": singular_values,
        "V": V,
        "k": k,
        "k_elbow": k_elbow,
        "explained_var": expl_ratio,
        "cumulative_var": cumulative,
        "V_k": V[:, :k],
        "V_noise": V[:, k:] if k < d else np.zeros((d, 0)),
        "mean": mean,
    }


def ica_analysis(X: np.ndarray, k: int, V_k: np.ndarray,
                 n_runs: int = 5, seed: int = 42) -> dict:
    mean = X.mean(axis=0)
    X_pca = (X - mean) @ V_k

    all_components = []
    for run in range(n_runs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica = FastICA(
                n_components=k, algorithm="parallel", whiten="unit-variance",
                max_iter=1000, tol=1e-4, random_state=seed + run,
            )
            ica.fit_transform(X_pca)
            W = ica.components_
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        all_components.append(W / (norms + 1e-12))

    W_ref   = all_components[0]
    X_white = X_pca @ W_ref.T
    kurt_values = np.array([scipy_kurtosis(X_white[:, i], fisher=True) for i in range(k)])

    stability_scores = np.ones(k)
    if n_runs > 1:
        sims_per = [[] for _ in range(k)]
        for run_idx in range(1, n_runs):
            W_other  = all_components[run_idx]
            cos_sim  = np.abs(W_ref @ W_other.T)
            matched  = set()
            for i in range(k):
                avail     = [j for j in range(k) if j not in matched]
                best_j    = avail[np.argmax(cos_sim[i, avail])]
                matched.add(best_j)
                sims_per[i].append(cos_sim[i, best_j])
        stability_scores = np.array([np.mean(s) if s else 1.0 for s in sims_per])

    ica_dirs = (W_ref @ V_k.T).T
    col_norms = np.linalg.norm(ica_dirs, axis=0, keepdims=True)
    ica_dirs  = ica_dirs / (col_norms + 1e-12)

    reliability = stability_scores * np.abs(kurt_values)
    ica_rank    = np.argsort(reliability)[::-1]

    # sign convention: largest-magnitude element positive
    signs = np.sign(ica_dirs[np.abs(ica_dirs).argmax(axis=0), np.arange(k)])
    ica_dirs *= signs[np.newaxis, :]

    print(f"\n{'='*60}")
    print(f"ICA ANALYSIS  (k={k}, {n_runs} runs)")
    print(f"{'='*60}")
    print(f"  Mean stability score       : {stability_scores.mean():.3f}")
    print(f"  Stable components (>0.8)   : {(stability_scores > 0.8).sum()}/{k}")
    print(f"  Stable components (>0.9)   : {(stability_scores > 0.9).sum()}/{k}")
    print(f"  Mean |kurtosis|            : {np.abs(kurt_values).mean():.2f}")
    print(f"\n  Top 10 by reliability (stability × |kurtosis|):")
    for rank, idx in enumerate(ica_rank[:10]):
        print(f"    {rank+1:2d}. IC {idx:3d}: "
              f"stability={stability_scores[idx]:.3f}, "
              f"kurtosis={kurt_values[idx]:+.2f}, "
              f"reliability={reliability[idx]:.3f}")

    return {
        "ica_directions":  ica_dirs,
        "kurtosis_values": kurt_values,
        "stability_scores": stability_scores,
        "reliability":     reliability,
        "ica_rank":        ica_rank,
    }


def compute_normalization_stats(X: np.ndarray) -> dict:
    mean = X.mean(axis=0)
    std  = np.maximum(X.std(axis=0), 1e-6)

    print(f"\n{'='*60}")
    print(f"NORMALIZATION STATS")
    print(f"{'='*60}")
    print(f"  Feature dimension          : {X.shape[1]}")
    print(f"  Mean range                 : [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std  range                 : [{std.min():.4f},  {std.max():.4f}]")
    print(f"  Near-zero std dims (<1e-4) : {(std < 1e-4).sum()}")
    print(f"  High-variance dims (>10)   : {(std > 10).sum()}")

    # ReLU dead-neuron check (feature is always 0 after ReLU → std ≈ 0)
    dead = (std < 1e-3).sum()
    print(f"  Likely dead ReLU dims      : {dead}  ({dead/X.shape[1]*100:.1f}%)")

    return {"mean": mean, "std": std}


def action_distribution_analysis(actions: np.ndarray) -> dict:
    counts = np.bincount(actions, minlength=N_ACTIONS)
    freqs  = counts / counts.sum()

    print(f"\n{'='*60}")
    print(f"ACTION DISTRIBUTION  ({ENV_NAME})")
    print(f"{'='*60}")
    print(f"  Total steps: {len(actions):,}")
    for a in range(N_ACTIONS):
        bar = "█" * int(freqs[a] * 50)
        print(f"  {ACTION_NAMES[a]:12s}: {counts[a]:7,} ({freqs[a]*100:5.1f}%) {bar}")

    freqs_nz  = freqs[freqs > 0]
    entropy    = -np.sum(freqs_nz * np.log2(freqs_nz))
    max_entropy = np.log2(N_ACTIONS)
    print(f"\n  Entropy: {entropy:.3f} / {max_entropy:.3f} (max)")
    print(f"  Normalized entropy: {entropy/max_entropy:.3f}")
    if entropy / max_entropy < 0.5:
        print("  ⚠  Low entropy — agent strongly prefers a few actions.")

    return {"counts": counts, "freqs": freqs,
            "entropy": entropy, "action_names": ACTION_NAMES}


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_diagnostics(svd, ica, norm_stats, act_stats, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # ── panel 1: SVD + ICA ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Pong PPO — Stage 1: Feature Space Analysis",
                 fontsize=14, fontweight="bold")

    # singular value spectrum
    ax = axes[0, 0]
    sv = svd["singular_values"]
    ax.semilogy(range(1, len(sv)+1), sv, "b.-", markersize=2, linewidth=0.8)
    ax.axvline(svd["k"],       color="r", ls="--", alpha=0.7, label=f'k={svd["k"]}')
    ax.axvline(svd["k_elbow"], color="g", ls="--", alpha=0.7, label=f'k_elbow={svd["k_elbow"]}')
    ax.set_xlabel("Component"); ax.set_ylabel("Singular value (log)")
    ax.set_title("Singular Value Spectrum"); ax.legend(); ax.grid(True, alpha=0.3)

    # cumulative variance
    ax = axes[0, 1]
    cum = svd["cumulative_var"]
    ax.plot(range(1, len(cum)+1), cum*100, "b-", lw=2)
    ax.axhline(95,  color="r",      ls="--", alpha=0.5, label="95%")
    ax.axhline(99,  color="orange", ls="--", alpha=0.5, label="99%")
    ax.axvline(svd["k"], color="r", ls="--", alpha=0.3)
    ax.set_xlabel("Components"); ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Cumulative Explained Variance"); ax.legend(); ax.grid(True, alpha=0.3)

    # per-component variance (top 40)
    ax = axes[0, 2]
    n_show = min(40, len(svd["explained_var"]))
    ax.bar(range(1, n_show+1), svd["explained_var"][:n_show]*100,
           color="steelblue", alpha=0.8)
    ax.set_xlabel("Component"); ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"Per-Component Variance (top {n_show})"); ax.grid(True, alpha=0.3)

    # ICA |kurtosis| ranked by reliability
    ax = axes[1, 0]
    kurt     = ica["kurtosis_values"]
    stab     = ica["stability_scores"]
    rank_idx = ica["ica_rank"]
    colors   = ["green" if stab[rank_idx[i]] > 0.8 else "red"
                for i in range(len(rank_idx))]
    ax.bar(range(1, len(rank_idx)+1), np.abs(kurt[rank_idx]),
           color=colors, alpha=0.7)
    ax.set_xlabel("IC rank (by reliability)"); ax.set_ylabel("|Kurtosis|")
    ax.set_title("ICA |Kurtosis| (green=stable >0.8)"); ax.grid(True, alpha=0.3)

    # ICA stability
    ax = axes[1, 1]
    sorted_stab = np.sort(stab)[::-1]
    ax.bar(range(1, len(sorted_stab)+1), sorted_stab, color="teal", alpha=0.7)
    ax.axhline(0.8, color="r",      ls="--", alpha=0.5, label="0.8")
    ax.axhline(0.9, color="orange", ls="--", alpha=0.5, label="0.9")
    ax.set_xlabel("IC rank (by stability)"); ax.set_ylabel("Stability")
    ax.set_title("ICA Stability Across Runs"); ax.legend(); ax.grid(True, alpha=0.3)

    # feature std histogram
    ax = axes[1, 2]
    ax.hist(norm_stats["std"], bins=40, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(1e-3, color="r", ls="--", alpha=0.6, label="dead-ReLU threshold")
    ax.set_xlabel("Per-dimension std"); ax.set_ylabel("Count")
    ax.set_title("Feature Std Distribution"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(save_dir, "stage1_diagnostics.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  Diagnostic plot  → {p}")

    # ── panel 2: action distribution ────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    freqs = act_stats["freqs"]
    bars  = ax2.bar(ACTION_NAMES, freqs*100, color="steelblue", alpha=0.8, edgecolor="black")
    ax2.set_ylabel("Frequency (%)"); ax2.set_title("Action Distribution — Pong rollout")
    ax2.grid(True, alpha=0.3, axis="y"); plt.xticks(rotation=20, ha="right")
    for bar, f in zip(bars, freqs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{f*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(save_dir, "action_distribution.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Action dist plot → {p2}")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_outputs(svd, ica, norm_stats, act_stats, save_dir: str,
                 n_episodes: int, model_path: str) -> str:
    os.makedirs(save_dir, exist_ok=True)

    stage1_data = {
        # SVD
        "V_k":              torch.from_numpy(svd["V_k"]).float(),
        "V_noise":          torch.from_numpy(svd["V_noise"]).float(),
        "singular_values":  torch.from_numpy(svd["singular_values"]).float(),
        "explained_var":    torch.from_numpy(svd["explained_var"]).float(),
        "cumulative_var":   torch.from_numpy(svd["cumulative_var"]).float(),
        "k":                svd["k"],
        "k_elbow":          svd["k_elbow"],
        "feature_mean_svd": torch.from_numpy(svd["mean"]).float(),
        # ICA
        "ica_directions":   torch.from_numpy(ica["ica_directions"]).float(),
        "ica_kurtosis":     torch.from_numpy(ica["kurtosis_values"]).float(),
        "ica_stability":    torch.from_numpy(ica["stability_scores"]).float(),
        "ica_reliability":  torch.from_numpy(ica["reliability"]).float(),
        "ica_rank":         torch.from_numpy(ica["ica_rank"].copy()).long(),
        # Norm
        "feature_mean":     torch.from_numpy(norm_stats["mean"]).float(),
        "feature_std":      torch.from_numpy(norm_stats["std"]).float(),
        # Actions
        "action_counts":    torch.from_numpy(act_stats["counts"]).long(),
        "action_freqs":     torch.from_numpy(act_stats["freqs"]).float(),
        "action_entropy":   act_stats["entropy"],
        "action_names":     ACTION_NAMES,
        # Meta
        "meta": {
            "env_name":    ENV_NAME,
            "env_type":    "atari",
            "features_dim": FEATURES_DIM,
            "n_stack":     N_STACK,
            "n_actions":   N_ACTIONS,
            "model_path":  model_path,
            "n_episodes":  n_episodes,
        },
    }

    pt_path = os.path.join(save_dir, "stage1_outputs.pt")
    torch.save(stage1_data, pt_path)
    print(f"\n  stage1_outputs.pt → {pt_path}")

    summary = {
        "env_name":              ENV_NAME,
        "env_type":              "atari",
        "features_dim":          FEATURES_DIM,
        "signal_dim_k_95pct":   int(svd["k"]),
        "signal_dim_k_elbow":   int(svd["k_elbow"]),
        "top10_cumvar_pct":     float(svd["cumulative_var"][min(9, svd["k"]-1)] * 100),
        "ica_n_stable_08":      int((ica["stability_scores"] > 0.8).sum()),
        "ica_n_stable_09":      int((ica["stability_scores"] > 0.9).sum()),
        "ica_mean_stability":   float(ica["stability_scores"].mean()),
        "ica_mean_abs_kurtosis":float(np.abs(ica["kurtosis_values"]).mean()),
        "feature_mean_range":   [float(norm_stats["mean"].min()), float(norm_stats["mean"].max())],
        "feature_std_range":    [float(norm_stats["std"].min()),  float(norm_stats["std"].max())],
        "dead_relu_dims":       int((norm_stats["std"] < 1e-3).sum()),
        "action_entropy":       float(act_stats["entropy"]),
        "action_entropy_norm":  float(act_stats["entropy"] / np.log2(N_ACTIONS)),
        "action_names":         ACTION_NAMES,
    }
    json_path = os.path.join(save_dir, "stage1_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  stage1_summary.json → {json_path}")

    return pt_path


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_analysis(features: torch.Tensor, actions: torch.Tensor,
                 variance_threshold: float, ica_n_runs: int,
                 seed: int, save_dir: str,
                 n_episodes: int, model_path: str):

    X = features.numpy()
    A = actions.numpy().astype(int)

    print(f"\n{'#'*60}")
    print(f"  PONG STAGE 1 FEATURE ANALYSIS")
    print(f"  N={X.shape[0]:,}  d={X.shape[1]}")
    print(f"{'#'*60}")

    svd       = svd_analysis(X, variance_threshold)
    ica       = ica_analysis(X, svd["k"], svd["V_k"], ica_n_runs, seed)
    norm_stat = compute_normalization_stats(X)
    act_stat  = action_distribution_analysis(A)

    plot_diagnostics(svd, ica, norm_stat, act_stat, save_dir)
    save_outputs(svd, ica, norm_stat, act_stat, save_dir, n_episodes, model_path)

    k = svd["k"]
    d = X.shape[1]
    n_stable = int((ica["stability_scores"] > 0.8).sum())
    dead     = int((norm_stat["std"] < 1e-3).sum())

    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS FOR STAGE 2 (SAE TRAINING)")
    print(f"{'='*60}")
    print(f"  Effective feature dim       : {k} (of {d})")
    print(f"  Recommended hidden_dim (4×) : {4*k}")
    print(f"  Recommended hidden_dim (8×) : {8*k}")
    print(f"  Stable ICA anchors (>0.8)   : {n_stable}")
    print(f"  Dead ReLU dims to exclude   : {dead}")
    print(f"  → Normalize features with saved mean/std")
    print(f"  → Initialize first {n_stable} SAE decoder cols with stable ICA directions")
    print(f"  → Use V_k for signal-subspace regularization")
    print(f"  → Consider masking {dead} dead-ReLU dims from reconstruction loss")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 feature-space analysis for ppo_pong.zip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", default="ppo_pong.zip",
                        help="Path to PPO zip (or best_pong/best_model.zip)")
    parser.add_argument("--features_path", default=None,
                        help="Skip collection; load pre-saved collected_data.pt")
    parser.add_argument("--n_episodes", type=int, default=200,
                        help="Episodes to collect (default 200 ≈ 40–60k steps)")
    parser.add_argument("--variance_threshold", type=float, default=0.95)
    parser.add_argument("--ica_n_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="./pong_stage1")
    args = parser.parse_args()

    if args.features_path is not None:
        print(f"[main] Loading pre-collected features from {args.features_path}")
        data     = torch.load(args.features_path, map_location="cpu", weights_only=False)
        features = data["features"]
        actions  = data["actions"]
        model_path = data.get("model_path", args.model_path)
        n_episodes = data.get("n_episodes", args.n_episodes)
    else:
        features, actions = collect_features(
            args.model_path, args.n_episodes, args.seed
        )
        model_path = args.model_path
        n_episodes = args.n_episodes

        # save raw data so you can re-run analysis without re-collecting
        os.makedirs(args.save_dir, exist_ok=True)
        raw_path = os.path.join(args.save_dir, "collected_data.pt")
        torch.save({
            "features":   features,
            "actions":    actions,
            "model_path": model_path,
            "n_episodes": n_episodes,
            "env_name":   ENV_NAME,
            "env_type":   "atari",
        }, raw_path)
        print(f"[main] Raw data saved → {raw_path}")

    run_analysis(
        features, actions,
        variance_threshold=args.variance_threshold,
        ica_n_runs=args.ica_n_runs,
        seed=args.seed,
        save_dir=args.save_dir,
        n_episodes=n_episodes,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()