"""
Stage 1: Feature Space Analysis
================================
Analyzes the frozen PPO feature space before SAE training.
Supports both MiniGrid and Atari environments.

Outputs:
    - SVD: signal subspace V_k, effective dimensionality k, explained variance curve
    - ICA: k identifiable directions in R^d (stable anchors for SAE init)
    - Feature normalization stats (mean, std)
    - Diagnostic plots and report

Usage (MiniGrid):
    python feature_space_analysis.py \
        --model_path ppo_doorkey_5x5.zip \
        --env_name MiniGrid-DoorKey-5x5-v0 \
        --env_type minigrid

Usage (Atari):
    python feature_space_analysis.py \
        --model_path ppo_atari_breakout.zip \
        --env_name "ALE/Breakout-v5" \
        --env_type atari \
        --n_episodes 200

Usage (pre-collected features):
    python feature_space_analysis.py \
        --features_path ./collected_data/features.pt
"""

import argparse
import os
import json
import warnings

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis as scipy_kurtosis


# ---------------------------------------------------------------------------
# Environment registry — add new envs here, no other changes needed
# ---------------------------------------------------------------------------

ENV_REGISTRY = {
    # MiniGrid
    "MiniGrid-DoorKey-5x5-v0":     {"type": "minigrid", "n_actions": 7},
    "MiniGrid-DoorKey-8x8-v0":     {"type": "minigrid", "n_actions": 7},
    "MiniGrid-DoorKey-16x16-v0":   {"type": "minigrid", "n_actions": 7},
    "MiniGrid-FourRooms-v0":       {"type": "minigrid", "n_actions": 7},
    "MiniGrid-Empty-5x5-v0":       {"type": "minigrid", "n_actions": 7},
    # Atari
    "ALE/Breakout-v5":             {"type": "atari",    "n_actions": 4,
                                    "action_names": ["NOOP","FIRE","RIGHT","LEFT"]},
    "ALE/Pong-v5":                 {"type": "atari",    "n_actions": 6,
                                    "action_names": ["NOOP","FIRE","RIGHT","LEFT","RIGHTFIRE","LEFTFIRE"]},
    "ALE/SpaceInvaders-v5":        {"type": "atari",    "n_actions": 6,
                                    "action_names": ["NOOP","FIRE","RIGHT","LEFT","RIGHTFIRE","LEFTFIRE"]},
    "ALE/Enduro-v5":               {"type": "atari",    "n_actions": 9,
                                    "action_names": ["NOOP","FIRE","RIGHT","LEFT","DOWN",
                                                     "DOWNRIGHT","DOWNLEFT","RIGHTFIRE","LEFTFIRE"]},
    "ALE/Qbert-v5":                {"type": "atari",    "n_actions": 6,
                                    "action_names": ["NOOP","FIRE","RIGHT","LEFT","RIGHTFIRE","LEFTFIRE"]},
    "ALE/MontezumaRevenge-v5":     {"type": "atari",    "n_actions": 18, "action_names": None},
}

MINIGRID_ACTION_NAMES = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]


def resolve_env_info(env_name: str, env_type: str = None):
    """Return (env_type, n_actions, action_names) for a given env."""
    info = ENV_REGISTRY.get(env_name)
    if info:
        etype = info["type"]
        n_actions = info["n_actions"]
        action_names = info.get("action_names", None)
        if etype == "minigrid":
            action_names = MINIGRID_ACTION_NAMES
        return etype, n_actions, action_names

    # Not in registry — fall back to the explicit --env_type flag
    if env_type == "minigrid":
        return "minigrid", 7, MINIGRID_ACTION_NAMES
    elif env_type == "atari":
        # n_actions will be inferred from collected data
        return "atari", None, None
    else:
        raise ValueError(
            f"Unknown env '{env_name}'. Either add it to ENV_REGISTRY or pass --env_type."
        )


# ---------------------------------------------------------------------------
# 1a  SVD Analysis
# ---------------------------------------------------------------------------

def svd_analysis(X: np.ndarray, variance_threshold: float = 0.95):
    N, d = X.shape
    mean = X.mean(axis=0)
    X_centered = X - mean

    if N > 5 * d:
        cov = (X_centered.T @ X_centered) / (N - 1)
        eigenvalues, V = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]
        singular_values = np.sqrt(np.maximum(eigenvalues * (N - 1), 0))
    else:
        _, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)
        V = Vt.T

    var_explained = singular_values ** 2
    total_var = var_explained.sum()
    explained_ratio = var_explained / (total_var + 1e-12)
    cumulative = np.cumsum(explained_ratio)

    k = int(np.searchsorted(cumulative, variance_threshold) + 1)
    k = min(k, d)

    if d > 3:
        log_sv = np.log(singular_values + 1e-12)
        second_deriv = np.diff(log_sv, n=2)
        elbow_idx = int(np.argmax(second_deriv)) + 2
        k_elbow = elbow_idx
    else:
        k_elbow = d

    print(f"\n{'='*60}")
    print(f"SVD ANALYSIS")
    print(f"{'='*60}")
    print(f"  Feature dimension d        : {d}")
    print(f"  Number of samples N        : {N}")
    print(f"  Signal dim k ({variance_threshold*100:.0f}% var)   : {k}")
    print(f"  Signal dim k (elbow)       : {k_elbow}")
    print(f"  Top-1 explained variance   : {explained_ratio[0]*100:.1f}%")
    print(f"  Top-10 explained variance  : {cumulative[min(9,d-1)]*100:.1f}%")
    print(f"  Top-{k} explained variance  : {cumulative[k-1]*100:.1f}%")
    print(f"  Condition number (σ1/σk)   : {singular_values[0]/(singular_values[k-1]+1e-12):.1f}")

    if k < d:
        noise_energy = var_explained[k:].sum() / total_var * 100
        print(f"  Noise subspace energy      : {noise_energy:.2f}%")
        print(f"  Noise dimensions           : {d - k}")

    return {
        "singular_values": singular_values,
        "V": V,
        "k": k,
        "k_elbow": k_elbow,
        "explained_var": explained_ratio,
        "cumulative_var": cumulative,
        "V_k": V[:, :k],
        "V_noise": V[:, k:] if k < d else np.zeros((d, 0)),
        "mean": mean,
    }


# ---------------------------------------------------------------------------
# 1b  ICA Analysis
# ---------------------------------------------------------------------------

def ica_analysis(X: np.ndarray, k: int, V_k: np.ndarray, n_runs: int = 5, seed: int = 42):
    N, d = X.shape
    mean = X.mean(axis=0)
    X_centered = X - mean
    X_pca = X_centered @ V_k

    all_components = []
    for run in range(n_runs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica = FastICA(
                n_components=k, algorithm="parallel", whiten="unit-variance",
                max_iter=1000, tol=1e-4, random_state=seed + run,
            )
            S = ica.fit_transform(X_pca)
            W = ica.components_
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W_norm = W / (norms + 1e-12)
        all_components.append(W_norm)

    W_ref = all_components[0]
    A_ref = np.linalg.pinv(W_ref)
    X_white = X_pca @ W_ref.T
    kurt_values = np.array([scipy_kurtosis(X_white[:, i], fisher=True) for i in range(k)])

    stability_scores = np.ones(k)
    if n_runs > 1:
        similarities_per_component = [[] for _ in range(k)]
        for run_idx in range(1, n_runs):
            W_other = all_components[run_idx]
            cos_sim = np.abs(W_ref @ W_other.T)
            matched = set()
            for i in range(k):
                available = [j for j in range(k) if j not in matched]
                sims = cos_sim[i, available]
                best_local = np.argmax(sims)
                best_j = available[best_local]
                matched.add(best_j)
                similarities_per_component[i].append(cos_sim[i, best_j])
        stability_scores = np.array([np.mean(s) if s else 1.0 for s in similarities_per_component])

    ica_directions_d = (W_ref @ V_k.T).T
    col_norms = np.linalg.norm(ica_directions_d, axis=0, keepdims=True)
    ica_directions_d = ica_directions_d / (col_norms + 1e-12)

    reliability = stability_scores * np.abs(kurt_values)
    ica_rank = np.argsort(reliability)[::-1]

    print(f"\n{'='*60}")
    print(f"ICA ANALYSIS (k={k} components, {n_runs} runs)")
    print(f"{'='*60}")
    print(f"  Mean stability score       : {stability_scores.mean():.3f}")
    print(f"  Stable components (>0.8)   : {(stability_scores > 0.8).sum()}/{k}")
    print(f"  Stable components (>0.9)   : {(stability_scores > 0.9).sum()}/{k}")
    print(f"  Mean |kurtosis|            : {np.abs(kurt_values).mean():.2f}")
    print(f"\n  Top 10 components by reliability (stability × |kurtosis|):")
    for rank, idx in enumerate(ica_rank[:10]):
        print(
            f"    {rank+1:2d}. IC {idx:3d}: "
            f"stability={stability_scores[idx]:.3f}, "
            f"kurtosis={kurt_values[idx]:+.2f}, "
            f"reliability={reliability[idx]:.3f}"
        )

    signs = np.sign(
        ica_directions_d[np.abs(ica_directions_d).argmax(axis=0), np.arange(k)]
    )
    ica_directions_d *= signs[np.newaxis, :]
    return {
        "ica_directions": ica_directions_d,
        "mixing_matrix": A_ref,
        "kurtosis_values": kurt_values,
        "stability_scores": stability_scores,
        "reliability": reliability,
        "ica_rank": ica_rank,
    }


# ---------------------------------------------------------------------------
# Feature normalization stats
# ---------------------------------------------------------------------------

def compute_normalization_stats(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.maximum(std, 1e-6)

    print(f"\n{'='*60}")
    print(f"NORMALIZATION STATS")
    print(f"{'='*60}")
    print(f"  Feature dimension          : {X.shape[1]}")
    print(f"  Mean range                 : [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range                  : [{std.min():.4f}, {std.max():.4f}]")
    print(f"  Near-zero std dims (<1e-4) : {(std < 1e-4).sum()}")
    print(f"  High-variance dims (>10)   : {(std > 10).sum()}")

    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Action distribution analysis
# ---------------------------------------------------------------------------

def action_distribution_analysis(actions: np.ndarray, action_names=None):
    n_actions = int(actions.max()) + 1

    if action_names is None:
        action_names = [f"action_{i}" for i in range(n_actions)]
    else:
        # Pad/trim to match actual number of actions seen
        if len(action_names) < n_actions:
            action_names = action_names + [f"action_{i}" for i in range(len(action_names), n_actions)]
        action_names = action_names[:n_actions]

    counts = np.bincount(actions, minlength=n_actions)
    freqs = counts / counts.sum()

    print(f"\n{'='*60}")
    print(f"ACTION DISTRIBUTION")
    print(f"{'='*60}")
    print(f"  Total samples: {len(actions)}")
    for a in range(n_actions):
        bar = "█" * int(freqs[a] * 50)
        print(f"  {action_names[a]:12s}: {counts[a]:6d} ({freqs[a]*100:5.1f}%) {bar}")

    freqs_nonzero = freqs[freqs > 0]
    entropy = -np.sum(freqs_nonzero * np.log2(freqs_nonzero))
    max_entropy = np.log2(n_actions)
    print(f"\n  Entropy: {entropy:.3f} / {max_entropy:.3f} (max)")
    print(f"  Normalized entropy: {entropy/max_entropy:.3f}")

    return {"counts": counts, "freqs": freqs, "entropy": entropy, "action_names": action_names}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_diagnostics(svd_result, ica_result, norm_stats, action_stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Stage 1: Feature Space Analysis", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    sv = svd_result["singular_values"]
    ax.semilogy(range(1, len(sv) + 1), sv, "b.-", markersize=3)
    ax.axvline(svd_result["k"], color="r", linestyle="--", alpha=0.7, label=f'k={svd_result["k"]}')
    ax.axvline(svd_result["k_elbow"], color="g", linestyle="--", alpha=0.7, label=f'k_elbow={svd_result["k_elbow"]}')
    ax.set_xlabel("Component index"); ax.set_ylabel("Singular value (log)")
    ax.set_title("Singular Value Spectrum"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    cum = svd_result["cumulative_var"]
    ax.plot(range(1, len(cum) + 1), cum * 100, "b-", linewidth=2)
    ax.axhline(95, color="r", linestyle="--", alpha=0.5, label="95%")
    ax.axhline(99, color="orange", linestyle="--", alpha=0.5, label="99%")
    ax.axvline(svd_result["k"], color="r", linestyle="--", alpha=0.3)
    ax.set_xlabel("Number of components"); ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Cumulative Explained Variance"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    n_show = min(30, len(svd_result["explained_var"]))
    ax.bar(range(1, n_show + 1), svd_result["explained_var"][:n_show] * 100, color="steelblue", alpha=0.8)
    ax.set_xlabel("Component index"); ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"Per-Component Variance (top {n_show})"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    kurt = ica_result["kurtosis_values"]
    ranked_kurt = kurt[ica_result["ica_rank"]]
    colors = ["green" if ica_result["stability_scores"][ica_result["ica_rank"][i]] > 0.8 else "red"
              for i in range(len(ranked_kurt))]
    ax.bar(range(1, len(ranked_kurt) + 1), np.abs(ranked_kurt), color=colors, alpha=0.7)
    ax.set_xlabel("IC rank (by reliability)"); ax.set_ylabel("|Kurtosis|")
    ax.set_title("ICA Components: |Kurtosis| (green=stable)"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    stab = ica_result["stability_scores"]
    sorted_stab = np.sort(stab)[::-1]
    ax.bar(range(1, len(sorted_stab) + 1), sorted_stab, color="teal", alpha=0.7)
    ax.axhline(0.8, color="r", linestyle="--", alpha=0.5, label="0.8 threshold")
    ax.axhline(0.9, color="orange", linestyle="--", alpha=0.5, label="0.9 threshold")
    ax.set_xlabel("IC rank (by stability)"); ax.set_ylabel("Stability score")
    ax.set_title("ICA Stability Across Runs"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    std = norm_stats["std"]
    ax.hist(std, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Per-dimension std"); ax.set_ylabel("Count")
    ax.set_title("Feature Std Distribution"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "stage1_diagnostics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Diagnostic plot saved: {plot_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    action_names = action_stats["action_names"]
    freqs = action_stats["freqs"]
    bars = ax2.bar(action_names, freqs * 100, color="steelblue", alpha=0.8, edgecolor="black")
    ax2.set_ylabel("Frequency (%)"); ax2.set_title("Action Distribution in Rollout Data")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    for bar, f in zip(bars, freqs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{f*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    action_plot_path = os.path.join(save_dir, "action_distribution.png")
    plt.savefig(action_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Action distribution plot saved: {action_plot_path}")


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_stage1_outputs(svd_result, ica_result, norm_stats, action_stats, save_dir, meta=None):
    os.makedirs(save_dir, exist_ok=True)

    stage1_data = {
        "V_k": torch.from_numpy(svd_result["V_k"]).float(),
        "V_noise": torch.from_numpy(svd_result["V_noise"]).float(),
        "singular_values": torch.from_numpy(svd_result["singular_values"]).float(),
        "explained_var": torch.from_numpy(svd_result["explained_var"]).float(),
        "cumulative_var": torch.from_numpy(svd_result["cumulative_var"]).float(),
        "k": svd_result["k"],
        "k_elbow": svd_result["k_elbow"],
        "feature_mean_svd": torch.from_numpy(svd_result["mean"]).float(),
        "ica_directions": torch.from_numpy(ica_result["ica_directions"]).float(),
        "ica_kurtosis": torch.from_numpy(ica_result["kurtosis_values"]).float(),
        "ica_stability": torch.from_numpy(ica_result["stability_scores"]).float(),
        "ica_reliability": torch.from_numpy(ica_result["reliability"]).float(),
        "ica_rank": torch.from_numpy(ica_result["ica_rank"].copy()).long(),
        "feature_mean": torch.from_numpy(norm_stats["mean"]).float(),
        "feature_std": torch.from_numpy(norm_stats["std"]).float(),
        "action_counts": torch.from_numpy(action_stats["counts"]).long(),
        "action_freqs": torch.from_numpy(action_stats["freqs"]).float(),
        "action_entropy": action_stats["entropy"],
        "action_names": action_stats["action_names"],
    }
    if meta:
        stage1_data["meta"] = meta

    save_path = os.path.join(save_dir, "stage1_outputs.pt")
    torch.save(stage1_data, save_path)
    print(f"\n  Stage 1 outputs saved: {save_path}")

    summary = {
        "env_name": meta.get("env_name", "unknown") if meta else "unknown",
        "env_type": meta.get("env_type", "unknown") if meta else "unknown",
        "feature_dim": int(svd_result["V"].shape[0]),
        "signal_dim_k_95pct": int(svd_result["k"]),
        "signal_dim_k_elbow": int(svd_result["k_elbow"]),
        "top10_cumulative_var": float(svd_result["cumulative_var"][min(9, len(svd_result["cumulative_var"])-1)]),
        "ica_n_stable_08": int((ica_result["stability_scores"] > 0.8).sum()),
        "ica_n_stable_09": int((ica_result["stability_scores"] > 0.9).sum()),
        "ica_mean_stability": float(ica_result["stability_scores"].mean()),
        "ica_mean_abs_kurtosis": float(np.abs(ica_result["kurtosis_values"]).mean()),
        "feature_mean_range": [float(norm_stats["mean"].min()), float(norm_stats["mean"].max())],
        "feature_std_range": [float(norm_stats["std"].min()), float(norm_stats["std"].max())],
        "action_entropy": float(action_stats["entropy"]),
        "action_names": action_stats["action_names"],
    }

    summary_path = os.path.join(save_dir, "stage1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Stage 1 summary saved: {summary_path}")

    return save_path


def load_stage1_outputs(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Loaded Stage 1 outputs from {path}")
    print(f"  Signal dimension k: {data['k']}")
    print(f"  ICA directions shape: {data['ica_directions'].shape}")
    return data


# ---------------------------------------------------------------------------
# Data collection — MiniGrid
# ---------------------------------------------------------------------------

def collect_features_minigrid(model_path, env_name, n_episodes=800, seed=42, tile_size=8):
    """Collect features from a MiniGrid environment."""
    import gymnasium as gym
    import minigrid  # noqa: F401
    from minigrid.wrappers import ImgObsWrapper
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    from tqdm import tqdm

    print(f"[MiniGrid] Loading PPO model from {model_path}...")
    model = PPO.load(model_path)

    raw_env_ref = [None]

    def make_env():
        def _init():
            env = gym.make(env_name, render_mode="rgb_array")
            env = ImgObsWrapper(env)
            raw_env_ref[0] = env
            return env
        return _init

    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    features_list, actions_list = [], []
    obs_grid_list, obs_pixel_list = [], []

    print(f"[MiniGrid] Collecting {n_episodes} episodes...")
    obs = env.reset()
    episode_count = 0

    with torch.no_grad():
        pbar = tqdm(total=n_episodes)
        while episode_count < n_episodes:
            raw_env = raw_env_ref[0].unwrapped
            try:
                grid_obs = raw_env.gen_obs()['image']
                obs_grid_list.append(grid_obs.copy())
                pixel_obs = raw_env.get_obs_render(grid_obs, tile_size=tile_size)
                obs_pixel_list.append(pixel_obs)
            except Exception:
                obs_grid_list.append(np.zeros((7, 7, 3), dtype=np.uint8))
                obs_pixel_list.append(np.zeros((7 * tile_size, 7 * tile_size, 3), dtype=np.uint8))

            action, _ = model.predict(obs, deterministic=False)
            obs_tensor = torch.as_tensor(obs).float().to(model.device)
            features = model.policy.features_extractor(obs_tensor)
            features_list.append(features.cpu())
            actions_list.append(torch.tensor(action))
            obs, _, dones, _ = env.step(action)
            if dones[0]:
                episode_count += 1
                pbar.update(1)
                obs = env.reset()
        pbar.close()

    env.close()

    features = torch.cat(features_list, dim=0)
    actions = torch.cat(actions_list, dim=0)
    observations = torch.tensor(np.stack(obs_grid_list))
    observations_pixel = torch.tensor(np.stack(obs_pixel_list))

    print(f"[MiniGrid] Collected {len(features)} samples, feature dim = {features.shape[1]}")
    print(f"  Grid observations: {observations.shape}")
    print(f"  Pixel observations: {observations_pixel.shape}")

    return features, actions, {"observations": observations, "observations_pixel": observations_pixel}


# ---------------------------------------------------------------------------
# Data collection — Atari
# ---------------------------------------------------------------------------

def collect_features_atari(model_path, env_name, n_episodes=200, seed=42):
    """Collect features from an Atari (ALE) environment."""
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
    from tqdm import tqdm

    print(f"[Atari] Loading PPO model from {model_path}...")
    model = PPO.load(model_path)

    env = make_atari_env(env_name, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    features_list, actions_list = [], []

    print(f"[Atari] Collecting {n_episodes} episodes...")
    obs = env.reset()
    episode_count = 0

    with torch.no_grad():
        pbar = tqdm(total=n_episodes)
        while episode_count < n_episodes:
            action, _ = model.predict(obs, deterministic=False)
            obs_tensor = torch.as_tensor(obs).float().to(model.device)
            features = model.policy.features_extractor(obs_tensor)
            features_list.append(features.cpu())
            actions_list.append(torch.tensor(action))
            obs, _, dones, _ = env.step(action)
            if dones[0]:
                episode_count += 1
                pbar.update(1)
                obs = env.reset()
        pbar.close()

    env.close()

    features = torch.cat(features_list, dim=0)
    actions = torch.cat(actions_list, dim=0)

    print(f"[Atari] Collected {len(features)} samples, feature dim = {features.shape[1]}")
    return features, actions, {}   # no grid/pixel obs for Atari


# ---------------------------------------------------------------------------
# Unified collect_features dispatcher
# ---------------------------------------------------------------------------

def collect_features(model_path, env_name, env_type, n_episodes=800, seed=42, tile_size=8):
    if env_type == "minigrid":
        return collect_features_minigrid(model_path, env_name, n_episodes, seed, tile_size)
    elif env_type == "atari":
        return collect_features_atari(model_path, env_name, n_episodes, seed)
    else:
        raise ValueError(f"Unknown env_type: {env_type}. Must be 'minigrid' or 'atari'.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stage1(features: torch.Tensor, actions: torch.Tensor,
               action_names=None,
               variance_threshold: float = 0.95,
               ica_n_runs: int = 5,
               seed: int = 42,
               save_dir: str = "./stage1_outputs",
               meta: dict = None):
    X = features.numpy()
    A = actions.numpy().astype(int)

    print(f"\n{'#'*60}")
    print(f"  STAGE 1: FEATURE SPACE ANALYSIS")
    print(f"  N = {X.shape[0]}, d = {X.shape[1]}")
    print(f"{'#'*60}")

    svd_result = svd_analysis(X, variance_threshold=variance_threshold)
    ica_result = ica_analysis(
        X, k=svd_result["k"], V_k=svd_result["V_k"],
        n_runs=ica_n_runs, seed=seed
    )
    norm_stats = compute_normalization_stats(X)
    action_stats = action_distribution_analysis(A, action_names=action_names)
    plot_diagnostics(svd_result, ica_result, norm_stats, action_stats, save_dir)
    save_path = save_stage1_outputs(svd_result, ica_result, norm_stats, action_stats, save_dir, meta=meta)

    k = svd_result["k"]
    d = X.shape[1]
    n_stable_ica = int((ica_result["stability_scores"] > 0.8).sum())

    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS FOR STAGE 2 (SAE TRAINING)")
    print(f"{'='*60}")
    print(f"  Effective feature dim       : {k} (of {d})")
    print(f"  Recommended hidden_dim (4x) : {4 * k}")
    print(f"  Recommended hidden_dim (8x) : {8 * k}")
    print(f"  Stable ICA anchors          : {n_stable_ica}")
    print(f"  → Initialize first {n_stable_ica} decoder columns with stable ICA directions")
    print(f"  → Use V_k for signal subspace regularization")
    print(f"  → Normalize features with saved mean/std before SAE training")

    return load_stage1_outputs(save_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Feature Space Analysis (MiniGrid + Atari)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MiniGrid
  python feature_space_analysis.py \\
      --model_path ppo_doorkey_5x5.zip \\
      --env_name MiniGrid-DoorKey-5x5-v0

  # Atari
  python feature_space_analysis.py \\
      --model_path ppo_atari_breakout.zip \\
      --env_name "ALE/Breakout-v5" \\
      --env_type atari \\
      --n_episodes 200

  # Pre-collected features (env-agnostic)
  python feature_space_analysis.py \\
      --features_path ./collected_data/features.pt
        """
    )
    parser.add_argument("--features_path", type=str, default=None,
                        help="Path to pre-collected .pt file (with 'features' and 'actions' keys). "
                             "Skips collection entirely.")
    parser.add_argument("--model_path", type=str, default="ppo_doorkey_5x5.zip",
                        help="PPO model path (used if --features_path not given)")
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-5x5-v0",
                        help="Gymnasium environment ID")
    parser.add_argument("--env_type", type=str, default=None, choices=["minigrid", "atari"],
                        help="Override env type. Auto-detected from ENV_REGISTRY if not given.")
    parser.add_argument("--n_episodes", type=int, default=800,
                        help="Number of episodes to collect (Atari: recommend 200, MiniGrid: 800)")
    parser.add_argument("--variance_threshold", type=float, default=0.95)
    parser.add_argument("--ica_n_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./stage1_outputs")
    parser.add_argument("--tile_size", type=int, default=8,
                        help="MiniGrid only: tile size for pixel rendering")

    args = parser.parse_args()

    extra_obs = {}

    if args.features_path is not None:
        # ── Pre-collected path ──────────────────────────────────────────────
        print(f"Loading features from {args.features_path}...")
        data = torch.load(args.features_path, map_location="cpu", weights_only=False)
        features = data["features"]
        actions = data["actions"]
        action_names = data.get("action_names", None)
        env_type = data.get("env_type", args.env_type or "unknown")
        env_name = data.get("env_name", args.env_name)
    else:
        # ── Live collection path ────────────────────────────────────────────
        env_type, n_actions, action_names = resolve_env_info(args.env_name, args.env_type)
        env_name = args.env_name

        print(f"\nEnvironment : {env_name}")
        print(f"Type        : {env_type}")
        print(f"Actions     : {n_actions} {action_names}")

        features, actions, extra_obs = collect_features(
            args.model_path, env_name, env_type,
            n_episodes=args.n_episodes, seed=args.seed, tile_size=args.tile_size
        )

        # Infer n_actions from data if not known (e.g. unlisted Atari game)
        if action_names is None:
            n_actions_actual = int(actions.max().item()) + 1
            action_names = [f"action_{i}" for i in range(n_actions_actual)]

        # Save raw collected data
        os.makedirs(args.save_dir, exist_ok=True)
        raw_path = os.path.join(args.save_dir, "collected_data.pt")
        save_dict = {
            "features": features,
            "actions": actions,
            "env_name": env_name,
            "env_type": env_type,
            "action_names": action_names,
        }
        save_dict.update(extra_obs)
        torch.save(save_dict, raw_path)
        print(f"\nRaw data saved: {raw_path}")
        for k, v in extra_obs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")

    meta = {"env_name": env_name if 'env_name' in dir() else args.env_name,
            "env_type": env_type if 'env_type' in dir() else (args.env_type or "unknown")}

    stage1_data = run_stage1(
        features, actions,
        action_names=action_names if 'action_names' in dir() else None,
        variance_threshold=args.variance_threshold,
        ica_n_runs=args.ica_n_runs,
        seed=args.seed,
        save_dir=args.save_dir,
        meta=meta,
    )


if __name__ == "__main__":
    main()