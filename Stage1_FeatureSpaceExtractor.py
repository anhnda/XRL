"""
Stage 1: Feature Space Analysis
================================
Analyzes the frozen PPO feature space before SAE training.

Outputs:
    - SVD: signal subspace V_k, effective dimensionality k, explained variance curve
    - ICA: k identifiable directions in R^d (stable anchors for SAE init)
    - Feature normalization stats (mean, std)
    - Diagnostic plots and report

Usage:
    python feature_space_analysis.py --features_path ./collected_data/features.pt
    python feature_space_analysis.py --model_path ppo_doorkey_5x5.zip --env_name MiniGrid-DoorKey-5x5-v0
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
# 1a  SVD Analysis
# ---------------------------------------------------------------------------

def svd_analysis(X: np.ndarray, variance_threshold: float = 0.95):
    """
    SVD on centered feature matrix.

    Args:
        X: (N, d) feature matrix
        variance_threshold: cumulative variance fraction to determine k

    Returns:
        dict with keys:
            singular_values : (d,)
            V               : (d, d) right singular vectors (columns = directions)
            k               : effective signal dimension
            explained_var   : (d,) per-component explained variance ratio
            cumulative_var  : (d,) cumulative explained variance
            V_k             : (d, k) signal subspace basis
            V_noise         : (d, d-k) noise subspace basis
    """
    N, d = X.shape

    # Center
    mean = X.mean(axis=0)
    X_centered = X - mean

    # SVD  (economy: U is N×d, S is d, Vt is d×d)
    # For large N we only need S and Vt, so use covariance trick if N >> d
    if N > 5 * d:
        # Covariance approach — more memory-efficient for large N
        cov = (X_centered.T @ X_centered) / (N - 1)
        eigenvalues, V = np.linalg.eigh(cov)
        # eigh returns ascending order — flip to descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]
        singular_values = np.sqrt(np.maximum(eigenvalues * (N - 1), 0))
    else:
        _, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)
        V = Vt.T  # (d, d) columns are right singular vectors

    # Explained variance
    var_explained = singular_values ** 2
    total_var = var_explained.sum()
    explained_ratio = var_explained / (total_var + 1e-12)
    cumulative = np.cumsum(explained_ratio)

    # Effective dimension at threshold
    k = int(np.searchsorted(cumulative, variance_threshold) + 1)
    k = min(k, d)

    # Elbow detection (second derivative of singular value curve)
    if d > 3:
        log_sv = np.log(singular_values + 1e-12)
        second_deriv = np.diff(log_sv, n=2)
        elbow_idx = int(np.argmax(second_deriv)) + 2  # +2 for diff offset
        # Use the more conservative of threshold-based and elbow-based
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

    # Noise floor stats
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
    """
    ICA on PCA-reduced features to extract identifiable directions.

    We run ICA on the k-dimensional PCA projection, then project the
    independent components back to the original d-dimensional space.

    Multiple runs are used to assess ICA stability (identifiability).

    Args:
        X      : (N, d) feature matrix
        k      : signal subspace dimension (from SVD)
        V_k    : (d, k) signal subspace basis
        n_runs : number of ICA runs with different seeds for stability check
        seed   : base random seed

    Returns:
        dict with keys:
            ica_directions   : (d, k) ICA directions in original space (unit norm)
            mixing_matrix    : (k, k) ICA mixing matrix
            kurtosis_values  : (k,) kurtosis of each IC (higher = more non-Gaussian = more stable)
            stability_scores : (k,) cross-run stability per component
            ica_rank         : (k,) indices sorted by stability * |kurtosis|
    """
    N, d = X.shape
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Project to PCA subspace
    X_pca = X_centered @ V_k  # (N, k)

    # --- Run ICA multiple times for stability assessment ---
    all_components = []  # list of (k, k) matrices

    for run in range(n_runs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica = FastICA(
                n_components=k,
                algorithm="parallel",
                whiten="unit-variance",
                max_iter=1000,
                tol=1e-4,
                random_state=seed + run,
            )
            S = ica.fit_transform(X_pca)  # (N, k) sources
            W = ica.components_  # (k, k) unmixing in PCA space

        # Normalize rows to unit norm
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W_norm = W / (norms + 1e-12)

        all_components.append(W_norm)

    # Use first run as reference
    W_ref = all_components[0]  # (k, k) — rows are IC directions in PCA space
    A_ref = np.linalg.pinv(W_ref)  # mixing matrix

    # Compute sources for kurtosis
    X_white = X_pca @ W_ref.T  # (N, k)
    kurt_values = np.array([scipy_kurtosis(X_white[:, i], fisher=True) for i in range(k)])

    # --- Stability: match components across runs via cosine similarity ---
    stability_scores = np.ones(k)  # 1.0 = perfectly stable

    if n_runs > 1:
        similarities_per_component = [[] for _ in range(k)]

        for run_idx in range(1, n_runs):
            W_other = all_components[run_idx]
            # Cosine similarity matrix between reference and this run
            cos_sim = np.abs(W_ref @ W_other.T)  # (k, k), absolute because ICA sign is arbitrary

            # Greedy matching (Hungarian would be better but greedy is fine for diagnostics)
            matched = set()
            for i in range(k):
                # Find best unmatched partner for reference component i
                available = [j for j in range(k) if j not in matched]
                sims = cos_sim[i, available]
                best_local = np.argmax(sims)
                best_j = available[best_local]
                matched.add(best_j)
                similarities_per_component[i].append(cos_sim[i, best_j])

        stability_scores = np.array([np.mean(s) if s else 1.0 for s in similarities_per_component])

    # Project ICA directions back to original d-dimensional space
    # W_ref rows are directions in PCA space → multiply by V_k.T to get original space
    ica_directions_d = (W_ref @ V_k.T).T  # (d, k) — columns are IC directions in R^d
    # Re-normalize
    col_norms = np.linalg.norm(ica_directions_d, axis=0, keepdims=True)
    ica_directions_d = ica_directions_d / (col_norms + 1e-12)

    # Rank by stability * |kurtosis| (high = reliable and non-Gaussian)
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

    return {
        "ica_directions": ica_directions_d,  # (d, k) in original space
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
    """Compute mean and std for feature normalization before SAE training."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Avoid division by zero — clamp std
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

def action_distribution_analysis(actions: np.ndarray, n_actions: int = 7):
    """Analyze action distribution in the collected data."""
    action_names = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]
    counts = np.bincount(actions, minlength=n_actions)
    freqs = counts / counts.sum()

    print(f"\n{'='*60}")
    print(f"ACTION DISTRIBUTION")
    print(f"{'='*60}")
    print(f"  Total samples: {len(actions)}")
    for a in range(n_actions):
        bar = "█" * int(freqs[a] * 50)
        print(f"  {action_names[a]:10s}: {counts[a]:6d} ({freqs[a]*100:5.1f}%) {bar}")

    # Entropy
    freqs_nonzero = freqs[freqs > 0]
    entropy = -np.sum(freqs_nonzero * np.log2(freqs_nonzero))
    max_entropy = np.log2(n_actions)
    print(f"\n  Entropy: {entropy:.3f} / {max_entropy:.3f} (max)")
    print(f"  Normalized entropy: {entropy/max_entropy:.3f}")

    return {"counts": counts, "freqs": freqs, "entropy": entropy}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_diagnostics(svd_result, ica_result, norm_stats, action_stats, save_dir):
    """Generate diagnostic plots."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Stage 1: Feature Space Analysis", fontsize=14, fontweight="bold")

    # --- 1. Singular value spectrum ---
    ax = axes[0, 0]
    sv = svd_result["singular_values"]
    ax.semilogy(range(1, len(sv) + 1), sv, "b.-", markersize=3)
    ax.axvline(svd_result["k"], color="r", linestyle="--", alpha=0.7, label=f'k={svd_result["k"]}')
    ax.axvline(svd_result["k_elbow"], color="g", linestyle="--", alpha=0.7, label=f'k_elbow={svd_result["k_elbow"]}')
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value (log)")
    ax.set_title("Singular Value Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 2. Cumulative explained variance ---
    ax = axes[0, 1]
    cum = svd_result["cumulative_var"]
    ax.plot(range(1, len(cum) + 1), cum * 100, "b-", linewidth=2)
    ax.axhline(95, color="r", linestyle="--", alpha=0.5, label="95%")
    ax.axhline(99, color="orange", linestyle="--", alpha=0.5, label="99%")
    ax.axvline(svd_result["k"], color="r", linestyle="--", alpha=0.3)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Cumulative Explained Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 3. Per-component explained variance (top 30) ---
    ax = axes[0, 2]
    n_show = min(30, len(svd_result["explained_var"]))
    ax.bar(range(1, n_show + 1), svd_result["explained_var"][:n_show] * 100, color="steelblue", alpha=0.8)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"Per-Component Variance (top {n_show})")
    ax.grid(True, alpha=0.3)

    # --- 4. ICA kurtosis distribution ---
    ax = axes[1, 0]
    kurt = ica_result["kurtosis_values"]
    ranked_kurt = kurt[ica_result["ica_rank"]]
    colors = ["green" if ica_result["stability_scores"][ica_result["ica_rank"][i]] > 0.8 else "red"
              for i in range(len(ranked_kurt))]
    ax.bar(range(1, len(ranked_kurt) + 1), np.abs(ranked_kurt), color=colors, alpha=0.7)
    ax.set_xlabel("IC rank (by reliability)")
    ax.set_ylabel("|Kurtosis|")
    ax.set_title("ICA Components: |Kurtosis| (green=stable)")
    ax.grid(True, alpha=0.3)

    # --- 5. ICA stability scores ---
    ax = axes[1, 1]
    stab = ica_result["stability_scores"]
    sorted_stab = np.sort(stab)[::-1]
    ax.bar(range(1, len(sorted_stab) + 1), sorted_stab, color="teal", alpha=0.7)
    ax.axhline(0.8, color="r", linestyle="--", alpha=0.5, label="0.8 threshold")
    ax.axhline(0.9, color="orange", linestyle="--", alpha=0.5, label="0.9 threshold")
    ax.set_xlabel("IC rank (by stability)")
    ax.set_ylabel("Stability score")
    ax.set_title("ICA Stability Across Runs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 6. Feature dimension statistics ---
    ax = axes[1, 2]
    std = norm_stats["std"]
    ax.hist(std, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Per-dimension std")
    ax.set_ylabel("Count")
    ax.set_title("Feature Std Distribution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "stage1_diagnostics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Diagnostic plot saved: {plot_path}")

    # --- Separate action distribution plot ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    action_names = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]
    freqs = action_stats["freqs"]
    bars = ax2.bar(action_names, freqs * 100, color="steelblue", alpha=0.8, edgecolor="black")
    ax2.set_ylabel("Frequency (%)")
    ax2.set_title("Action Distribution in Rollout Data")
    ax2.grid(True, alpha=0.3, axis="y")
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

def save_stage1_outputs(svd_result, ica_result, norm_stats, action_stats, save_dir):
    """Save all Stage 1 outputs for use in Stage 2 (SAE training)."""
    os.makedirs(save_dir, exist_ok=True)

    # Save as a single .pt file for easy loading
    stage1_data = {
        # SVD
        "V_k": torch.from_numpy(svd_result["V_k"]).float(),          # (d, k)
        "V_noise": torch.from_numpy(svd_result["V_noise"]).float(),  # (d, d-k)
        "singular_values": torch.from_numpy(svd_result["singular_values"]).float(),
        "explained_var": torch.from_numpy(svd_result["explained_var"]).float(),
        "cumulative_var": torch.from_numpy(svd_result["cumulative_var"]).float(),
        "k": svd_result["k"],
        "k_elbow": svd_result["k_elbow"],
        "feature_mean_svd": torch.from_numpy(svd_result["mean"]).float(),
        # ICA
        "ica_directions": torch.from_numpy(ica_result["ica_directions"]).float(),  # (d, k)
        "ica_kurtosis": torch.from_numpy(ica_result["kurtosis_values"]).float(),
        "ica_stability": torch.from_numpy(ica_result["stability_scores"]).float(),
        "ica_reliability": torch.from_numpy(ica_result["reliability"]).float(),
        "ica_rank": torch.from_numpy(ica_result["ica_rank"].copy()).long(),
        # Normalization
        "feature_mean": torch.from_numpy(norm_stats["mean"]).float(),
        "feature_std": torch.from_numpy(norm_stats["std"]).float(),
        # Action distribution
        "action_counts": torch.from_numpy(action_stats["counts"]).long(),
        "action_freqs": torch.from_numpy(action_stats["freqs"]).float(),
        "action_entropy": action_stats["entropy"],
    }

    save_path = os.path.join(save_dir, "stage1_outputs.pt")
    torch.save(stage1_data, save_path)
    print(f"\n  Stage 1 outputs saved: {save_path}")

    # Also save a human-readable summary
    summary = {
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
    }

    summary_path = os.path.join(save_dir, "stage1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Stage 1 summary saved: {summary_path}")

    return save_path


def load_stage1_outputs(path):
    """Load Stage 1 outputs for use in Stage 2."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Loaded Stage 1 outputs from {path}")
    print(f"  Signal dimension k: {data['k']}")
    print(f"  ICA directions shape: {data['ica_directions'].shape}")
    return data


# ---------------------------------------------------------------------------
# Data collection helper (reuses logic from ConceptExtractor)
# ---------------------------------------------------------------------------

def collect_features(model_path, env_name, n_episodes=800, seed=42):
    """Collect features and actions from a frozen PPO policy."""
    import gymnasium as gym
    import minigrid  # noqa: F401
    from minigrid.wrappers import ImgObsWrapper
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    from tqdm import tqdm

    print(f"Loading PPO model from {model_path}...")
    model = PPO.load(model_path)

    def make_env():
        def _init():
            env = gym.make(env_name)
            env = ImgObsWrapper(env)
            return env
        return _init

    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    features_list = []
    actions_list = []

    print(f"Collecting {n_episodes} episodes...")
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
            obs, rewards, dones, infos = env.step(action)
            if dones[0]:
                episode_count += 1
                pbar.update(1)
                obs = env.reset()
        pbar.close()

    env.close()

    features = torch.cat(features_list, dim=0)
    actions = torch.cat(actions_list, dim=0)
    print(f"Collected {len(features)} samples, feature dim = {features.shape[1]}")

    return features, actions


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stage1(features: torch.Tensor, actions: torch.Tensor,
               variance_threshold: float = 0.95,
               ica_n_runs: int = 5,
               seed: int = 42,
               save_dir: str = "./stage1_outputs"):
    """
    Run the full Stage 1 analysis pipeline.

    Args:
        features          : (N, d) tensor of frozen PPO features
        actions           : (N,) tensor of action indices
        variance_threshold: cumulative variance for signal dim k
        ica_n_runs        : number of ICA runs for stability assessment
        seed              : random seed
        save_dir          : where to save outputs

    Returns:
        stage1_data dict (same as saved .pt file)
    """
    X = features.numpy()
    A = actions.numpy().astype(int)

    print(f"\n{'#'*60}")
    print(f"  STAGE 1: FEATURE SPACE ANALYSIS")
    print(f"  N = {X.shape[0]}, d = {X.shape[1]}")
    print(f"{'#'*60}")

    # 1a. SVD
    svd_result = svd_analysis(X, variance_threshold=variance_threshold)

    # 1b. ICA on PCA-reduced features
    ica_result = ica_analysis(
        X, k=svd_result["k"], V_k=svd_result["V_k"],
        n_runs=ica_n_runs, seed=seed
    )

    # Normalization stats
    norm_stats = compute_normalization_stats(X)

    # Action distribution
    action_stats = action_distribution_analysis(A)

    # Plots
    plot_diagnostics(svd_result, ica_result, norm_stats, action_stats, save_dir)

    # Save
    save_path = save_stage1_outputs(svd_result, ica_result, norm_stats, action_stats, save_dir)

    # Print recommendations for Stage 2
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
    parser = argparse.ArgumentParser(description="Stage 1: Feature Space Analysis")
    parser.add_argument("--features_path", type=str, default=None,
                        help="Path to pre-collected features .pt file (with 'features' and 'actions' keys)")
    parser.add_argument("--model_path", type=str, default="ppo_doorkey_5x5.zip",
                        help="PPO model path (used if --features_path not given)")
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-5x5-v0",
                        help="Environment name (used if --features_path not given)")
    parser.add_argument("--n_episodes", type=int, default=800,
                        help="Number of episodes to collect")
    parser.add_argument("--variance_threshold", type=float, default=0.95,
                        help="Cumulative variance threshold for signal dim k")
    parser.add_argument("--ica_n_runs", type=int, default=5,
                        help="Number of ICA runs for stability check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./stage1_outputs")

    args = parser.parse_args()

    # Load or collect features
    if args.features_path is not None:
        print(f"Loading features from {args.features_path}...")
        data = torch.load(args.features_path, map_location="cpu", weights_only=False)
        features = data["features"]
        actions = data["actions"]
    else:
        features, actions = collect_features(
            args.model_path, args.env_name, args.n_episodes, args.seed
        )
        # Save raw collected data for reuse
        os.makedirs(args.save_dir, exist_ok=True)
        raw_path = os.path.join(args.save_dir, "collected_data.pt")
        torch.save({"features": features, "actions": actions}, raw_path)
        print(f"Raw data saved: {raw_path}")

    # Run Stage 1
    stage1_data = run_stage1(
        features, actions,
        variance_threshold=args.variance_threshold,
        ica_n_runs=args.ica_n_runs,
        seed=args.seed,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()