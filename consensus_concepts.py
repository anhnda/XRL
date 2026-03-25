"""
Stage 3: Consensus Concept Extraction & Grounding
===================================================
Takes M trained SAEs from Stage 2 and:
  1. Collects all decoder columns across runs
  2. Clusters them in direction space (spherical k-means / agglomerative)
  3. Identifies consensus concepts (appear in >= M_min runs)
  4. Computes CKA stability across run pairs
  5. Grounds concepts via NMI with environment properties
  6. Computes monosemanticity scores and auto-labels concepts
  7. Selects the best single run and retrains action predictor on consensus concepts

Usage:
    python consensus_concepts.py \\
        --stage2_dir ./stage2_outputs \\
        --features_path ./stage1_outputs/collected_data.pt \\
        --stage1_path ./stage1_outputs/stage1_outputs.pt \\
        --save_dir ./stage3_outputs

    # With environment grounding (requires MiniGrid + PPO model)
    python consensus_concepts.py \\
        --stage2_dir ./stage2_outputs \\
        --features_path ./stage1_outputs/collected_data.pt \\
        --stage1_path ./stage1_outputs/stage1_outputs.pt \\
        --model_path ppo_doorkey_5x5.zip \\
        --env_name MiniGrid-DoorKey-5x5-v0 \\
        --save_dir ./stage3_outputs
"""

import argparse
import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


# ============================================================================
# 3.1  Collect decoder columns from all runs
# ============================================================================

def collect_decoder_columns(stage2_dir: str) -> dict:
    """
    Load decoder matrices from all Stage 2 runs.

    Returns:
        dict with:
            columns    : (M*D, d) tensor — all decoder columns stacked
            run_ids    : (M*D,) int array — which run each column came from
            col_ids    : (M*D,) int array — column index within its run
            active_rates: (M*D,) float — activation rate per column
            n_runs     : int
            hidden_dim : int
            input_dim  : int
    """
    run_dirs = sorted(glob.glob(os.path.join(stage2_dir, "run_*")))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {stage2_dir}")

    all_columns = []
    all_run_ids = []
    all_col_ids = []
    all_active_rates = []

    for run_dir in run_dirs:
        sae_data = torch.load(
            os.path.join(run_dir, "sae_weights.pt"),
            map_location="cpu", weights_only=True
        )
        analysis = torch.load(
            os.path.join(run_dir, "analysis.pt"),
            map_location="cpu", weights_only=True
        )

        W_d = sae_data["W_d"]  # (d, D)
        active_rate = analysis["active_rate"]  # (D,)

        # Extract run_id from directory name
        run_id = int(os.path.basename(run_dir).split("_")[1])

        d, D = W_d.shape
        for j in range(D):
            col = W_d[:, j]
            # Normalize to unit norm (should already be, but defensive)
            col = col / (col.norm() + 1e-8)
            all_columns.append(col)
            all_run_ids.append(run_id)
            all_col_ids.append(j)
            all_active_rates.append(active_rate[j].item())

    columns = torch.stack(all_columns)  # (M*D, d)
    run_ids = np.array(all_run_ids)
    col_ids = np.array(all_col_ids)
    active_rates = np.array(all_active_rates)

    n_runs = len(run_dirs)
    hidden_dim = D
    input_dim = d

    print(f"\n{'='*60}")
    print(f"COLLECTED DECODER COLUMNS")
    print(f"{'='*60}")
    print(f"  Runs: {n_runs}")
    print(f"  Columns per run: {hidden_dim}")
    print(f"  Total columns: {len(columns)}")
    print(f"  Feature dim: {input_dim}")
    print(f"  Mean active rate: {active_rates.mean():.4f}")
    print(f"  Active columns (>1%): {(active_rates > 0.01).sum()}/{len(active_rates)}")

    return {
        "columns": columns,
        "run_ids": run_ids,
        "col_ids": col_ids,
        "active_rates": active_rates,
        "n_runs": n_runs,
        "hidden_dim": hidden_dim,
        "input_dim": input_dim,
    }


# ============================================================================
# 3.2  Cluster decoder columns in direction space
# ============================================================================

def cluster_decoder_columns(
    columns: torch.Tensor,
    active_rates: np.ndarray,
    min_active_rate: float = 0.005,
    distance_threshold: float = 0.3,
) -> dict:
    """
    Cluster decoder columns using agglomerative clustering with cosine distance.

    We use agglomerative clustering because:
      - It doesn't require specifying the number of clusters a priori
      - It handles variable-density clusters well
      - Cosine distance on unit vectors = angular distance, which is natural
        for comparing directions

    Args:
        columns          : (N, d) unit-norm decoder columns
        active_rates     : (N,) activation frequency per column
        min_active_rate  : discard columns below this rate (dead concepts)
        distance_threshold: cosine distance threshold for cluster merging

    Returns:
        dict with:
            labels          : (N,) cluster label per column (-1 = dead/discarded)
            centroids       : (n_clusters, d) cluster centroids (unit norm)
            cluster_sizes   : (n_clusters,) number of members per cluster
            n_clusters      : int
            active_mask     : (N,) bool — which columns were included
    """
    N, d = columns.shape

    # Filter out dead columns
    active_mask = active_rates > min_active_rate
    n_active = active_mask.sum()

    print(f"\n{'='*60}")
    print(f"CLUSTERING DECODER COLUMNS")
    print(f"{'='*60}")
    print(f"  Total columns: {N}")
    print(f"  Active (>{min_active_rate*100:.1f}%): {n_active}")
    print(f"  Dead (discarded): {N - n_active}")

    if n_active < 2:
        print("  WARNING: fewer than 2 active columns, skipping clustering")
        return {
            "labels": np.full(N, -1),
            "centroids": torch.zeros(0, d),
            "cluster_sizes": np.array([]),
            "n_clusters": 0,
            "active_mask": active_mask,
        }

    active_cols = columns[active_mask].numpy()  # (n_active, d)

    # Cosine distance: 1 - |cos(a, b)|
    # We use absolute cosine because direction sign is arbitrary
    # (concept j and -concept j represent the same direction)
    cos_sim = np.abs(active_cols @ active_cols.T)
    cos_sim = np.clip(cos_sim, 0, 1)
    cos_dist = 1.0 - cos_sim

    # Convert to condensed distance matrix for scipy
    condensed = pdist(active_cols, metric=lambda u, v: 1.0 - abs(np.dot(u, v)))

    # Agglomerative clustering
    Z = linkage(condensed, method="average")
    cluster_labels_active = fcluster(Z, t=distance_threshold, criterion="distance")
    # fcluster labels start at 1; shift to 0-indexed
    cluster_labels_active = cluster_labels_active - 1
    n_clusters = cluster_labels_active.max() + 1

    # Compute centroids
    centroids = []
    cluster_sizes = []
    for c in range(n_clusters):
        mask = cluster_labels_active == c
        members = active_cols[mask]
        # Sign-align members to the first one before averaging
        ref = members[0]
        signs = np.sign(members @ ref)
        aligned = members * signs[:, None]
        centroid = aligned.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
        cluster_sizes.append(mask.sum())

    centroids = torch.from_numpy(np.stack(centroids)).float()  # (n_clusters, d)
    cluster_sizes = np.array(cluster_sizes)

    # Map back to full label array
    labels = np.full(N, -1)
    active_indices = np.where(active_mask)[0]
    for i, idx in enumerate(active_indices):
        labels[idx] = cluster_labels_active[i]

    print(f"  Clusters found: {n_clusters}")
    print(f"  Cluster size range: [{cluster_sizes.min()}, {cluster_sizes.max()}]")
    print(f"  Mean cluster size: {cluster_sizes.mean():.1f}")

    return {
        "labels": labels,
        "centroids": centroids,
        "cluster_sizes": cluster_sizes,
        "n_clusters": n_clusters,
        "active_mask": active_mask,
    }


# ============================================================================
# 3.3  Identify consensus concepts
# ============================================================================

def identify_consensus(
    cluster_result: dict,
    run_ids: np.ndarray,
    n_runs: int,
    m_min_frac: float = 0.8,
) -> dict:
    """
    A consensus concept = a cluster containing members from >= M_min runs.

    Args:
        cluster_result : output of cluster_decoder_columns
        run_ids        : (N,) which run each column came from
        n_runs         : total number of runs
        m_min_frac     : fraction of runs required (0.8 = 80%)

    Returns:
        dict with:
            consensus_indices  : list of cluster indices that are consensus
            consensus_centroids: (K_consensus, d) tensor
            consensus_run_counts: (K_consensus,) how many runs each appears in
            stability_curve    : dict mapping m_min -> K_consensus
            per_cluster_runs   : list of sets — which runs each cluster appears in
    """
    labels = cluster_result["labels"]
    centroids = cluster_result["centroids"]
    n_clusters = cluster_result["n_clusters"]

    m_min = max(1, int(np.ceil(m_min_frac * n_runs)))

    # For each cluster, find which runs contributed members
    per_cluster_runs = []
    for c in range(n_clusters):
        member_mask = labels == c
        runs_in_cluster = set(run_ids[member_mask].tolist())
        per_cluster_runs.append(runs_in_cluster)

    # Consensus = clusters with >= m_min distinct runs
    consensus_indices = []
    consensus_run_counts = []
    for c in range(n_clusters):
        n_contributing_runs = len(per_cluster_runs[c])
        if n_contributing_runs >= m_min:
            consensus_indices.append(c)
            consensus_run_counts.append(n_contributing_runs)

    consensus_centroids = centroids[consensus_indices] if consensus_indices else torch.zeros(0, centroids.shape[1])
    consensus_run_counts = np.array(consensus_run_counts) if consensus_run_counts else np.array([])

    # Stability curve: sweep m_min from 1 to n_runs
    stability_curve = {}
    for m in range(1, n_runs + 1):
        k_cons = sum(1 for c in range(n_clusters) if len(per_cluster_runs[c]) >= m)
        stability_curve[m] = k_cons

    print(f"\n{'='*60}")
    print(f"CONSENSUS CONCEPTS (M_min = {m_min}, {m_min_frac*100:.0f}% of {n_runs} runs)")
    print(f"{'='*60}")
    print(f"  Total clusters: {n_clusters}")
    print(f"  Consensus concepts: {len(consensus_indices)}")
    if len(consensus_run_counts) > 0:
        print(f"  Run coverage range: [{consensus_run_counts.min()}, {consensus_run_counts.max()}]")
    print(f"\n  Stability curve (m_min -> K_consensus):")
    for m, k in sorted(stability_curve.items()):
        bar = "█" * k
        marker = " ← selected" if m == m_min else ""
        print(f"    m_min={m:2d}: {k:3d} concepts {bar}{marker}")

    return {
        "consensus_indices": consensus_indices,
        "consensus_centroids": consensus_centroids,
        "consensus_run_counts": consensus_run_counts,
        "stability_curve": stability_curve,
        "per_cluster_runs": per_cluster_runs,
        "m_min": m_min,
    }


# ============================================================================
# 3.4  CKA Stability
# ============================================================================

def compute_cka(
    features: torch.Tensor,
    stage2_dir: str,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    n_sample: int = 2000,
) -> dict:
    """
    Compute CKA (Centered Kernel Alignment) between concept activation
    matrices from different SAE runs.

    CKA is invariant to permutation and orthogonal transformation of
    individual concepts, providing an aggregate geometric stability measure.

    Returns:
        dict with:
            cka_matrix   : (M, M) pairwise CKA values
            mean_cka     : float — mean off-diagonal CKA
            min_cka      : float
    """
    from sparse_concept_autoencoder import load_run

    run_dirs = sorted(glob.glob(os.path.join(stage2_dir, "run_*")))
    M = len(run_dirs)

    # Normalize features
    features_norm = (features[:n_sample] - feat_mean) / feat_std

    # Collect concept activations from each run
    all_activations = []
    for run_dir in run_dirs:
        run_data = load_run(run_dir)
        model = run_data["model"]
        model.eval()
        with torch.no_grad():
            _, z_sparse, _ = model(features_norm)
            all_activations.append(z_sparse)  # (n_sample, D)

    # CKA computation
    def linear_cka(X, Y):
        """Linear CKA between two activation matrices."""
        # X: (n, p), Y: (n, q)
        # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        YtX = Y.t() @ X  # (q, p)
        XtX = X.t() @ X  # (p, p)
        YtY = Y.t() @ Y  # (q, q)

        numerator = (YtX * YtX).sum()
        denominator = torch.sqrt((XtX * XtX).sum() * (YtY * YtY).sum()) + 1e-10

        return (numerator / denominator).item()

    cka_matrix = np.ones((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            cka_val = linear_cka(all_activations[i], all_activations[j])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val

    # Off-diagonal stats
    mask = ~np.eye(M, dtype=bool)
    off_diag = cka_matrix[mask]
    mean_cka = off_diag.mean()
    min_cka = off_diag.min()

    print(f"\n{'='*60}")
    print(f"CKA STABILITY ({M} runs)")
    print(f"{'='*60}")
    print(f"  Mean CKA: {mean_cka:.4f}")
    print(f"  Min  CKA: {min_cka:.4f}")
    print(f"  Max  CKA: {off_diag.max():.4f}")

    if M <= 10:
        print(f"\n  CKA matrix:")
        header = "       " + "  ".join([f"R{i:02d}" for i in range(M)])
        print(header)
        for i in range(M):
            row = f"  R{i:02d}  " + "  ".join([f"{cka_matrix[i,j]:.2f}" for j in range(M)])
            print(row)

    return {
        "cka_matrix": cka_matrix,
        "mean_cka": mean_cka,
        "min_cka": min_cka,
    }


# ============================================================================
# 3.5  Concept Grounding via NMI
# ============================================================================

def extract_ground_truth_properties(
    model_path: str,
    env_name: str,
    n_episodes: int = 200,
    seed: int = 42,
) -> dict:
    """
    Collect ground-truth environment properties alongside observations.

    For MiniGrid, extracts binary properties from the full grid state:
      - key_visible, key_adjacent, key_facing, holding_key
      - door_visible, door_adjacent, door_facing, door_open
      - goal_visible, goal_adjacent, goal_facing
      - facing_wall, facing_empty

    Returns:
        dict with:
            properties : (N, P) bool array — ground-truth property values
            property_names : list of str
            features   : (N, d) tensor — corresponding PPO features
            actions    : (N,) tensor
    """
    import gymnasium as gym
    import minigrid  # noqa: F401
    from minigrid.wrappers import ImgObsWrapper
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    from tqdm import tqdm

    model = PPO.load(model_path)

    def make_env():
        def _init():
            env = gym.make(env_name)
            env = ImgObsWrapper(env)
            return env
        return _init

    # We need the unwrapped env for state access
    raw_env = gym.make(env_name)
    raw_env = ImgObsWrapper(raw_env)

    vec_env = DummyVecEnv([make_env()])
    vec_env = VecTransposeImage(vec_env)

    features_list = []
    actions_list = []
    properties_list = []

    property_names = [
        "key_visible", "key_adjacent", "key_facing", "holding_key",
        "door_visible", "door_adjacent", "door_facing", "door_open",
        "goal_visible", "goal_adjacent",
        "facing_wall", "facing_empty",
    ]

    def get_properties(env):
        """Extract binary properties from MiniGrid state."""
        props = {}
        grid = env.unwrapped.grid
        agent_pos = env.unwrapped.agent_pos
        agent_dir = env.unwrapped.agent_dir

        # Direction vectors
        dir_vec = env.unwrapped.dir_vec
        fwd_pos = agent_pos + dir_vec

        # Get front cell
        fwd_cell = grid.get(*fwd_pos) if (
            0 <= fwd_pos[0] < grid.width and 0 <= fwd_pos[1] < grid.height
        ) else None

        # Check what agent is carrying
        carrying = env.unwrapped.carrying

        # Find objects
        key_pos = None
        door_pos = None
        goal_pos = None
        door_is_open = False

        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None:
                    if cell.type == "key":
                        key_pos = np.array([x, y])
                    elif cell.type == "door":
                        door_pos = np.array([x, y])
                        door_is_open = cell.is_open
                    elif cell.type == "goal":
                        goal_pos = np.array([x, y])

        # Helper: is position adjacent to agent?
        def is_adjacent(pos):
            if pos is None:
                return False
            return abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1]) == 1

        # Helper: is agent facing position?
        def is_facing(pos):
            if pos is None:
                return False
            return np.array_equal(fwd_pos, pos)

        # Helper: is position in agent's field of view?
        # Simple approximation: within Manhattan distance 3 and in front half
        def is_visible(pos):
            if pos is None:
                return False
            diff = pos - agent_pos
            dist = abs(diff[0]) + abs(diff[1])
            if dist > 5:
                return False
            # Check if roughly in the forward direction
            dot = diff[0] * dir_vec[0] + diff[1] * dir_vec[1]
            return dot >= 0 or dist <= 1

        # Key properties
        holding = carrying is not None and carrying.type == "key"
        props["key_visible"] = (not holding) and key_pos is not None and is_visible(key_pos)
        props["key_adjacent"] = (not holding) and key_pos is not None and is_adjacent(key_pos)
        props["key_facing"] = (not holding) and key_pos is not None and is_facing(key_pos)
        props["holding_key"] = holding

        # Door properties
        props["door_visible"] = door_pos is not None and is_visible(door_pos)
        props["door_adjacent"] = door_pos is not None and is_adjacent(door_pos)
        props["door_facing"] = door_pos is not None and is_facing(door_pos)
        props["door_open"] = door_is_open

        # Goal properties
        props["goal_visible"] = goal_pos is not None and is_visible(goal_pos)
        props["goal_adjacent"] = goal_pos is not None and is_adjacent(goal_pos)

        # Wall / empty facing
        props["facing_wall"] = fwd_cell is not None and fwd_cell.type == "wall"
        props["facing_empty"] = fwd_cell is None

        return [bool(props[name]) for name in property_names]

    obs = vec_env.reset()
    # Also reset raw env for state tracking
    raw_env.reset(seed=seed)
    episode_count = 0

    print(f"\n  Collecting ground-truth properties ({n_episodes} episodes)...")
    pbar = tqdm(total=n_episodes)

    while episode_count < n_episodes:
        action, _ = model.predict(obs, deterministic=False)

        # Get features
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float().to(model.device)
            feats = model.policy.features_extractor(obs_tensor).cpu()

        # Get properties from raw env state
        try:
            props = get_properties(raw_env)
        except Exception:
            props = [False] * len(property_names)

        features_list.append(feats)
        actions_list.append(torch.tensor(action))
        properties_list.append(props)

        obs, rewards, dones, infos = vec_env.step(action)
        raw_env.step(action[0])

        if dones[0]:
            episode_count += 1
            pbar.update(1)
            obs = vec_env.reset()
            raw_env.reset()

    pbar.close()
    vec_env.close()
    raw_env.close()

    features = torch.cat(features_list, dim=0)
    actions = torch.cat(actions_list, dim=0)
    properties = np.array(properties_list, dtype=bool)  # (N, P)

    print(f"  Collected {len(features)} samples with {len(property_names)} properties")

    # Property statistics
    for i, name in enumerate(property_names):
        rate = properties[:, i].mean()
        print(f"    {name:20s}: {rate*100:5.1f}%")

    return {
        "properties": properties,
        "property_names": property_names,
        "features": features,
        "actions": actions,
    }


def compute_nmi_grounding(
    consensus_centroids: torch.Tensor,
    features: torch.Tensor,
    properties: np.ndarray,
    property_names: List[str],
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    k: int = 10,
) -> dict:
    """
    Compute NMI between each consensus concept's activation and each
    ground-truth property. Assign monosemanticity scores and labels.

    We activate concepts by projecting features onto consensus directions
    and applying TopK + thresholding.

    Returns:
        dict with:
            nmi_matrix      : (K_consensus, P) NMI values
            mono_scores     : (K_consensus,) monosemanticity scores
            labels          : list of str — best-matching property per concept
            best_nmi        : (K_consensus,) best NMI per concept
            second_nmi      : (K_consensus,) second-best NMI
    """
    from sklearn.metrics import normalized_mutual_info_score

    K = len(consensus_centroids)
    P = properties.shape[1]

    # Normalize features
    features_norm = (features - feat_mean) / feat_std

    # Project features onto consensus directions
    # Each concept's activation = features_norm @ centroid
    activations = features_norm @ consensus_centroids.t()  # (N, K)

    # Binarize: threshold at median activation per concept
    binary_concepts = np.zeros_like(activations.numpy(), dtype=bool)
    for i in range(K):
        act = activations[:, i].numpy()
        threshold = np.median(act[act > 0]) if (act > 0).any() else 0.0
        binary_concepts[:, i] = act > threshold

    # Compute NMI matrix
    nmi_matrix = np.zeros((K, P))
    for i in range(K):
        for j in range(P):
            # Skip if either is constant (NMI undefined)
            if binary_concepts[:, i].std() < 1e-8 or properties[:, j].std() < 1e-8:
                nmi_matrix[i, j] = 0.0
            else:
                nmi_matrix[i, j] = normalized_mutual_info_score(
                    binary_concepts[:, i].astype(int),
                    properties[:, j].astype(int),
                )

    # Monosemanticity scores
    mono_scores = np.zeros(K)
    best_nmi = np.zeros(K)
    second_nmi = np.zeros(K)
    labels = []

    for i in range(K):
        sorted_nmi = np.sort(nmi_matrix[i])[::-1]
        best_nmi[i] = sorted_nmi[0]
        second_nmi[i] = sorted_nmi[1] if P > 1 else 0.0
        mono_scores[i] = best_nmi[i] / (second_nmi[i] + 1e-4)

        best_prop_idx = np.argmax(nmi_matrix[i])
        labels.append(property_names[best_prop_idx])

    print(f"\n{'='*60}")
    print(f"CONCEPT GROUNDING (NMI)")
    print(f"{'='*60}")
    print(f"  Consensus concepts: {K}")
    print(f"  Ground-truth properties: {P}")
    print(f"  Mean best NMI: {best_nmi.mean():.4f}")
    print(f"  Mean mono score: {mono_scores.mean():.2f}")
    print(f"  Monosemantic (>2.0): {(mono_scores > 2.0).sum()}/{K}")
    print(f"\n  Concept labels:")
    for i in range(K):
        mono_tag = "✓" if mono_scores[i] > 2.0 else "✗"
        print(f"    C{i:02d}: {labels[i]:20s}  "
              f"NMI={best_nmi[i]:.3f}  Mono={mono_scores[i]:.2f} {mono_tag}")

    return {
        "nmi_matrix": nmi_matrix,
        "mono_scores": mono_scores,
        "labels": labels,
        "best_nmi": best_nmi,
        "second_nmi": second_nmi,
        "binary_concepts": binary_concepts,
    }


def compute_nmi_grounding_synthetic(
    consensus_centroids: torch.Tensor,
    features: torch.Tensor,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
) -> dict:
    """
    Fallback grounding when no environment properties are available.
    Reports per-concept activation statistics instead of NMI.
    """
    K = len(consensus_centroids)
    features_norm = (features - feat_mean) / feat_std
    activations = features_norm @ consensus_centroids.t()  # (N, K)

    print(f"\n{'='*60}")
    print(f"CONCEPT ACTIVATION STATISTICS (no env grounding)")
    print(f"{'='*60}")
    for i in range(K):
        act = activations[:, i].numpy()
        pos_rate = (act > 0).mean()
        print(f"    C{i:02d}: active={pos_rate*100:5.1f}%  "
              f"mean={act.mean():.3f}  std={act.std():.3f}  "
              f"max={act.max():.3f}")

    return {
        "nmi_matrix": None,
        "mono_scores": np.zeros(K),
        "labels": [f"concept_{i}" for i in range(K)],
        "best_nmi": np.zeros(K),
        "second_nmi": np.zeros(K),
    }


# ============================================================================
# 3.6  Select best run & map consensus concepts
# ============================================================================

def select_best_run(
    stage2_dir: str,
    consensus_centroids: torch.Tensor,
    consensus_labels: List[str],
) -> dict:
    """
    Select the best SAE run (highest action accuracy) and map its
    concepts to consensus concepts via cosine similarity matching.

    Returns:
        dict with:
            best_run_dir        : str
            best_run_id         : int
            concept_mapping     : (K_consensus,) — index in best run's hidden dim
            mapping_similarities: (K_consensus,) — cosine similarity of match
    """
    run_dirs = sorted(glob.glob(os.path.join(stage2_dir, "run_*")))

    best_acc = -1
    best_dir = None
    best_id = -1

    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "metadata.json")) as f:
            meta = json.load(f)
        acc = meta["final_action_acc"]
        rid = meta["run_id"]
        if acc > best_acc:
            best_acc = acc
            best_dir = run_dir
            best_id = rid

    # Load best run's decoder
    sae_data = torch.load(
        os.path.join(best_dir, "sae_weights.pt"),
        map_location="cpu", weights_only=True
    )
    W_d = sae_data["W_d"]  # (d, D)

    # Normalize columns
    W_d_norm = W_d / (W_d.norm(dim=0, keepdim=True) + 1e-8)

    # Match consensus centroids to best run columns
    # Cosine similarity: consensus_centroids (K, d) @ W_d_norm (d, D) -> (K, D)
    K = len(consensus_centroids)
    sim_matrix = torch.abs(consensus_centroids @ W_d_norm)  # (K, D)

    # Greedy matching (no duplicates)
    concept_mapping = []
    mapping_sims = []
    used = set()

    for i in range(K):
        sims = sim_matrix[i].numpy()
        # Mask already-used columns
        for u in used:
            sims[u] = -1
        best_col = int(np.argmax(sims))
        concept_mapping.append(best_col)
        mapping_sims.append(float(sim_matrix[i, best_col]))
        used.add(best_col)

    concept_mapping = np.array(concept_mapping)
    mapping_sims = np.array(mapping_sims)

    print(f"\n{'='*60}")
    print(f"BEST RUN SELECTION & CONCEPT MAPPING")
    print(f"{'='*60}")
    print(f"  Best run: {best_id} (acc={best_acc:.2f}%)")
    print(f"  Consensus -> best run mapping:")
    for i in range(K):
        print(f"    C{i:02d} ({consensus_labels[i]:20s}) -> col {concept_mapping[i]:3d}  "
              f"(sim={mapping_sims[i]:.3f})")
    print(f"  Mean mapping similarity: {mapping_sims.mean():.3f}")

    return {
        "best_run_dir": best_dir,
        "best_run_id": best_id,
        "best_run_acc": best_acc,
        "concept_mapping": concept_mapping,
        "mapping_similarities": mapping_sims,
    }


# ============================================================================
# 3.7  Visualization
# ============================================================================

def plot_stage3_diagnostics(
    cluster_result: dict,
    consensus_result: dict,
    cka_result: Optional[dict],
    grounding_result: dict,
    save_dir: str,
):
    """Generate Stage 3 diagnostic plots."""
    os.makedirs(save_dir, exist_ok=True)

    n_plots = 4 if cka_result else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    fig.suptitle("Stage 3: Consensus Concept Extraction", fontsize=14, fontweight="bold")

    # 1. Cluster size distribution
    ax = axes[0]
    sizes = cluster_result["cluster_sizes"]
    if len(sizes) > 0:
        ax.bar(range(len(sizes)), sorted(sizes, reverse=True), color="steelblue", alpha=0.8)
    ax.set_xlabel("Cluster rank")
    ax.set_ylabel("Size (columns)")
    ax.set_title("Cluster Size Distribution")
    ax.grid(True, alpha=0.3)

    # 2. Stability curve
    ax = axes[1]
    curve = consensus_result["stability_curve"]
    ms = sorted(curve.keys())
    ks = [curve[m] for m in ms]
    ax.plot(ms, ks, "bo-", linewidth=2, markersize=8)
    ax.axvline(consensus_result["m_min"], color="r", linestyle="--",
               alpha=0.7, label=f"M_min={consensus_result['m_min']}")
    ax.set_xlabel("M_min (required runs)")
    ax.set_ylabel("K_consensus")
    ax.set_title("Stability Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Monosemanticity scores
    ax = axes[2]
    mono = grounding_result["mono_scores"]
    if len(mono) > 0:
        labels = grounding_result["labels"]
        sorted_idx = np.argsort(mono)[::-1]
        colors = ["green" if mono[i] > 2.0 else "orange" if mono[i] > 1.0 else "red"
                  for i in sorted_idx]
        ax.barh(range(len(mono)), mono[sorted_idx], color=colors, alpha=0.7)
        ax.set_yticks(range(len(mono)))
        ax.set_yticklabels([f"C{i}:{labels[i][:12]}" for i in sorted_idx], fontsize=7)
        ax.axvline(2.0, color="green", linestyle="--", alpha=0.5, label="Mono>2")
    ax.set_xlabel("Monosemanticity Score")
    ax.set_title("Concept Monosemanticity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. CKA matrix
    if cka_result:
        ax = axes[3]
        M = len(cka_result["cka_matrix"])
        im = ax.imshow(cka_result["cka_matrix"], vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(M))
        ax.set_yticks(range(M))
        ax.set_xticklabels([f"R{i}" for i in range(M)], fontsize=8)
        ax.set_yticklabels([f"R{i}" for i in range(M)], fontsize=8)
        ax.set_title(f"CKA Matrix (mean={cka_result['mean_cka']:.3f})")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(save_dir, "stage3_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Diagnostic plot saved: {path}")

    # NMI heatmap (separate figure if grounding available)
    if grounding_result["nmi_matrix"] is not None:
        nmi = grounding_result["nmi_matrix"]
        K, P = nmi.shape
        fig2, ax2 = plt.subplots(figsize=(max(8, P * 0.8), max(4, K * 0.4)))
        im = ax2.imshow(nmi, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax2.set_xticks(range(P))
        ax2.set_xticklabels(grounding_result.get("property_names",
                            [f"p{j}" for j in range(P)]), rotation=45, ha="right", fontsize=8)
        ax2.set_yticks(range(K))
        ax2.set_yticklabels([f"C{i}:{grounding_result['labels'][i][:15]}" for i in range(K)],
                            fontsize=8)
        ax2.set_title("NMI: Concepts × Environment Properties")
        plt.colorbar(im, ax=ax2, shrink=0.8)
        plt.tight_layout()
        nmi_path = os.path.join(save_dir, "nmi_heatmap.png")
        plt.savefig(nmi_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  NMI heatmap saved: {nmi_path}")


# ============================================================================
# 3.8  Save / Load
# ============================================================================

def save_stage3_outputs(
    cluster_result: dict,
    consensus_result: dict,
    cka_result: Optional[dict],
    grounding_result: dict,
    best_run_result: dict,
    col_data: dict,
    save_dir: str,
):
    """Save all Stage 3 outputs."""
    os.makedirs(save_dir, exist_ok=True)

    stage3_data = {
        # Consensus concepts
        "consensus_centroids": consensus_result["consensus_centroids"],
        "consensus_indices": consensus_result["consensus_indices"],
        "consensus_run_counts": torch.from_numpy(
            np.array(consensus_result["consensus_run_counts"])).long(),
        "stability_curve": consensus_result["stability_curve"],
        "m_min": consensus_result["m_min"],
        "n_clusters": cluster_result["n_clusters"],
        # Grounding
        "concept_labels": grounding_result["labels"],
        "mono_scores": torch.from_numpy(grounding_result["mono_scores"]).float(),
        "best_nmi": torch.from_numpy(grounding_result["best_nmi"]).float(),
        # CKA
        "mean_cka": cka_result["mean_cka"] if cka_result else None,
        "cka_matrix": cka_result["cka_matrix"] if cka_result else None,
        # Best run mapping
        "best_run_dir": best_run_result["best_run_dir"],
        "best_run_id": best_run_result["best_run_id"],
        "best_run_acc": best_run_result["best_run_acc"],
        "concept_mapping": torch.from_numpy(best_run_result["concept_mapping"]).long(),
        "mapping_similarities": torch.from_numpy(
            best_run_result["mapping_similarities"]).float(),
        # Meta
        "n_runs": col_data["n_runs"],
        "hidden_dim": col_data["hidden_dim"],
        "input_dim": col_data["input_dim"],
    }

    # NMI matrix if available
    if grounding_result["nmi_matrix"] is not None:
        stage3_data["nmi_matrix"] = torch.from_numpy(grounding_result["nmi_matrix"]).float()

    save_path = os.path.join(save_dir, "stage3_outputs.pt")
    torch.save(stage3_data, save_path)
    print(f"\n  Stage 3 outputs saved: {save_path}")

    # Human-readable summary
    K = len(consensus_result["consensus_indices"])
    summary = {
        "n_runs": int(col_data["n_runs"]),
        "n_clusters": int(cluster_result["n_clusters"]),
        "n_consensus_concepts": int(K),
        "m_min": int(consensus_result["m_min"]),
        "mean_cka": float(cka_result["mean_cka"]) if cka_result else None,
        "mean_mono_score": float(grounding_result["mono_scores"].mean()) if K > 0 else None,
        "n_monosemantic": int((grounding_result["mono_scores"] > 2.0).sum()) if K > 0 else 0,
        "best_run_id": int(best_run_result["best_run_id"]),
        "best_run_acc": float(best_run_result["best_run_acc"]),
        "stability_curve": {int(k): int(v) for k, v in consensus_result["stability_curve"].items()},
        "concept_summary": [
            {
                "id": int(i),
                "label": str(grounding_result["labels"][i]),
                "mono_score": float(grounding_result["mono_scores"][i]),
                "best_nmi": float(grounding_result["best_nmi"][i]),
                "run_count": int(consensus_result["consensus_run_counts"][i])
                    if i < len(consensus_result["consensus_run_counts"]) else 0,
                "mapping_col": int(best_run_result["concept_mapping"][i]),
                "mapping_sim": float(best_run_result["mapping_similarities"][i]),
            }
            for i in range(K)
        ],
    }

    summary_path = os.path.join(save_dir, "stage3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Stage 3 summary saved: {summary_path}")

    return save_path


# ============================================================================
# Main pipeline
# ============================================================================

def run_stage3(
    stage2_dir: str,
    features: torch.Tensor,
    actions: torch.Tensor,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    save_dir: str = "./stage3_outputs",
    m_min_frac: float = 0.8,
    distance_threshold: float = 0.3,
    min_active_rate: float = 0.005,
    # Optional env grounding
    model_path: Optional[str] = None,
    env_name: Optional[str] = None,
    grounding_episodes: int = 200,
    seed: int = 42,
) -> dict:
    """
    Run the full Stage 3 pipeline.

    Returns:
        stage3_data dict (same as saved .pt file)
    """
    print(f"\n{'#'*60}")
    print(f"  STAGE 3: CONSENSUS CONCEPT EXTRACTION")
    print(f"{'#'*60}")

    # 3.1 Collect decoder columns
    col_data = collect_decoder_columns(stage2_dir)

    # 3.2 Cluster
    cluster_result = cluster_decoder_columns(
        col_data["columns"],
        col_data["active_rates"],
        min_active_rate=min_active_rate,
        distance_threshold=distance_threshold,
    )

    # 3.3 Consensus
    consensus_result = identify_consensus(
        cluster_result,
        col_data["run_ids"],
        col_data["n_runs"],
        m_min_frac=m_min_frac,
    )

    # 3.4 CKA stability
    cka_result = None
    if col_data["n_runs"] > 1:
        cka_result = compute_cka(
            features, stage2_dir, feat_mean, feat_std
        )

    # 3.5 Grounding
    if model_path and env_name:
        gt_data = extract_ground_truth_properties(
            model_path, env_name, n_episodes=grounding_episodes, seed=seed
        )
        grounding_result = compute_nmi_grounding(
            consensus_result["consensus_centroids"],
            gt_data["features"],
            gt_data["properties"],
            gt_data["property_names"],
            feat_mean, feat_std,
        )
        grounding_result["property_names"] = gt_data["property_names"]
    else:
        print("\n  No environment specified — skipping NMI grounding")
        grounding_result = compute_nmi_grounding_synthetic(
            consensus_result["consensus_centroids"],
            features, feat_mean, feat_std,
        )

    # 3.6 Best run selection & mapping
    best_run_result = select_best_run(
        stage2_dir,
        consensus_result["consensus_centroids"],
        grounding_result["labels"],
    )

    # 3.7 Plots
    plot_stage3_diagnostics(
        cluster_result, consensus_result, cka_result,
        grounding_result, save_dir,
    )

    # 3.8 Save
    save_stage3_outputs(
        cluster_result, consensus_result, cka_result,
        grounding_result, best_run_result, col_data,
        save_dir,
    )

    print(f"\n{'='*60}")
    print(f"STAGE 3 COMPLETE")
    print(f"{'='*60}")
    K = len(consensus_result["consensus_indices"])
    print(f"  Consensus concepts: {K}")
    if cka_result:
        print(f"  Mean CKA: {cka_result['mean_cka']:.4f}")
    print(f"  Best run: {best_run_result['best_run_id']} "
          f"(acc={best_run_result['best_run_acc']:.2f}%)")

    return torch.load(os.path.join(save_dir, "stage3_outputs.pt"),
                      map_location="cpu", weights_only=False)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Consensus Concept Extraction")

    parser.add_argument("--stage2_dir", type=str, required=True,
                        help="Directory containing Stage 2 run_* subdirectories")
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to collected_data.pt (features + actions)")
    parser.add_argument("--stage1_path", type=str, required=True,
                        help="Path to stage1_outputs.pt (for feat_mean/std)")

    # Clustering
    parser.add_argument("--distance_threshold", type=float, default=0.3,
                        help="Cosine distance threshold for clustering")
    parser.add_argument("--min_active_rate", type=float, default=0.005,
                        help="Minimum activation rate to include column")
    parser.add_argument("--m_min_frac", type=float, default=0.8,
                        help="Fraction of runs required for consensus")

    # Environment grounding (optional)
    parser.add_argument("--model_path", type=str, default=None,
                        help="PPO model path (for ground-truth property extraction)")
    parser.add_argument("--env_name", type=str, default=None,
                        help="Environment name (for ground-truth property extraction)")
    parser.add_argument("--grounding_episodes", type=int, default=200)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./stage3_outputs")

    args = parser.parse_args()

    # Load data
    print(f"Loading features from {args.features_path}...")
    raw = torch.load(args.features_path, map_location="cpu", weights_only=False)
    features = raw["features"]
    actions = raw["actions"]

    print(f"Loading Stage 1 outputs from {args.stage1_path}...")
    stage1_data = torch.load(args.stage1_path, map_location="cpu", weights_only=False)
    feat_mean = stage1_data["feature_mean"]
    feat_std = stage1_data["feature_std"]

    # Run
    run_stage3(
        stage2_dir=args.stage2_dir,
        features=features,
        actions=actions,
        feat_mean=feat_mean,
        feat_std=feat_std,
        save_dir=args.save_dir,
        m_min_frac=args.m_min_frac,
        distance_threshold=args.distance_threshold,
        min_active_rate=args.min_active_rate,
        model_path=args.model_path,
        env_name=args.env_name,
        grounding_episodes=args.grounding_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()