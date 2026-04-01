"""
Visualize and Analyze Learned Logic Rules (V2)
===============================================
Corrected for SAELogicAgentV2:
  - Uses _get_selection_probs() not raw clause_weights
  - Adds clause diversity / duplicate detection
  - Adds per-action class-balance audit
  - Adds feature co-occurrence heatmap
  - Fixes model import

Usage:
    python visualize_logic_rules_v2.py \
        --model_path ./sae_logic_v2_outputs/sae_logic_v2_model.pt \
        --features_path ./stage1_outputs/collected_data.pt \
        --stage1_path ./stage1_outputs/stage1_outputs.pt \
        --save_dir ./rule_visualizations_v2
"""

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset

from train_sae_logic import SAELogicAgentV2, SAELogicConfig

ACTION_NAMES = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]


# ============================================================================
# Data loading
# ============================================================================

def load_model_and_data(model_path: str, features_path: str, stage1_path: Optional[str], device: str):
    """Load V2 model and dataset, applying correct normalization."""
    print("Loading model...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    config = SAELogicConfig(**ckpt['config'])
    model = SAELogicAgentV2(config, device=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Val accuracy at save: {ckpt['best_val_acc']:.3f}")

    print("Loading features...")
    data = torch.load(features_path, weights_only=False)
    features = data['features']
    actions  = data['actions']

    # Apply same normalization used at training time
    feat_mean = ckpt.get('feature_mean', features.mean(0))
    feat_std  = ckpt.get('feature_std',  features.std(0).clamp(min=1e-6))
    features  = (features - feat_mean) / feat_std

    print(f"  Samples: {len(features)}, dim: {features.shape[1]}")
    return model, features, actions, config


# ============================================================================
# 1. Class balance audit  (prerequisite for interpreting everything else)
# ============================================================================

def audit_class_balance(actions: torch.Tensor, save_dir: str) -> Dict:
    """
    Show action frequency distribution.
    Rare classes (<5%) are the root cause of missing rules.
    """
    n_actions = len(ACTION_NAMES)
    counts = torch.bincount(actions, minlength=n_actions).numpy()
    freqs  = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#e05c5c" if f < 0.05 else "#5b8dd9" for f in freqs]
    bars = ax.bar(ACTION_NAMES, freqs * 100, color=colors, edgecolor="white", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(c), ha='center', va='bottom', fontsize=9)
    ax.axhline(5, color='#e05c5c', linestyle='--', linewidth=0.8, label='5% threshold')
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Action class distribution\n(red bars = likely to get no rules learned)")
    ax.legend()
    ax.set_ylim(0, max(freqs)*100 * 1.2)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_balance.png"), dpi=150)
    plt.close()

    print("\n=== CLASS BALANCE AUDIT ===")
    for name, cnt, freq in zip(ACTION_NAMES, counts, freqs):
        flag = "  <-- RARE, likely no rules" if freq < 0.05 else ""
        print(f"  {name:10s}: {cnt:6d} ({freq*100:5.1f}%){flag}")

    # Recommended class weights for loss rebalancing
    inv_freq = 1.0 / (freqs + 1e-8)
    weights  = inv_freq / inv_freq.sum() * n_actions  # normalize so mean=1
    print("\n  Recommended class weights for weighted cross-entropy:")
    for name, w in zip(ACTION_NAMES, weights):
        print(f"    {name:10s}: {w:.3f}")

    return {"counts": counts.tolist(), "freqs": freqs.tolist(), "recommended_weights": weights.tolist()}


# ============================================================================
# 2. Clause selection probabilities  (the correct quantity for V2)
# ============================================================================

def get_selection_probs(model: SAELogicAgentV2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns p, n each of shape (total_clauses, n_features).
    _get_selection_probs() returns (total_clauses, n_features) — rows are clauses.
    All downstream functions index accordingly: p[clause_idx, feature_idx].
    """
    with torch.no_grad():
        p, n = model.logic_layer._get_selection_probs()
    p, n = p.cpu().numpy(), n.cpu().numpy()
    # Defensive: always ensure shape is (total_clauses, n_features)
    total_clauses = model.config.n_actions * model.config.n_clauses_per_action
    n_features    = model.config.hidden_dim
    if p.shape == (n_features, total_clauses):
        p, n = p.T, n.T   # transpose if returned the other way
    assert p.shape == (total_clauses, n_features), \
        f"Unexpected shape after normalisation: {p.shape}, expected ({total_clauses}, {n_features})"
    return p, n


# ============================================================================
# 3. Clause diversity / duplicate detection
# ============================================================================

def analyze_clause_diversity(model: SAELogicAgentV2, save_dir: str) -> Dict:
    """
    For each action, compute pairwise cosine similarity between its clause
    selection-probability vectors.  Near-1.0 = duplicate clause.
    """
    p, n = get_selection_probs(model)
    # p, n shape: (total_clauses, n_features)
    usage = p + n  # (total_clauses, n_features)

    n_actions = model.config.n_actions
    nc        = model.config.n_clauses_per_action
    diversity_stats = {}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for a in range(n_actions):
        start = a * nc
        U = usage[start:start+nc, :]  # (nc, n_features)

        # Cosine similarity matrix
        norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
        U_norm = U / norms
        sim = U_norm @ U_norm.T  # (nc, nc)

        # Off-diagonal mean = average duplicate-ness
        mask = ~np.eye(nc, dtype=bool)
        mean_sim = sim[mask].mean() if mask.any() else 0.0
        max_sim  = sim[mask].max()  if mask.any() else 0.0
        n_duplicates = int((sim[mask] > 0.95).sum() // 2)

        diversity_stats[ACTION_NAMES[a]] = {
            "mean_pairwise_similarity": float(mean_sim),
            "max_pairwise_similarity":  float(max_sim),
            "n_near_duplicate_pairs":   n_duplicates,
        }

        ax = axes[a]
        im = ax.imshow(sim, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set_title(f"{ACTION_NAMES[a]}\nmean sim={mean_sim:.2f}", fontsize=10)
        ax.set_xlabel("Clause"); ax.set_ylabel("Clause")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_actions, 8):
        axes[idx].axis('off')

    plt.suptitle("Clause pairwise cosine similarity per action\n"
                 "(bright off-diagonal = duplicate clauses → L0 penalty too weak)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "clause_diversity.png"), dpi=150)
    plt.close()

    print("\n=== CLAUSE DIVERSITY ===")
    for name, s in diversity_stats.items():
        dup_flag = "  <-- DUPLICATES DETECTED" if s['n_near_duplicate_pairs'] > 0 else ""
        print(f"  {name:10s}: mean_sim={s['mean_pairwise_similarity']:.3f}  "
              f"max_sim={s['max_pairwise_similarity']:.3f}  "
              f"dup_pairs={s['n_near_duplicate_pairs']}{dup_flag}")

    return diversity_stats


# ============================================================================
# 4. Per-action selection probability heatmaps  (correct V2 version)
# ============================================================================

def visualize_selection_probs(model: SAELogicAgentV2, save_dir: str):
    """
    Heatmap of p (positive selection) and n (negative selection) per action.
    This is what the V1 code tried to do with raw weights, but incorrectly.
    """
    p, n = get_selection_probs(model)   # (n_features, total_clauses)
    nc = model.config.n_clauses_per_action
    n_actions = model.config.n_actions

    fig, axes = plt.subplots(n_actions, 2, figsize=(16, 2.5 * n_actions))

    for a in range(n_actions):
        start = a * nc
        p_a = p[start:start+nc, :]   # (nc, n_features)
        n_a = n[start:start+nc, :]

        for col, (mat, label, cmap) in enumerate([(p_a, "P(positive)", "Blues"),
                                                   (n_a, "P(negative)", "Reds")]):
            ax = axes[a, col]
            im = ax.imshow(mat, vmin=0, vmax=0.8, cmap=cmap, aspect='auto')
            ax.set_ylabel(f"{ACTION_NAMES[a]}\nClause", fontsize=9)
            if a == 0:
                ax.set_title(label, fontsize=11)
            if a == n_actions - 1:
                ax.set_xlabel("Feature index")
            plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)

    plt.suptitle("Selection probabilities per action and clause\n"
                 "(threshold >0.3 = literal appears in extracted rule)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "selection_probs.png"), dpi=150)
    plt.close()
    print("  Saved: selection_probs.png")


# ============================================================================
# 5. Feature co-occurrence across actions
# ============================================================================

def analyze_feature_sharing(model: SAELogicAgentV2, threshold: float = 0.3, save_dir: str = ".") -> Dict:
    """
    Which features appear in rules for multiple actions?
    Shared features are candidates for semantic interpretation.
    """
    p, n = get_selection_probs(model)
    nc = model.config.n_clauses_per_action
    n_actions = model.config.n_actions
    n_features = model.config.hidden_dim

    # Binary: does feature f appear in any clause for action a?
    action_feature_use = np.zeros((n_actions, n_features), dtype=bool)
    for a in range(n_actions):
        start = a * nc
        p_a = p[start:start+nc, :]   # (nc, n_features)
        n_a = n[start:start+nc, :]   # (nc, n_features)
        action_feature_use[a] = ((p_a > threshold) | (n_a > threshold)).any(axis=0)

    # How many actions use each feature?
    use_count = action_feature_use.sum(axis=0)  # (n_features,)

    # Feature-action co-occurrence matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: usage count histogram
    ax = axes[0]
    bins = np.arange(0, n_actions + 2) - 0.5
    ax.hist(use_count, bins=bins, color='#5b8dd9', edgecolor='white')
    ax.set_xlabel("Number of actions feature appears in")
    ax.set_ylabel("Feature count")
    ax.set_title("Feature sharing across actions\n(>1 = shared, semantically interesting)")
    ax.set_xticks(range(n_actions + 1))

    # Right: action × feature heatmap (only features used by ≥1 action)
    ax = axes[1]
    used_features = np.where(use_count >= 1)[0]
    if len(used_features) > 0:
        mat = action_feature_use[:, used_features].astype(float)
        im = ax.imshow(mat, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_yticks(range(n_actions))
        ax.set_yticklabels(ACTION_NAMES, fontsize=9)
        ax.set_xlabel(f"Active features (n={len(used_features)})")
        ax.set_title("Which actions use which features\n(blue = feature appears in ≥1 clause for that action)")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_sharing.png"), dpi=150)
    plt.close()

    # Find the most shared features
    shared_features = np.where(use_count >= 2)[0]
    print(f"\n=== FEATURE SHARING (threshold={threshold}) ===")
    print(f"  Features used by ≥1 action : {(use_count >= 1).sum()}")
    print(f"  Features used by ≥2 actions: {(use_count >= 2).sum()}  (shared — check these first)")
    print(f"  Features used by ≥3 actions: {(use_count >= 3).sum()}")

    if len(shared_features) > 0:
        print(f"\n  Top shared features (by action count):")
        sorted_shared = shared_features[np.argsort(use_count[shared_features])[::-1]]
        for f in sorted_shared[:20]:
            actions_using = [ACTION_NAMES[a] for a in range(n_actions) if action_feature_use[a, f]]
            sign_per_action = []
            for a in range(n_actions):
                start = a * nc
                p_f = p[start:start+nc, f].max()
                n_f = n[start:start+nc, f].max()
                if p_f > threshold:
                    sign_per_action.append(f"+{ACTION_NAMES[a]}")
                elif n_f > threshold:
                    sign_per_action.append(f"¬{ACTION_NAMES[a]}")
            print(f"    f_{f:3d}: {', '.join(sign_per_action)}")

    return {"use_count": use_count.tolist(), "shared_features": shared_features.tolist()}


# ============================================================================
# 6. Per-action fidelity with confusion breakdown
# ============================================================================

def evaluate_fidelity(
    model: SAELogicAgentV2,
    features: torch.Tensor,
    actions: torch.Tensor,
    device: str,
    save_dir: str,
) -> Dict:
    """Full fidelity evaluation including confusion matrix."""
    model.eval()
    loader = DataLoader(TensorDataset(features, actions), batch_size=512, shuffle=False)

    n_actions = model.config.n_actions
    confusion  = np.zeros((n_actions, n_actions), dtype=int)
    all_preds, all_true = [], []

    with torch.no_grad():
        for bx, ba in loader:
            bx, ba = bx.to(device), ba.to(device)
            preds = model(bx).argmax(1)
            for t, p in zip(ba.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(ba.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    overall_acc = (all_preds == all_true).mean()

    # Per-action recall
    per_action_recall = {}
    for a in range(n_actions):
        total_a = confusion[a].sum()
        per_action_recall[ACTION_NAMES[a]] = float(confusion[a, a] / total_a) if total_a > 0 else 0.0

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 7))
    # Normalize by row (true label) for recall-style confusion
    row_sums = confusion.sum(axis=1, keepdims=True).clip(min=1)
    conf_norm = confusion / row_sums

    im = ax.imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n_actions)); ax.set_xticklabels(ACTION_NAMES, rotation=35, ha='right')
    ax.set_yticks(range(n_actions)); ax.set_yticklabels(ACTION_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalized recall)\nOverall acc: {overall_acc:.3f}")
    plt.colorbar(im, ax=ax)

    for i in range(n_actions):
        for j in range(n_actions):
            ax.text(j, i, f"{conf_norm[i,j]:.2f}", ha='center', va='center',
                    fontsize=8, color='white' if conf_norm[i,j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    print(f"\n=== FIDELITY EVALUATION ===")
    print(f"  Overall accuracy: {overall_acc:.3f}")
    print(f"\n  Per-action recall:")
    for name, acc in per_action_recall.items():
        flag = "  <-- model never predicts this" if acc < 0.01 else (
               "  <-- poor" if acc < 0.5 else "")
        print(f"    {name:10s}: {acc:.3f}{flag}")

    return {
        "overall_accuracy": float(overall_acc),
        "per_action_recall": per_action_recall,
        "confusion_matrix": confusion.tolist(),
    }


# ============================================================================
# 7. Bottleneck binarization quality
# ============================================================================

def analyze_binarization(
    model: SAELogicAgentV2,
    features: torch.Tensor,
    device: str,
    save_dir: str,
):
    """Distribution of bottleneck values — should be bimodal near {0, 1}."""
    model.eval()
    loader = DataLoader(TensorDataset(features), batch_size=512, shuffle=False)
    all_z = []

    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            z_sparse, _ = model.sae.encode(bx)
            z_bin = model.bottleneck(z_sparse)
            all_z.append(z_bin.cpu())

    all_z = torch.cat(all_z).numpy().flatten()

    near_0   = (all_z < 0.05).mean()
    near_1   = (all_z > 0.95).mean()
    ambig    = ((all_z > 0.4) & (all_z < 0.6)).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(all_z, bins=80, color='#5b8dd9', edgecolor='none', alpha=0.8)
    ax.axvline(0.05, color='#2ecc71', linestyle='--', label=f'near-0: {near_0:.1%}')
    ax.axvline(0.95, color='#e74c3c', linestyle='--', label=f'near-1: {near_1:.1%}')
    ax.set_xlabel("Bottleneck activation value")
    ax.set_ylabel("Count")
    ax.set_title("Bottleneck value distribution\n(ideal: spike at 0 and spike at 1)")
    ax.legend()

    ax = axes[1]
    # Per-feature mean activation (sorted)
    all_z_mat = torch.cat([b for (b,) in DataLoader(
        TensorDataset(torch.cat([
            model.bottleneck(model.sae.encode(bx.to(device))[0]).cpu()
            for (bx,) in DataLoader(TensorDataset(features), batch_size=512)
        ])), batch_size=99999
    )]).numpy()
    per_feature_mean = all_z_mat.mean(axis=0)
    sorted_means = np.sort(per_feature_mean)[::-1]
    ax.bar(range(len(sorted_means)), sorted_means, color='#5b8dd9', width=1.0)
    ax.set_xlabel("Feature index (sorted by mean activation)")
    ax.set_ylabel("Mean activation")
    ax.set_title("Per-feature mean activation\n(dead features near 0, always-on near 1)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "binarization_quality.png"), dpi=150)
    plt.close()

    print(f"\n=== BINARIZATION QUALITY ===")
    print(f"  Near 0 (<0.05):        {near_0:.3f}")
    print(f"  Near 1 (>0.95):        {near_1:.3f}")
    print(f"  Near-binary total:     {near_0+near_1:.3f}")
    print(f"  Ambiguous (0.4–0.6):   {ambig:.3f}  (should be <0.05)")
    if ambig > 0.05:
        print("  WARNING: High ambiguity. Increase bimodal_max or extend bimodal_ramp.")


# ============================================================================
# 8. Summary report
# ============================================================================

def print_summary_and_recommendations(
    class_stats: Dict,
    diversity_stats: Dict,
    fidelity_stats: Dict,
):
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    issues = []
    fixes  = []

    # Check rare classes
    freqs = np.array(class_stats['freqs'])
    rare  = [ACTION_NAMES[i] for i, f in enumerate(freqs) if f < 0.05]
    if rare:
        issues.append(f"Rare actions with no rules: {rare}")
        weights = class_stats['recommended_weights']
        w_str = ', '.join(f'{ACTION_NAMES[i]}={weights[i]:.2f}'
                          for i in range(len(ACTION_NAMES)))
        fixes.append(f"Add class weights to cross_entropy: [{w_str}]")

    # Check duplicates
    dup_actions = [name for name, s in diversity_stats.items()
                   if s['n_near_duplicate_pairs'] > 0]
    if dup_actions:
        issues.append(f"Duplicate clauses in: {dup_actions}")
        fixes.append("Increase l0_penalty_weight (try 10× current value)")
        fixes.append("Add diversity loss: penalize pairwise cosine sim between clauses of same action")
        fixes.append("Reduce n_clauses_per_action from 10 to 4–5")

    # Check per-action recall
    zero_recall = [name for name, r in fidelity_stats['per_action_recall'].items()
                   if r < 0.01]
    if zero_recall:
        issues.append(f"Zero recall (never predicted): {zero_recall}")
        fixes.append("Class weighting is the primary fix for zero-recall actions")

    # Check overall accuracy
    acc = fidelity_stats['overall_accuracy']
    if acc > 0.90:
        issues.append(f"High overall accuracy ({acc:.3f}) masks per-action failures")
        fixes.append("Do NOT optimize for overall accuracy alone — use macro-averaged recall")

    print("\nIssues found:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print("\nRecommended fixes (in priority order):")
    for i, fix in enumerate(fixes, 1):
        print(f"  {i}. {fix}")

    if not issues:
        print("  No major issues detected.")


# ============================================================================
# Main
# ============================================================================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    model, features, actions, config = load_model_and_data(
        args.model_path, args.features_path, args.stage1_path, device
    )

    print("\n" + "=" * 70)
    print("Running analysis pipeline...")
    print("=" * 70)

    # 1. Class balance
    class_stats = audit_class_balance(actions, args.save_dir)

    # 2. Clause diversity / duplicate detection
    diversity_stats = analyze_clause_diversity(model, args.save_dir)

    # 3. Selection probability heatmaps (correct V2 version)
    print("\nGenerating selection probability heatmaps...")
    visualize_selection_probs(model, args.save_dir)

    # 4. Feature sharing
    sharing_stats = analyze_feature_sharing(model, threshold=0.3, save_dir=args.save_dir)

    # 5. Fidelity + confusion matrix
    fidelity_stats = evaluate_fidelity(model, features, actions, device, args.save_dir)

    # 6. Binarization quality
    analyze_binarization(model, features, device, args.save_dir)

    # 7. Print extracted rules (from checkpoint if available)
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    if 'rules' in ckpt:
        rules = ckpt['rules']
    else:
        rules = model.extract_rules(action_names=ACTION_NAMES)

    print("\n" + "=" * 70)
    print("EXTRACTED RULES")
    print("=" * 70)
    for action, clauses in rules.items():
        print(f"\n{action} ←")
        for clause in clauses:
            print(f"    {clause}")

    # 8. Summary and recommendations
    print_summary_and_recommendations(class_stats, diversity_stats, fidelity_stats)

    # Save all stats
    all_stats = {
        "class_balance": class_stats,
        "clause_diversity": diversity_stats,
        "feature_sharing": {k: v for k, v in sharing_stats.items() if k != 'use_count'},
        "fidelity": {k: v for k, v in fidelity_stats.items() if k != 'confusion_matrix'},
    }
    with open(os.path.join(args.save_dir, "analysis_summary.json"), 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n  All outputs saved to: {args.save_dir}/")
    print("  Files: class_balance.png, clause_diversity.png, selection_probs.png,")
    print("         feature_sharing.png, confusion_matrix.png, binarization_quality.png,")
    print("         analysis_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    type=str, required=True)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--stage1_path",   type=str, default=None)
    parser.add_argument("--save_dir",      type=str, default="./rule_visualizations_v2")
    args = parser.parse_args()
    main(args)