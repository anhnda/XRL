"""
Visualize and Analyze Learned Logic Rules
==========================================
Analyzes the neural logic network to:
  - Visualize rule structure
  - Compute rule statistics
  - Test rule fidelity
  - Compare with original policy

Usage:
    python visualize_logic_rules.py \
        --model_path ./sae_logic_outputs/sae_logic_model.pt \
        --features_path ./stage1_outputs/collected_data.pt \
        --save_dir ./rule_visualizations
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

from train_sae_logic import SAELogicAgent, SAELogicConfig


def visualize_clause_structure(model: SAELogicAgent, save_path: str):
    """Visualize the structure of learned clauses"""
    model.eval()

    # Get clause weights
    clause_weights = model.logic_layer.get_clause_weights().detach().cpu().numpy()
    n_features, n_clauses = clause_weights.shape

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 8))

    # Transpose so clauses are rows and features are columns
    sns.heatmap(
        clause_weights.T,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Weight'},
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )

    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Clauses', fontsize=12)
    ax.set_title('Neural Logic Network Structure\n(Red=Negative, Blue=Positive, White=Unused)',
                 fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✓ Saved clause structure to: {save_path}")


def visualize_action_rules(model: SAELogicAgent, action_names: List[str], save_path: str):
    """Visualize rules for each action"""
    model.eval()

    clause_weights = model.logic_layer.get_clause_weights().detach().cpu().numpy()
    n_features, n_clauses = clause_weights.shape
    n_actions = model.config.n_actions
    n_clauses_per_action = model.config.n_clauses_per_action

    # Create subplot for each action
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for action_idx in range(min(n_actions, 8)):
        ax = axes[action_idx]

        # Get clauses for this action
        start_clause = action_idx * n_clauses_per_action
        end_clause = start_clause + n_clauses_per_action
        action_clauses = clause_weights[:, start_clause:end_clause]

        # Plot heatmap
        sns.heatmap(
            action_clauses.T,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar=True,
            xticklabels=False,
            yticklabels=range(1, n_clauses_per_action + 1),
            ax=ax
        )

        action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action {action_idx}"
        ax.set_title(f'{action_name}', fontsize=12)
        ax.set_xlabel('Features')
        ax.set_ylabel('Clause')

    # Hide unused subplots
    for idx in range(n_actions, 8):
        axes[idx].axis('off')

    plt.suptitle('Logic Rules per Action\n(Each row is a clause: OR of these clauses → action)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✓ Saved action rules to: {save_path}")


def compute_rule_statistics(model: SAELogicAgent, action_names: List[str]) -> Dict:
    """Compute detailed statistics about learned rules"""
    model.eval()

    clause_weights = model.logic_layer.get_clause_weights().detach().cpu().numpy()
    n_features, n_clauses = clause_weights.shape

    stats = {
        'overall': {},
        'per_action': {}
    }

    # Overall statistics
    active_literals = np.abs(clause_weights) > 0.1
    stats['overall']['total_clauses'] = n_clauses
    stats['overall']['non_empty_clauses'] = (active_literals.sum(axis=0) > 0).sum()
    stats['overall']['total_literals'] = active_literals.sum()
    stats['overall']['avg_literals_per_clause'] = active_literals.sum() / max((active_literals.sum(axis=0) > 0).sum(), 1)
    stats['overall']['sparsity'] = 1 - (active_literals.sum() / (n_features * n_clauses))

    # Count positive vs negative literals
    pos_literals = (clause_weights > 0.1).sum()
    neg_literals = (clause_weights < -0.1).sum()
    stats['overall']['positive_literals'] = int(pos_literals)
    stats['overall']['negative_literals'] = int(neg_literals)
    stats['overall']['pos_neg_ratio'] = pos_literals / max(neg_literals, 1)

    # Per-action statistics
    n_clauses_per_action = model.config.n_clauses_per_action
    n_actions = model.config.n_actions

    for action_idx in range(n_actions):
        action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action_{action_idx}"
        start_clause = action_idx * n_clauses_per_action
        end_clause = start_clause + n_clauses_per_action

        action_clauses = clause_weights[:, start_clause:end_clause]
        action_active = np.abs(action_clauses) > 0.1

        stats['per_action'][action_name] = {
            'n_clauses': n_clauses_per_action,
            'non_empty_clauses': int((action_active.sum(axis=0) > 0).sum()),
            'total_literals': int(action_active.sum()),
            'avg_literals_per_clause': float(action_active.sum() / max((action_active.sum(axis=0) > 0).sum(), 1)),
            'positive_literals': int((action_clauses > 0.1).sum()),
            'negative_literals': int((action_clauses < -0.1).sum())
        }

    return stats


def evaluate_rule_fidelity(
    model: SAELogicAgent,
    data_loader: DataLoader,
    device: str
) -> Dict:
    """
    Evaluate how well the logic rules match the learned SAE features.

    Fidelity = accuracy of logic rules using SAE features
    """
    model.eval()

    total = 0
    correct = 0
    per_action_correct = {i: 0 for i in range(model.config.n_actions)}
    per_action_total = {i: 0 for i in range(model.config.n_actions)}

    with torch.no_grad():
        for batch_x, batch_a in data_loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)

            # Get predictions
            logits = model(batch_x)
            preds = logits.argmax(1)

            # Overall accuracy
            correct += (preds == batch_a).sum().item()
            total += batch_a.size(0)

            # Per-action accuracy
            for action_idx in range(model.config.n_actions):
                mask = (batch_a == action_idx)
                if mask.sum() > 0:
                    per_action_correct[action_idx] += (preds[mask] == action_idx).sum().item()
                    per_action_total[action_idx] += mask.sum().item()

    fidelity = {
        'overall_accuracy': correct / total,
        'per_action_accuracy': {}
    }

    for action_idx in range(model.config.n_actions):
        if per_action_total[action_idx] > 0:
            fidelity['per_action_accuracy'][action_idx] = \
                per_action_correct[action_idx] / per_action_total[action_idx]
        else:
            fidelity['per_action_accuracy'][action_idx] = 0.0

    return fidelity


def print_readable_rules(rules: Dict[str, List[str]]):
    """Print rules in a readable format"""
    print("\n" + "=" * 80)
    print("LEARNED LOGIC RULES (DNF FORM)")
    print("=" * 80)

    for action_name, clauses in rules.items():
        print(f"\n{action_name}:")
        if clauses == ["⊥"]:
            print("  (never taken)")
            continue

        for i, clause in enumerate(clauses, 1):
            print(f"  {i}. {clause}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)

    config_dict = checkpoint['config']
    config = SAELogicConfig(**config_dict)

    model = SAELogicAgent(config, device=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"✓ Loaded model (Val Acc: {checkpoint['best_val_acc']:.3f})")

    # Load data for fidelity evaluation
    print("\nLoading data...")
    data = torch.load(args.features_path)
    features = data['features']
    actions = data['actions']

    dataset = TensorDataset(features, actions)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    # Action names
    action_names = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]

    # 1. Print rules
    if 'rules' in checkpoint:
        print_readable_rules(checkpoint['rules'])
    else:
        rules = model.extract_rules(action_names=action_names)
        print_readable_rules(rules)

    # 2. Compute statistics
    print("\n" + "=" * 80)
    print("RULE STATISTICS")
    print("=" * 80)

    stats = compute_rule_statistics(model, action_names)

    print("\nOverall:")
    for key, value in stats['overall'].items():
        print(f"  {key}: {value}")

    print("\nPer Action:")
    for action_name, action_stats in stats['per_action'].items():
        print(f"\n  {action_name}:")
        for key, value in action_stats.items():
            print(f"    {key}: {value}")

    # Save statistics
    stats_path = os.path.join(args.save_dir, "rule_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved statistics to: {stats_path}")

    # 3. Evaluate fidelity
    print("\n" + "=" * 80)
    print("RULE FIDELITY EVALUATION")
    print("=" * 80)

    fidelity = evaluate_rule_fidelity(model, data_loader, device)
    print(f"\nOverall Accuracy: {fidelity['overall_accuracy']:.3f}")
    print("\nPer-Action Accuracy:")
    for action_idx, acc in fidelity['per_action_accuracy'].items():
        action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action_{action_idx}"
        print(f"  {action_name}: {acc:.3f}")

    # Save fidelity
    fidelity_path = os.path.join(args.save_dir, "rule_fidelity.json")
    with open(fidelity_path, 'w') as f:
        json.dump(fidelity, f, indent=2)
    print(f"\n✓ Saved fidelity to: {fidelity_path}")

    # 4. Visualize clause structure
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    structure_path = os.path.join(args.save_dir, "clause_structure.png")
    visualize_clause_structure(model, structure_path)

    action_rules_path = os.path.join(args.save_dir, "action_rules.png")
    visualize_action_rules(model, action_names, action_rules_path)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and analyze learned logic rules")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (sae_logic_model.pt)")
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to collected_data.pt for evaluation")
    parser.add_argument("--save_dir", type=str, default="./rule_visualizations",
                        help="Directory to save visualizations")

    args = parser.parse_args()
    main(args)
