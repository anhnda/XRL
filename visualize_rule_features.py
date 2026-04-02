#!/usr/bin/env python3
"""
Visualize SAE Rule Features — Strategy 1: Maximally Activating Examples
========================================================================

For each feature that appears in the learned DNF rules:
  1. Run the frozen SAE over the full dataset
  2. Collect top-K observations that MAXIMALLY activate the feature
  3. Collect top-K observations where the feature is INACTIVE (near zero)
  4. Render them side-by-side for visual comparison
  5. Save per-feature images + a per-rule combo summary
  6. Print a suggested prompt for concept naming (e.g. send to Claude with the image)

Usage:
    python visualize_rule_features.py \
        --model_path ./sae_logic_v3_outputs/sae_logic_v3_model.pt \
        --features_path ./stage1_outputs/collected_data.pt \
        --obs_path ./stage1_outputs/collected_data.pt \
        --top_k 16 \
        --save_dir ./feature_visualizations

Notes:
    - Expects `collected_data.pt` to contain 'observations' (raw pixel or grid)
      alongside 'features' and 'actions'.
    - If observations are MiniGrid-style grids (H, W, 3) with object encoding,
      we render them with a simple color map.
    - If observations are raw images, we display them directly.
"""

import argparse
import os
import json
import sys
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ──────────────────────────────────────────────────────────────────────
# MiniGrid rendering helpers
# ──────────────────────────────────────────────────────────────────────

# MiniGrid object type IDs → names
OBJECT_NAMES = {
    0: "unseen", 1: "empty", 2: "wall", 3: "floor",
    4: "door", 5: "key", 6: "ball", 7: "box",
    8: "goal", 9: "lava", 10: "agent",
}

# MiniGrid color IDs → RGB
COLOR_MAP = {
    0: (100, 100, 100),   # red (dimmed for grid bg)
    1: (0, 200, 0),       # green
    2: (0, 0, 200),       # blue
    3: (200, 0, 200),     # purple
    4: (200, 200, 0),     # yellow
    5: (100, 100, 100),   # grey
}

# Base colors per object type (before color channel modulation)
OBJECT_BASE_COLORS = {
    0: np.array([40, 40, 40]),       # unseen — dark
    1: np.array([220, 220, 220]),    # empty — light grey
    2: np.array([100, 100, 100]),    # wall — grey
    3: np.array([180, 180, 160]),    # floor — beige
    4: np.array([150, 75, 0]),       # door — brown
    5: np.array([255, 215, 0]),      # key — gold
    6: np.array([200, 50, 50]),      # ball — red
    7: np.array([160, 82, 45]),      # box — sienna
    8: np.array([0, 200, 50]),       # goal — green
    9: np.array([255, 69, 0]),       # lava — orange-red
    10: np.array([30, 144, 255]),    # agent — dodger blue
}


def render_minigrid_obs(obs: np.ndarray, cell_size: int = 16) -> np.ndarray:
    """
    Render a MiniGrid partial observation (H, W, 3) into an RGB image.

    Channel 0 = object type, Channel 1 = color, Channel 2 = state.
    Returns an (H*cell_size, W*cell_size, 3) uint8 image.
    """
    if obs.ndim == 1:
        # Flattened — try to reshape. Common sizes: 7x7x3=147, 5x5x3=75
        n = obs.shape[0]
        for side in [7, 5, 6, 8, 11, 13]:
            if n == side * side * 3:
                obs = obs.reshape(side, side, 3)
                break
        else:
            # Can't reshape — return a placeholder
            img = np.full((cell_size * 4, cell_size * 4, 3), 128, dtype=np.uint8)
            return img

    H, W = obs.shape[0], obs.shape[1]
    img = np.zeros((H * cell_size, W * cell_size, 3), dtype=np.uint8)

    for r in range(H):
        for c in range(W):
            obj_type = int(obs[r, c, 0])
            color_id = int(obs[r, c, 1])
            state = int(obs[r, c, 2])

            base = OBJECT_BASE_COLORS.get(obj_type, np.array([128, 128, 128]))

            # Modulate by color channel for colored objects
            if obj_type in (4, 5, 6, 7) and color_id in COLOR_MAP:
                crgb = np.array(COLOR_MAP[color_id], dtype=np.float32)
                base = (base.astype(np.float32) * 0.3 + crgb * 0.7).astype(np.uint8)

            # Door state: 0=open, 1=closed, 2=locked → darken if locked
            if obj_type == 4 and state == 2:
                base = (base * 0.5).astype(np.uint8)
            elif obj_type == 4 and state == 0:
                base = np.minimum(base.astype(np.int32) + 60, 255).astype(np.uint8)

            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size
            img[y0:y1, x0:x1] = base

            # Grid lines
            img[y0, x0:x1] = np.minimum(base.astype(np.int32) - 30, 0).clip(0).astype(np.uint8)
            img[y0:y1, x0] = np.minimum(base.astype(np.int32) - 30, 0).clip(0).astype(np.uint8)

    return img


def render_image_obs(obs: np.ndarray) -> np.ndarray:
    """Handle already-image observations (H, W, 3) uint8 or float."""
    if obs.dtype in (np.float32, np.float64):
        if obs.max() <= 1.0:
            obs = (obs * 255).astype(np.uint8)
        else:
            obs = obs.astype(np.uint8)
    return obs


def smart_render(obs: np.ndarray, cell_size: int = 16) -> np.ndarray:
    """
    Detect whether obs is a MiniGrid grid encoding or a raw image, render accordingly.
    """
    if obs.ndim == 3 and obs.shape[2] == 3:
        # Check if values look like MiniGrid encoding (small integers 0-10 in channel 0)
        if obs[:, :, 0].max() <= 15 and obs.dtype in (np.int64, np.int32, np.float32, np.float64, np.uint8):
            unique_vals = np.unique(obs[:, :, 0])
            if len(unique_vals) < 16 and unique_vals.max() <= 15:
                return render_minigrid_obs(obs, cell_size)
    if obs.ndim == 3 and obs.shape[0] in (3, 1):
        # CHW → HWC
        obs = np.transpose(obs, (1, 2, 0))
    if obs.ndim == 3 and obs.shape[2] == 1:
        obs = np.repeat(obs, 3, axis=2)
    if obs.ndim == 2:
        obs = np.stack([obs] * 3, axis=-1)
    return render_image_obs(obs)


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def load_model(model_path: str, device: str = "cpu"):
    """
    Load the trained SAELogicAgentV3 model from checkpoint.
    
    We reconstruct the model from saved config and state dict.
    This avoids importing the training script directly — we rebuild
    the necessary classes inline (or import if available).
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config_dict = checkpoint['config']
    state_dict = checkpoint['model_state']
    
    # Try importing from training script
    try:
        from train_sae_logic import SAELogicAgentV3, SAELogicConfig
        config = SAELogicConfig(**{k: v for k, v in config_dict.items() 
                                   if k in SAELogicConfig.__dataclass_fields__})
        model = SAELogicAgentV3(config, device=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"  Loaded model via train_sae_logic_v3 import")
        return model, config, checkpoint
    except ImportError:
        pass

    # Fallback: reconstruct manually
    print(f"  Reconstructing model from checkpoint config...")
    print(f"  Config: input_dim={config_dict['input_dim']}, "
          f"hidden_dim={config_dict['hidden_dim']}, k={config_dict['k']}, "
          f"n_actions={config_dict['n_actions']}")
    
    # We need the SAE class — try importing
    try:
        from sparse_concept_autoencoder import OvercompleteSAE
    except ImportError:
        print("  ERROR: Cannot import OvercompleteSAE. Please ensure "
              "sparse_concept_autoencoder.py is in the Python path.")
        sys.exit(1)
    
    from train_sae_logic import SAELogicAgentV3, SAELogicConfig
    config = SAELogicConfig(**{k: v for k, v in config_dict.items() 
                               if k in SAELogicConfig.__dataclass_fields__})
    model = SAELogicAgentV3(config, device=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    model.eval()
    return model, config, checkpoint


# ──────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────

def parse_rules_for_features(rules: dict) -> dict:
    """
    Parse the saved rules dict to extract which features appear in each action's rules,
    and whether they appear as positive (f_i) or negated (¬f_i).
    
    Returns:
        {feature_index: {'actions': [...], 'polarities': {action: 'pos'|'neg'}}}
    """
    import re
    feature_info = {}
    
    for action_name, clauses in rules.items():
        for clause in clauses:
            if clause == "(no active clauses)":
                continue
            # Find all f_NNN and ¬f_NNN
            positives = re.findall(r'(?<![¬])f_(\d+)', clause)
            negatives = re.findall(r'¬f_(\d+)', clause)
            
            for fidx in positives:
                fidx = int(fidx)
                if fidx not in feature_info:
                    feature_info[fidx] = {'actions': [], 'polarities': {}}
                if action_name not in feature_info[fidx]['actions']:
                    feature_info[fidx]['actions'].append(action_name)
                feature_info[fidx]['polarities'][action_name] = 'pos'
            
            for fidx in negatives:
                fidx = int(fidx)
                if fidx not in feature_info:
                    feature_info[fidx] = {'actions': [], 'polarities': {}}
                if action_name not in feature_info[fidx]['actions']:
                    feature_info[fidx]['actions'].append(action_name)
                feature_info[fidx]['polarities'][action_name] = 'neg'
    
    return feature_info


@torch.no_grad()
def collect_feature_activations(model, features_tensor, device, batch_size=512):
    """
    Run frozen SAE over all data, return sparse activations (N, hidden_dim).
    """
    model.eval()
    all_z = []
    n = features_tensor.shape[0]
    
    for i in range(0, n, batch_size):
        batch = features_tensor[i:i+batch_size].to(device)
        # Normalize input the same way as training
        batch_norm = model.normalize_input(batch)
        z_sparse, _ = model.sae.encode(batch_norm)
        all_z.append(z_sparse.cpu())
    
    return torch.cat(all_z, dim=0)


# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────

def plot_feature_topk(
    feature_idx: int,
    top_obs: list,
    bottom_obs: list,
    top_acts: np.ndarray,
    bottom_acts: np.ndarray,
    top_actions: list,
    bottom_actions: list,
    feature_info: dict,
    action_names: list,
    save_path: str,
    k: int = 16,
):
    """
    Plot top-K activating vs top-K non-activating observations for one feature.
    
    Layout:
        Row 1: "TOP ACTIVATING" — observations where feature fires most
        Row 2: "LEAST ACTIVATING" — observations where feature is near zero
    """
    n_cols = min(k, 8)
    n_rows_per = max(1, (k + n_cols - 1) // n_cols)
    total_rows = n_rows_per * 2  # top + bottom
    
    fig = plt.figure(figsize=(n_cols * 2.2, total_rows * 2.2 + 2.5))
    
    # Title with rule context
    info = feature_info.get(feature_idx, {})
    polarity_strs = []
    for act, pol in info.get('polarities', {}).items():
        symbol = "f" if pol == 'pos' else "¬f"
        polarity_strs.append(f"{symbol}_{feature_idx} in {act}")
    context = " | ".join(polarity_strs) if polarity_strs else "not in any rule"
    
    fig.suptitle(
        f"Feature f_{feature_idx}\n{context}",
        fontsize=14, fontweight='bold', y=0.98
    )
    
    gs = GridSpec(total_rows + 1, n_cols, figure=fig, 
                  height_ratios=[0.3] + [1] * total_rows,
                  hspace=0.3, wspace=0.15)
    
    # Section label: TOP ACTIVATING
    ax_label_top = fig.add_subplot(gs[0, :n_cols//2])
    ax_label_top.text(0.5, 0.5, f"▲ TOP {k} ACTIVATING", 
                      ha='center', va='center', fontsize=12, fontweight='bold',
                      color='#2d6a2d')
    ax_label_top.axis('off')
    
    ax_label_bot = fig.add_subplot(gs[0, n_cols//2:])
    ax_label_bot.text(0.5, 0.5, f"▼ LEAST ACTIVATING (near zero)",
                      ha='center', va='center', fontsize=12, fontweight='bold',
                      color='#8b2020')
    ax_label_bot.axis('off')
    
    ACTION_COLORS = {
        "TurnLeft": "#e74c3c", "TurnRight": "#3498db", "Forward": "#2ecc71",
        "Pickup": "#f39c12", "Drop": "#9b59b6", "Toggle": "#e67e22", "Done": "#7f8c8d",
    }
    
    def plot_grid(obs_list, act_values, act_labels, row_offset, section_color):
        for i, (obs, act_val, act_label) in enumerate(zip(obs_list, act_values, act_labels)):
            r = i // n_cols
            c = i % n_cols
            ax = fig.add_subplot(gs[row_offset + r, c])
            
            img = smart_render(obs, cell_size=12)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Activation value + action label
            act_color = ACTION_COLORS.get(act_label, '#333')
            ax.set_title(f"{act_val:.2f}", fontsize=8, color=section_color)
            ax.set_xlabel(f"{act_label}", fontsize=7, color=act_color, fontweight='bold')
            
            # Border color by section
            for spine in ax.spines.values():
                spine.set_color(section_color)
                spine.set_linewidth(1.5)
    
    plot_grid(top_obs[:k], top_acts[:k], top_actions[:k], 1, '#2d6a2d')
    plot_grid(bottom_obs[:k], bottom_acts[:k], bottom_actions[:k], 1 + n_rows_per, '#8b2020')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_rule_summary(
    rule_action: str,
    clause_str: str,
    feature_indices: list,
    polarities: list,
    top_obs_per_feature: dict,
    action_names: list,
    save_path: str,
    n_examples: int = 6,
):
    """
    For one rule (action ← clause), show a summary grid:
    Each row = one feature in the clause, showing its top activating examples.
    
    This gives a visual "story" of what the rule means.
    """
    n_features = len(feature_indices)
    fig, axes = plt.subplots(
        n_features, n_examples,
        figsize=(n_examples * 2, n_features * 2.2 + 1.5),
        squeeze=False,
    )
    
    fig.suptitle(
        f"{rule_action} ←\n{clause_str}",
        fontsize=12, fontweight='bold', y=1.02, ha='center',
        fontfamily='monospace',
    )
    
    for row, (fidx, polarity) in enumerate(zip(feature_indices, polarities)):
        symbol = f"f_{fidx}" if polarity == 'pos' else f"¬f_{fidx}"
        
        # For positive features, show top activating; for negated, show low activating
        if polarity == 'pos':
            obs_list = top_obs_per_feature[fidx]['top_obs'][:n_examples]
            label_color = '#2d6a2d'
        else:
            obs_list = top_obs_per_feature[fidx]['bottom_obs'][:n_examples]
            label_color = '#8b2020'
        
        # Row label
        axes[row, 0].set_ylabel(
            symbol, fontsize=11, fontweight='bold', color=label_color,
            rotation=0, labelpad=40, va='center',
        )
        
        for col in range(n_examples):
            ax = axes[row, col]
            if col < len(obs_list):
                img = smart_render(obs_list[col], cell_size=12)
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(label_color)
                spine.set_linewidth(1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved rule summary: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Prompt generation
# ──────────────────────────────────────────────────────────────────────

def generate_naming_prompt(feature_idx, info, save_dir, game_name="MiniGrid"):
    """Generate a suggested prompt for Claude to name the concept."""
    polarity_context = []
    for act, pol in info.get('polarities', {}).items():
        polarity_context.append(f"{'present' if pol == 'pos' else 'absent'} → {act}")
    
    prompt = f"""I'm analyzing a learned policy for {game_name}. Here is an image showing 
the top activating observations (green border) vs non-activating observations (red border) 
for feature f_{feature_idx}.

Rule context: {', '.join(polarity_context)}

Based on the visual patterns, what concept does this feature represent? 
Give a short name (1-4 words) and a one-sentence explanation.

[Attach: {os.path.join(save_dir, f'feature_{feature_idx}.png')}]"""
    
    return prompt


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize SAE rule features")
    parser.add_argument("--model_path", type=str, 
                        default="./sae_logic_v3_outputs/sae_logic_v3_model.pt")
    parser.add_argument("--features_path", type=str,
                        default="./stage1_outputs/collected_data.pt",
                        help="Path to collected_data.pt with 'features', 'actions', and 'observations'")
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="./feature_visualizations")
    parser.add_argument("--game_name", type=str, default="MiniGrid",
                        help="Game name for prompt generation")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ── Load model ──
    print("=" * 70)
    print("FEATURE VISUALIZATION — Strategy 1: Maximally Activating Examples")
    print("=" * 70)
    print(f"\nLoading model from {args.model_path}...")
    model, config, checkpoint = load_model(args.model_path, device)
    
    # ── Load rules ──
    rules = checkpoint.get('rules', {})
    if not rules:
        rules_path = os.path.join(os.path.dirname(args.model_path), "learned_rules.json")
        if os.path.exists(rules_path):
            with open(rules_path) as f:
                rules = json.load(f)
    
    print(f"\nLoaded rules for {len(rules)} actions")
    feature_info = parse_rules_for_features(rules)
    rule_features = sorted(feature_info.keys())
    print(f"Features appearing in rules: {rule_features}")
    print(f"  ({len(rule_features)} unique features to visualize)")
    
    # ── Load data ──
    print(f"\nLoading data from {args.features_path}...")
    data = torch.load(args.features_path, map_location='cpu', weights_only=False)
    
    features_tensor = data['features']
    actions_tensor = data['actions']
    
    # Observations — the key data for visualization
    if 'observations' in data:
        observations = data['observations']
        if isinstance(observations, torch.Tensor):
            observations = observations.numpy()
        print(f"  Observations: shape={observations.shape}, dtype={observations.dtype}")
    elif 'obs' in data:
        observations = data['obs']
        if isinstance(observations, torch.Tensor):
            observations = observations.numpy()
        print(f"  Observations (from 'obs'): shape={observations.shape}")
    else:
        print("  WARNING: No 'observations' key found in data!")
        print(f"  Available keys: {list(data.keys())}")
        print("  Will use feature vectors as placeholder (not visual).")
        observations = None
    
    action_names = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]
    print(f"  Features: {features_tensor.shape}")
    print(f"  Actions: {actions_tensor.shape}")
    
    # ── Collect SAE activations ──
    print(f"\nRunning frozen SAE over {len(features_tensor)} samples...")
    all_z = collect_feature_activations(model, features_tensor, device)
    print(f"  Activations shape: {all_z.shape}")
    
    # ── Per-feature visualization ──
    print(f"\n{'='*70}")
    print(f"Generating per-feature visualizations (top-{args.top_k})")
    print(f"{'='*70}")
    
    top_obs_per_feature = {}
    prompts = {}
    
    for fidx in rule_features:
        print(f"\n  Feature f_{fidx}:")
        info = feature_info[fidx]
        
        # Get activation values for this feature
        z_col = all_z[:, fidx].numpy()
        
        # Top-K activating (highest activation)
        top_indices = np.argsort(z_col)[::-1][:args.top_k]
        # Bottom-K: near zero activation (sorted by ascending, only truly low)
        # Filter to samples where activation is actually near zero
        near_zero_mask = z_col < 0.01
        if near_zero_mask.sum() >= args.top_k:
            # Among near-zero, pick randomly for diversity
            zero_indices = np.where(near_zero_mask)[0]
            rng = np.random.RandomState(42 + fidx)
            bottom_indices = rng.choice(zero_indices, size=args.top_k, replace=False)
        else:
            # Just take the lowest
            bottom_indices = np.argsort(z_col)[:args.top_k]
        
        top_act_values = z_col[top_indices]
        bottom_act_values = z_col[bottom_indices]
        
        top_action_labels = [action_names[actions_tensor[i].item()] for i in top_indices]
        bottom_action_labels = [action_names[actions_tensor[i].item()] for i in bottom_indices]
        
        print(f"    Top activation range: [{top_act_values[-1]:.3f}, {top_act_values[0]:.3f}]")
        print(f"    Bottom activation range: [{bottom_act_values.min():.3f}, {bottom_act_values.max():.3f}]")
        print(f"    Actions in rules: {info['polarities']}")
        
        # Get observations
        if observations is not None:
            top_obs = [observations[i] for i in top_indices]
            bottom_obs = [observations[i] for i in bottom_indices]
        else:
            # Placeholder: render feature vectors as tiny heatmaps
            top_obs = [features_tensor[i].numpy().reshape(-1)[:49].reshape(7, 7) 
                       for i in top_indices]
            bottom_obs = [features_tensor[i].numpy().reshape(-1)[:49].reshape(7, 7)
                          for i in bottom_indices]
        
        top_obs_per_feature[fidx] = {
            'top_obs': top_obs,
            'bottom_obs': bottom_obs,
            'top_indices': top_indices,
            'bottom_indices': bottom_indices,
            'top_acts': top_act_values,
            'bottom_acts': bottom_act_values,
        }
        
        # Plot
        save_path = os.path.join(args.save_dir, f"feature_{fidx}.png")
        plot_feature_topk(
            feature_idx=fidx,
            top_obs=top_obs,
            bottom_obs=bottom_obs,
            top_acts=top_act_values,
            bottom_acts=bottom_act_values,
            top_actions=top_action_labels,
            bottom_actions=bottom_action_labels,
            feature_info=feature_info,
            action_names=action_names,
            save_path=save_path,
            k=args.top_k,
        )
        
        # Generate naming prompt
        prompt = generate_naming_prompt(fidx, info, args.save_dir, args.game_name)
        prompts[f"f_{fidx}"] = prompt
    
    # ── Per-rule summary visualization ──
    print(f"\n{'='*70}")
    print("Generating per-rule combo summaries")
    print(f"{'='*70}")
    
    import re
    for action_name, clauses in rules.items():
        for clause_idx, clause in enumerate(clauses):
            if clause == "(no active clauses)":
                continue
            
            # Parse features and polarities from this clause
            pos_features = [int(x) for x in re.findall(r'(?<![¬])f_(\d+)', clause)]
            neg_features = [int(x) for x in re.findall(r'¬f_(\d+)', clause)]
            
            all_feats = pos_features + neg_features
            all_pols = ['pos'] * len(pos_features) + ['neg'] * len(neg_features)
            
            if not all_feats:
                continue
            
            # Check all features have data
            missing = [f for f in all_feats if f not in top_obs_per_feature]
            if missing:
                print(f"  Skipping {action_name} clause {clause_idx}: "
                      f"missing features {missing}")
                continue
            
            safe_name = action_name.replace(" ", "_")
            save_path = os.path.join(
                args.save_dir, f"rule_{safe_name}_clause{clause_idx}.png"
            )
            
            plot_rule_summary(
                rule_action=action_name,
                clause_str=clause,
                feature_indices=all_feats,
                polarities=all_pols,
                top_obs_per_feature=top_obs_per_feature,
                action_names=action_names,
                save_path=save_path,
                n_examples=min(6, args.top_k),
            )
    
    # ── Save prompts ──
    prompts_path = os.path.join(args.save_dir, "naming_prompts.json")
    with open(prompts_path, 'w') as f:
        json.dump(prompts, f, indent=2)
    print(f"\n  Naming prompts saved to {prompts_path}")
    
    # ── Print suggested workflow ──
    print(f"\n{'='*70}")
    print("SUGGESTED NEXT STEPS")
    print(f"{'='*70}")
    print(f"""
  1. INSPECT per-feature images in {args.save_dir}/feature_*.png
     - Green section = top activating observations
     - Red section = inactive observations  
     - Compare: what's PRESENT in green but ABSENT in red?

  2. NAME CONCEPTS — For each feature, send to Claude:
     ┌─────────────────────────────────────────────────────────┐
     │ Prompt: "I'm analyzing a learned {args.game_name}       │
     │ policy. What visual concept does this feature detect?"  │
     │                                                         │
     │ [Attach: feature_XX.png]                                │
     └─────────────────────────────────────────────────────────┘

  3. REVIEW rule summaries in {args.save_dir}/rule_*.png
     - Each row = one literal in the clause
     - Positive literals show TOP activating obs
     - Negated literals show INACTIVE obs (what the rule requires to be absent)

  4. WRITE interpretable rules by replacing f_XX with concept names:
     e.g. Forward ← (path_clear ∧ ¬key_visible ∧ ¬wall_ahead ∧ ...)
""")
    
    # ── Print feature activation statistics ──
    print(f"\n{'='*70}")
    print("FEATURE ACTIVATION STATISTICS")
    print(f"{'='*70}")
    print(f"\n  {'Feature':<10} {'Density':>8} {'Mean(active)':>13} {'Max':>8} {'Actions'}")
    print(f"  {'-'*65}")
    for fidx in rule_features:
        z_col = all_z[:, fidx]
        density = (z_col > 0).float().mean().item()
        active = z_col[z_col > 0]
        mean_active = active.mean().item() if len(active) > 0 else 0
        max_val = z_col.max().item()
        acts = ', '.join(f"{'¬' if v == 'neg' else ''}{k}" 
                        for k, v in feature_info[fidx]['polarities'].items())
        print(f"  f_{fidx:<6} {density:>8.3f} {mean_active:>13.3f} {max_val:>8.3f}   {acts}")
    
    print(f"\nDone! All outputs in {args.save_dir}/")


if __name__ == "__main__":
    main()