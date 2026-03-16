# VisualizeConceptsmd.py
# Visualize and interpret learned concepts from Sparse Autoencoder

import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from ConceptExtractor import ActionConceptModel, SparseAutoencoder


def load_concept_model(model_dir='./concept_models'):
    """Load trained concept model and data"""
    print(f"Loading model from {model_dir}...")

    # Determine device (use MPS on Mac, CPU otherwise)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")

    # Load full model with device mapping
    checkpoint = torch.load(os.path.join(model_dir, 'concept_model.pt'),
                           map_location=device)
    config = checkpoint['config']

    model = ActionConceptModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        n_actions=config['n_actions'],
        k=config['k'],
        predictor_type=config['predictor_type'],
        binary=config.get('binary', False),
        gumbel=config.get('gumbel', False),
        tau=config.get('tau', 0.5)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load sample data with device mapping
    sample_data = torch.load(os.path.join(model_dir, 'sample_data.pt'),
                            map_location=device)
    features = sample_data['features']
    actions = sample_data['actions']

    print(f"Loaded model with {config['hidden_dim']} concepts")
    print(f"Sample data: {len(features)} samples")

    return model, features, actions, config


def find_top_activating_examples(model, features, concept_idx, k=10):
    """Find samples that maximally activate a specific concept"""
    with torch.no_grad():
        _, concepts = model.sae(features)
        activations = concepts[:, concept_idx]

        # Get top-k activating samples
        top_values, top_indices = torch.topk(activations, k)

    return top_indices, top_values


def visualize_observations_for_concept(ppo_model, env_name, concept_idx,
                                       top_indices, model, features, save_dir):
    """
    Visualize the actual observations that activate a concept
    by running the PPO model to reconstruct states
    """
    print(f"\nVisualizing concept {concept_idx}...")

    # Create environment
    def make_env():
        def _init():
            env = gym.make(env_name, render_mode='rgb_array')
            env = ImgObsWrapper(env)
            return env
        return _init

    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    # We'll collect observations by running episodes until we match the features
    # This is approximate - for exact visualization you'd need to save obs during collection
    print(f"Collecting observations (this may take a moment)...")

    obs_list = []
    feature_list = []

    obs = env.reset()
    max_steps = len(top_indices) * 100  # Heuristic

    with torch.no_grad():
        for step in range(max_steps):
            # Get features for current observation
            obs_tensor = torch.as_tensor(obs).float().to(ppo_model.device)
            current_features = ppo_model.policy.features_extractor(obs_tensor)

            # Get RGB observation for visualization
            rgb_obs = env.render()

            obs_list.append(rgb_obs)
            feature_list.append(current_features.cpu())

            # Step environment
            action, _ = ppo_model.predict(obs, deterministic=False)
            obs, _, dones, _ = env.step(action)

            if dones[0]:
                obs = env.reset()

            if len(obs_list) >= max_steps:
                break

    # Find closest matches to target features
    all_features = torch.cat(feature_list, dim=0)
    target_features = features[top_indices].cpu()  # Move to CPU for comparison

    # Match by L2 distance
    matched_indices = []
    for target_feat in target_features:
        distances = torch.norm(all_features - target_feat.unsqueeze(0), dim=1)
        matched_idx = distances.argmin().item()
        matched_indices.append(matched_idx)

    # Plot
    n_examples = min(len(matched_indices), 8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Concept {concept_idx}: Top Activating Examples', fontsize=16)

    # Get model device
    device = next(model.parameters()).device

    for idx, ax in enumerate(axes.flat):
        if idx < n_examples:
            obs_idx = matched_indices[idx]
            rgb = obs_list[obs_idx]

            # Get activation value
            with torch.no_grad():
                feat = all_features[obs_idx:obs_idx+1].to(device)
                _, concepts = model.sae(feat)
                activation = concepts[0, concept_idx].item()

            ax.imshow(rgb)
            ax.set_title(f'Activation: {activation:.3f}')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'concept_{concept_idx:03d}_examples.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved to {save_path}")


def plot_concept_activation_distribution(model, features, save_dir):
    """Plot distribution of concept activations"""
    print("\nPlotting concept activation distributions...")

    with torch.no_grad():
        _, concepts = model.sae(features)

    # Statistics
    activation_rates = (concepts > 0).float().mean(dim=0)
    mean_activations = concepts.mean(dim=0)
    max_activations = concepts.max(dim=0)[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Activation rate histogram
    axes[0, 0].hist(activation_rates.cpu().numpy() * 100, bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Activation Rate (%)')
    axes[0, 0].set_ylabel('Number of Concepts')
    axes[0, 0].set_title('Distribution of Concept Activation Rates')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Mean activation when active
    mean_when_active = []
    for i in range(concepts.shape[1]):
        active_mask = concepts[:, i] > 0
        if active_mask.sum() > 0:
            mean_when_active.append(concepts[active_mask, i].mean().item())
        else:
            mean_when_active.append(0)

    axes[0, 1].hist(mean_when_active, bins=50, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Mean Activation (when active)')
    axes[0, 1].set_ylabel('Number of Concepts')
    axes[0, 1].set_title('Distribution of Mean Activation Values')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Max activation
    axes[1, 0].hist(max_activations.cpu().numpy(), bins=50, edgecolor='black', color='green')
    axes[1, 0].set_xlabel('Max Activation')
    axes[1, 0].set_ylabel('Number of Concepts')
    axes[1, 0].set_title('Distribution of Maximum Activations')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Sparsity per sample
    sparsity_per_sample = (concepts > 0).float().sum(dim=1)
    axes[1, 1].hist(sparsity_per_sample.cpu().numpy(), bins=50, edgecolor='black', color='red')
    axes[1, 1].set_xlabel('Number of Active Concepts')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Distribution of Active Concepts per Sample')
    axes[1, 1].axvline(sparsity_per_sample.mean().item(), color='black',
                       linestyle='--', label=f'Mean: {sparsity_per_sample.mean():.1f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'activation_distributions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved to {save_path}")


def plot_concept_action_correlation(model, features, actions, save_dir):
    """Plot which concepts correlate with which actions"""
    print("\nAnalyzing concept-action correlations...")

    device = next(model.parameters()).device
    features = features.to(device)

    with torch.no_grad():
        _, concepts = model.sae(features)
        action_logits = model.action_predictor(concepts)

    concepts = concepts.cpu().numpy()
    actions = actions.cpu().numpy()

    n_concepts = concepts.shape[1]
    n_actions = action_logits.shape[1]

    # Compute correlation matrix: concept activations vs actions
    correlation_matrix = np.zeros((n_concepts, n_actions))

    for action_idx in range(n_actions):
        action_mask = (actions == action_idx)
        if action_mask.sum() > 0:
            # Mean activation of each concept when this action is taken
            correlation_matrix[:, action_idx] = concepts[action_mask].mean(axis=0)

    # Normalize by overall mean
    overall_mean = concepts.mean(axis=0, keepdims=True).T
    correlation_matrix = correlation_matrix / (overall_mean + 1e-8)

    # Plot top correlated concepts for each action
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Top Concepts for Each Action', fontsize=16)

    action_names = ['Turn Left', 'Turn Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']

    for action_idx, ax in enumerate(axes.flat):
        if action_idx < n_actions:
            # Get top concepts for this action
            top_concepts = np.argsort(correlation_matrix[:, action_idx])[-15:][::-1]
            top_scores = correlation_matrix[top_concepts, action_idx]

            ax.barh(range(len(top_concepts)), top_scores)
            ax.set_yticks(range(len(top_concepts)))
            ax.set_yticklabels([f'C{c}' for c in top_concepts])
            ax.set_xlabel('Relative Activation')
            ax.set_title(f'{action_names[action_idx]} (Action {action_idx})')
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'concept_action_correlation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved to {save_path}")

    return correlation_matrix


def analyze_action_predictor_weights(model, save_dir):
    """Analyze learned action predictor weights"""
    print("\nAnalyzing action predictor weights...")

    # Get weights
    if isinstance(model.action_predictor, torch.nn.Linear):
        weights = model.action_predictor.weight.data.cpu().numpy()  # (n_actions, n_concepts)
    else:
        # For MLP, get first layer weights
        weights = model.action_predictor[0].weight.data.cpu().numpy()

    n_actions, n_concepts = weights.shape

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 6))

    im = ax.imshow(weights, aspect='auto', cmap='RdBu_r',
                   vmin=-weights.max(), vmax=weights.max())

    action_names = ['Turn Left', 'Turn Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']
    ax.set_yticks(range(n_actions))
    ax.set_yticklabels(action_names[:n_actions])
    ax.set_xlabel('Concept Index')
    ax.set_ylabel('Action')
    ax.set_title('Action Predictor Weights (Red=Positive, Blue=Negative)')

    plt.colorbar(im, ax=ax, label='Weight Value')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'action_predictor_weights.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved to {save_path}")

    # Print top concepts for each action
    print("\nTop positive concepts for each action:")
    for action_idx, action_name in enumerate(action_names[:n_actions]):
        top_concepts = np.argsort(weights[action_idx])[-5:][::-1]
        top_weights = weights[action_idx, top_concepts]
        print(f"  {action_name}:")
        for concept_idx, weight in zip(top_concepts, top_weights):
            print(f"    Concept {concept_idx:3d}: {weight:+.3f}")


def generate_concept_summary(model, features, actions, save_dir, top_k=20):
    """Generate a text summary of the most important concepts"""
    print("\nGenerating concept summary...")

    device = next(model.parameters()).device
    features = features.to(device)

    with torch.no_grad():
        _, concepts = model.sae(features)

    concepts_np = concepts.cpu().numpy()
    actions_np = actions.cpu().numpy()

    # Metrics for each concept
    activation_rate = (concepts_np > 0).mean(axis=0)
    mean_activation = concepts_np.mean(axis=0)
    max_activation = concepts_np.max(axis=0)

    # Action predictor weights
    if isinstance(model.action_predictor, torch.nn.Linear):
        weights = model.action_predictor.weight.data.cpu().numpy()
    else:
        weights = model.action_predictor[0].weight.data.cpu().numpy()

    # Find most important concepts (by various metrics)
    importance_scores = activation_rate * max_activation  # Combined metric
    top_concepts = np.argsort(importance_scores)[-top_k:][::-1]

    # Write summary
    summary_path = os.path.join(save_dir, 'concept_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CONCEPT SUMMARY\n")
        f.write("="*80 + "\n\n")

        for rank, concept_idx in enumerate(top_concepts):
            f.write(f"Concept {concept_idx:3d} (Rank {rank+1}):\n")
            f.write(f"  Activation rate: {activation_rate[concept_idx]*100:5.1f}%\n")
            f.write(f"  Mean activation: {mean_activation[concept_idx]:.3f}\n")
            f.write(f"  Max activation:  {max_activation[concept_idx]:.3f}\n")

            # Top actions influenced by this concept
            action_names = ['Turn Left', 'Turn Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']
            concept_weights = weights[:, concept_idx]

            # Show top positive and negative influences
            positive_indices = np.where(concept_weights > 0)[0]
            negative_indices = np.where(concept_weights < 0)[0]

            if len(positive_indices) > 0:
                top_positive = positive_indices[np.argsort(concept_weights[positive_indices])[-2:][::-1]]
                f.write(f"  Most encouraged actions:\n")
                for action_idx in top_positive:
                    f.write(f"    {action_names[action_idx]:12s}: {concept_weights[action_idx]:+.3f}\n")

            if len(negative_indices) > 0:
                top_negative = negative_indices[np.argsort(concept_weights[negative_indices])[:2]]
                f.write(f"  Most inhibited actions:\n")
                for action_idx in top_negative:
                    f.write(f"    {action_names[action_idx]:12s}: {concept_weights[action_idx]:+.3f}\n")

            # When is this concept active?
            active_samples = np.where(concepts_np[:, concept_idx] > 0)[0]
            if len(active_samples) > 0:
                actions_when_active = actions_np[active_samples]
                unique, counts = np.unique(actions_when_active, return_counts=True)
                top_action_when_active = unique[np.argmax(counts)]
                f.write(f"  Most common action when active: {action_names[top_action_when_active]}")
                # Show the weight for this action to explain the connection
                f.write(f" (weight: {concept_weights[top_action_when_active]:+.3f})\n")

            f.write("\n")

    print(f"Saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize learned concepts')
    parser.add_argument('--model_dir', type=str, default='./concept_models',
                        help='Directory containing trained model')
    parser.add_argument('--ppo_path', type=str, default='ppo_doorkey_5x5.zip',
                        help='Path to trained PPO model (for observation visualization)')
    parser.add_argument('--env_name', type=str, default='MiniGrid-DoorKey-5x5-v0',
                        help='Environment name')
    parser.add_argument('--save_dir', type=str, default='./concept_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--visualize_concepts', type=str, default='',
                        help='Comma-separated concept indices to visualize (e.g., "0,1,5,10")')
    parser.add_argument('--top_k_concepts', type=int, default=10,
                        help='Number of top concepts to visualize')
    parser.add_argument('--skip_obs_viz', action='store_true',
                        help='Skip observation visualization (faster)')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model and data
    model, features, actions, config = load_concept_model(args.model_dir)

    # 1. Plot activation distributions
    plot_concept_activation_distribution(model, features, args.save_dir)

    # 2. Concept-action correlation
    correlation_matrix = plot_concept_action_correlation(model, features, actions, args.save_dir)

    # 3. Action predictor weights
    analyze_action_predictor_weights(model, args.save_dir)

    # 4. Generate text summary
    generate_concept_summary(model, features, actions, args.save_dir, top_k=20)

    # 5. Visualize specific concepts
    if not args.skip_obs_viz:
        # Load PPO model for observation reconstruction
        print(f"\nLoading PPO model from {args.ppo_path}...")
        # Determine device
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        ppo_model = PPO.load(args.ppo_path, device=device)

        # Determine which concepts to visualize
        if args.visualize_concepts:
            concept_indices = [int(x.strip()) for x in args.visualize_concepts.split(',')]
        else:
            # Auto-select top concepts
            with torch.no_grad():
                _, concepts = model.sae(features)
            activation_rates = (concepts > 0).float().mean(dim=0)
            max_activations = concepts.max(dim=0)[0]
            importance = activation_rates * max_activations
            top_indices = importance.argsort(descending=True)[:args.top_k_concepts]
            concept_indices = top_indices.tolist()

        print(f"\nVisualizing concepts: {concept_indices}")

        for concept_idx in concept_indices:
            top_indices, top_values = find_top_activating_examples(
                model, features, concept_idx, k=8
            )
            visualize_observations_for_concept(
                ppo_model, args.env_name, concept_idx,
                top_indices, model, features, args.save_dir
            )
    else:
        print("\nSkipping observation visualization (use --visualize_concepts to enable)")

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"All visualizations saved to: {args.save_dir}")
    print(f"\nGenerated files:")
    print(f"  - activation_distributions.png")
    print(f"  - concept_action_correlation.png")
    print(f"  - action_predictor_weights.png")
    print(f"  - concept_summary.txt")
    if not args.skip_obs_viz:
        print(f"  - concept_XXX_examples.png (for each visualized concept)")


if __name__ == '__main__':
    main()
