# ConceptExtractor.py
# Train Sparse Autoencoder to extract interpretable concepts from PPO policy features

import argparse
import os
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with TopK sparsity constraint"""

    def __init__(self, input_dim=128, hidden_dim=256, k=25):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k  # Top-K sparsity

        # Encoder: maps features to sparse concepts
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder: reconstructs original features from concepts
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialize with small weights for better sparsity
        nn.init.xavier_uniform_(self.encoder.weight, gain=0.5)
        nn.init.xavier_uniform_(self.decoder.weight, gain=0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        """Encode features to sparse concepts"""
        h = F.relu(self.encoder(x))  # Positive activations only
        return self.topk_sparse(h, self.k)

    def topk_sparse(self, h, k):
        """Keep only top-k activations per sample"""
        batch_size = h.shape[0]

        # Get top-k values and indices
        values, indices = torch.topk(h, k, dim=-1)

        # Create sparse tensor
        sparse_h = torch.zeros_like(h)
        sparse_h.scatter_(-1, indices, values)

        return sparse_h

    def decode(self, h):
        """Reconstruct original features from concepts"""
        return self.decoder(h)

    def forward(self, x):
        concepts = self.encode(x)
        x_recon = self.decode(concepts)
        return x_recon, concepts


class ActionConceptModel(nn.Module):
    """
    Combined model: SAE + Action Predictor
    Learns concepts that both reconstruct features AND predict actions
    """

    def __init__(self, input_dim=128, hidden_dim=256, n_actions=7, k=25,
                 predictor_type='linear'):
        super().__init__()
        self.sae = SparseAutoencoder(input_dim, hidden_dim, k)

        # Action predictor: concepts -> action logits
        if predictor_type == 'linear':
            self.action_predictor = nn.Linear(hidden_dim, n_actions)
        elif predictor_type == 'mlp':
            self.action_predictor = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions)
            )
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")

    def forward(self, features):
        x_recon, concepts = self.sae(features)
        action_logits = self.action_predictor(concepts)
        return x_recon, concepts, action_logits


def collect_rollout_data(model_path, env_name, n_episodes=100, seed=42):
    """
    Collect features and actions from trained PPO model rollouts

    Returns:
        features: (N, 128) tensor of CNN features
        actions: (N,) tensor of actions taken
    """
    print(f"Loading PPO model from {model_path}...")
    model = PPO.load(model_path)

    # Create environment
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

    print(f"Collecting rollout data from {n_episodes} episodes...")
    obs = env.reset()
    episode_count = 0

    with torch.no_grad():
        pbar = tqdm(total=n_episodes)
        while episode_count < n_episodes:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=False)

            # Extract features from the CNN
            # Access the features extractor's cached features
            obs_tensor = torch.as_tensor(obs).float().to(model.device)
            features = model.policy.features_extractor(obs_tensor)

            features_list.append(features.cpu())
            actions_list.append(torch.tensor(action))

            # Step environment
            obs, rewards, dones, infos = env.step(action)

            if dones[0]:
                episode_count += 1
                pbar.update(1)
                obs = env.reset()

        pbar.close()

    # Concatenate all data
    features = torch.cat(features_list, dim=0)
    actions = torch.cat(actions_list, dim=0)

    print(f"Collected {len(features)} samples")
    print(f"Features shape: {features.shape}")
    print(f"Actions shape: {actions.shape}")

    return features, actions


def train_concept_model(features, actions, config):
    """
    Train the ActionConceptModel with multi-objective loss

    Args:
        features: (N, 128) tensor
        actions: (N,) tensor
        config: dict with training hyperparameters

    Returns:
        trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Create model
    model = ActionConceptModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        n_actions=config['n_actions'],
        k=config['k'],
        predictor_type=config['predictor_type']
    ).to(device)

    # Create dataset and dataloader
    dataset = TensorDataset(features, actions)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Loss weights
    alpha = config['alpha']  # Reconstruction loss weight
    beta = config['beta']    # Action loss weight

    # Training loop
    print(f"\nTraining for {config['n_epochs']} epochs...")
    print(f"Loss weights: α(recon)={alpha}, β(action)={beta}")

    for epoch in range(config['n_epochs']):
        total_loss = 0
        total_recon_loss = 0
        total_action_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['n_epochs']}")
        for batch_features, batch_actions in pbar:
            batch_features = batch_features.to(device)
            batch_actions = batch_actions.to(device)

            # Forward pass
            x_recon, concepts, action_logits = model(batch_features)

            # Compute losses
            recon_loss = F.mse_loss(x_recon, batch_features)
            action_loss = F.cross_entropy(action_logits, batch_actions)

            # Multi-objective loss
            loss = alpha * recon_loss + beta * action_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_action_loss += action_loss.item()

            # Accuracy
            pred_actions = action_logits.argmax(dim=-1)
            total_correct += (pred_actions == batch_actions).sum().item()
            total_samples += len(batch_actions)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'action': f'{action_loss.item():.4f}',
                'acc': f'{100.0 * total_correct / total_samples:.1f}%'
            })

        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_action = total_action_loss / len(dataloader)
        accuracy = 100.0 * total_correct / total_samples

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Recon Loss: {avg_recon:.4f}")
        print(f"  Action Loss: {avg_action:.4f}")
        print(f"  Action Accuracy: {accuracy:.2f}%")

        # Analyze sparsity
        with torch.no_grad():
            sample_features = features[:1000].to(device)
            _, sample_concepts = model.sae(sample_features)
            active_rate = (sample_concepts > 0).float().mean(dim=0)
            n_active_concepts = (active_rate > 0.01).sum().item()
            avg_sparsity = (sample_concepts > 0).float().mean().item()

            print(f"  Active concepts: {n_active_concepts}/{config['hidden_dim']}")
            print(f"  Avg sparsity: {100 * avg_sparsity:.1f}%")

    return model


def save_model(model, features, actions, config, save_dir='./concept_models'):
    """Save trained model and metadata"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(save_dir, 'concept_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Save SAE separately for easy loading
    sae_path = os.path.join(save_dir, 'sae.pt')
    torch.save({
        'encoder_weight': model.sae.encoder.weight.data,
        'encoder_bias': model.sae.encoder.bias.data,
        'decoder_weight': model.sae.decoder.weight.data,
        'decoder_bias': model.sae.decoder.bias.data,
        'config': {
            'input_dim': config['input_dim'],
            'hidden_dim': config['hidden_dim'],
            'k': config['k']
        }
    }, sae_path)
    print(f"SAE saved to {sae_path}")

    # Save sample data for analysis
    sample_path = os.path.join(save_dir, 'sample_data.pt')
    torch.save({
        'features': features[:1000],
        'actions': actions[:1000]
    }, sample_path)
    print(f"Sample data saved to {sample_path}")


def analyze_concepts(model, features, actions, top_k=10):
    """Analyze learned concepts"""
    device = next(model.parameters()).device
    features = features.to(device)

    print("\n" + "="*60)
    print("CONCEPT ANALYSIS")
    print("="*60)

    with torch.no_grad():
        _, concepts = model.sae(features)

        # 1. Overall statistics
        print("\n1. Overall Statistics:")
        active_mask = concepts > 0
        active_rate = active_mask.float().mean(dim=0)
        n_active = (active_rate > 0.01).sum().item()
        n_dead = (active_rate < 0.001).sum().item()

        print(f"   Active concepts (>1% activation): {n_active}/{concepts.shape[1]}")
        print(f"   Dead concepts (<0.1% activation): {n_dead}/{concepts.shape[1]}")
        print(f"   Average sparsity per sample: {active_mask.float().mean().item()*100:.1f}%")

        # 2. Most active concepts
        print("\n2. Most Active Concepts (by activation rate):")
        top_active_indices = active_rate.argsort(descending=True)[:top_k]
        for rank, idx in enumerate(top_active_indices):
            idx_val = idx.item()
            rate = active_rate[idx].item()
            avg_value = concepts[active_mask[:, idx_val], idx_val].mean().item()
            print(f"   Rank {rank+1}: Concept {idx_val:3d} - Active {rate*100:5.1f}% of time, Avg value: {avg_value:.3f}")

        # 3. Strongest concepts
        print("\n3. Strongest Concepts (by max activation):")
        max_activations = concepts.max(dim=0)[0]
        top_strong_indices = max_activations.argsort(descending=True)[:top_k]
        for rank, idx in enumerate(top_strong_indices):
            idx_val = idx.item()
            max_val = max_activations[idx].item()
            rate = active_rate[idx].item()
            print(f"   Rank {rank+1}: Concept {idx_val:3d} - Max: {max_val:.3f}, Active: {rate*100:5.1f}%")

        # 4. Action prediction analysis
        print("\n4. Action Prediction Analysis:")
        action_logits = model.action_predictor(concepts)
        pred_actions = action_logits.argmax(dim=-1)
        accuracy = (pred_actions == actions.to(device)).float().mean().item()
        print(f"   Overall accuracy: {accuracy*100:.2f}%")

        # Per-action accuracy
        n_actions = action_logits.shape[-1]
        print(f"   Per-action accuracy:")
        for a in range(n_actions):
            mask = actions == a
            if mask.sum() > 0:
                acc = (pred_actions[mask.to(device)] == a).float().mean().item()
                count = mask.sum().item()
                print(f"     Action {a}: {acc*100:5.1f}% ({count:5d} samples)")


def main():
    parser = argparse.ArgumentParser(description='Train Sparse Autoencoder for concept extraction')
    parser.add_argument('--model_path', type=str, default='ppo_doorkey_5x5.zip',
                        help='Path to trained PPO model')
    parser.add_argument('--env_name', type=str, default='MiniGrid-DoorKey-5x5-v0',
                        help='Environment name')
    parser.add_argument('--n_episodes', type=int, default=200,
                        help='Number of episodes to collect')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='SAE hidden dimension')
    parser.add_argument('--k', type=int, default=5,
                        help='Top-K sparsity')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Action loss weight')
    parser.add_argument('--predictor_type', type=str, default='linear',
                        choices=['linear', 'mlp'], help='Action predictor type')
    parser.add_argument('--save_dir', type=str, default='./concept_models',
                        help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Collect rollout data
    features, actions = collect_rollout_data(
        args.model_path,
        args.env_name,
        n_episodes=args.n_episodes,
        seed=args.seed
    )

    # 2. Training config
    config = {
        'input_dim': 128,
        'hidden_dim': args.hidden_dim,
        'n_actions': 7,  # MiniGrid has 7 actions
        'k': args.k,
        'predictor_type': args.predictor_type,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'alpha': args.alpha,
        'beta': args.beta,
        'env_name': args.env_name,
        'model_path': args.model_path,
    }

    # 3. Train model
    model = train_concept_model(features, actions, config)

    # 4. Analyze concepts
    analyze_concepts(model, features, actions, top_k=15)

    # 5. Save model
    save_model(model, features, actions, config, args.save_dir)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {args.save_dir}")
    print(f"\nTo load the model:")
    print(f"  device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')")
    print(f"  checkpoint = torch.load('{args.save_dir}/concept_model.pt', map_location=device)")
    print(f"  model.load_state_dict(checkpoint['model_state_dict'])")
    print(f"  model.to(device)")


if __name__ == '__main__':
    main()
