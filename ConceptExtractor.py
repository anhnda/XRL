# ConceptExtractor.py
# Simple Sparse Autoencoder with Gumbel-Softmax and TopK sparsity

import argparse
import os
import random

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
    """Sparse Autoencoder with Gumbel-Softmax and TopK sparsity"""

    def __init__(self, input_dim=128, hidden_dim=64, k=10, tau=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k  # Top-K sparsity
        self.tau = tau  # Gumbel temperature

        # Encoder and decoder
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        # Initialize
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def gumbel_sigmoid(self, logits, tau=1.0, hard=True):
        """Gumbel-Sigmoid for differentiable binary sampling"""
        if not self.training and hard:
            return (torch.sigmoid(logits) > 0.5).float()

        # Gumbel noise
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)

        # Gumbel-Sigmoid
        y_soft = torch.sigmoid((logits + gumbel) / tau)

        if hard:
            # Straight-through estimator
            y_hard = (y_soft > 0.5).float()
            y = (y_hard - y_soft).detach() + y_soft
        else:
            y = y_soft

        return y

    def topk_mask(self, h, k):
        """Keep only top-k activations per sample"""
        # Get top-k indices
        _, indices = torch.topk(h, k, dim=-1)

        # Create mask
        mask = torch.zeros_like(h)
        mask.scatter_(-1, indices, 1.0)

        return h * mask

    def encode(self, x):
        """Encode to sparse binary concepts"""
        logits = self.encoder(x)

        # Gumbel-Sigmoid for binary activations
        h = self.gumbel_sigmoid(logits, tau=self.tau, hard=True)

        # TopK sparsity
        concepts = self.topk_mask(h, self.k)

        return concepts

    def decode(self, concepts):
        """Decode back to features"""
        return self.decoder(concepts)

    def forward(self, x):
        concepts = self.encode(x)
        x_recon = self.decode(concepts)
        return x_recon, concepts


def collect_rollout_data(model_path, env_name, n_episodes=500, seed=42):
    """Collect features from PPO rollouts"""
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

    features = torch.cat(features_list, dim=0)
    actions = torch.cat(actions_list, dim=0)

    print(f"Collected {len(features)} samples")
    return features, actions


def train_sae(features, actions, config):
    """Train the Sparse Autoencoder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make cudnn deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

    # Create model
    model = SparseAutoencoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        k=config['k'],
        tau=config['tau']
    ).to(device)

    # Optional action predictor
    if config.get('use_action_predictor', False):
        action_predictor = nn.Linear(config['hidden_dim'], config['n_actions']).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(action_predictor.parameters()),
            lr=config['lr']
        )
    else:
        action_predictor = None
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Dataset with deterministic DataLoader
    dataset = TensorDataset(features, actions)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        generator=generator,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )

    # Loss weights
    alpha = config.get('alpha', 1.0)  # Reconstruction
    beta = config.get('beta', 0.0)    # Action prediction

    print(f"\nTraining for {config['n_epochs']} epochs...")
    print(f"Loss weights: α(recon)={alpha}, β(action)={beta}")

    for epoch in range(config['n_epochs']):
        total_loss = 0
        total_recon = 0
        total_action = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['n_epochs']}")

        for batch_features, batch_actions in pbar:
            batch_features = batch_features.to(device)
            batch_actions = batch_actions.to(device)

            # Forward
            x_recon, concepts = model(batch_features)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, batch_features)

            # Total loss
            loss = alpha * recon_loss

            # Optional action loss
            if action_predictor is not None:
                action_logits = action_predictor(concepts)
                action_loss = F.cross_entropy(action_logits, batch_actions)
                loss += beta * action_loss

                pred = action_logits.argmax(dim=-1)
                total_correct += (pred == batch_actions).sum().item()
                total_action += action_loss.item()
            else:
                action_loss = torch.tensor(0.0)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_samples += len(batch_actions)

            # Progress
            if action_predictor is not None:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'action': f'{action_loss.item():.4f}',
                    'acc': f'{100.0 * total_correct / total_samples:.1f}%'
                })
            else:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}'
                })

        # Epoch summary
        print(f"Epoch {epoch+1}:")
        print(f"  Recon Loss: {total_recon / len(dataloader):.4f}")
        if action_predictor is not None:
            print(f"  Action Loss: {total_action / len(dataloader):.4f}")
            print(f"  Action Acc: {100.0 * total_correct / total_samples:.2f}%")

            # Per-action accuracy (every 20 epochs)
            if (epoch + 1) % 20 == 0:
                with torch.no_grad():
                    all_preds = []
                    all_targets = []
                    for batch_features, batch_actions in dataloader:
                        batch_features = batch_features.to(device)
                        batch_actions = batch_actions.to(device)
                        _, concepts = model(batch_features)
                        logits = action_predictor(concepts)
                        preds = logits.argmax(dim=-1)
                        all_preds.append(preds.cpu())
                        all_targets.append(batch_actions.cpu())

                    all_preds = torch.cat(all_preds)
                    all_targets = torch.cat(all_targets)

                    action_names = ['TurnLeft', 'TurnRight', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']
                    print(f"  Per-action accuracy:")
                    for a in range(config['n_actions']):
                        mask = all_targets == a
                        if mask.sum() > 0:
                            acc = (all_preds[mask] == a).float().mean().item()
                            count = mask.sum().item()
                            print(f"    {action_names[a]:10s}: {acc*100:5.1f}% ({count:5d} samples)")
        else:
            print(f"  Action Acc: (no action predictor)")

        # Sparsity and interpretability analysis
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_features = features[:1000].to(device)
                sample_actions = actions[:1000].to(device)
                _, sample_concepts = model(sample_features)

                # Sparsity stats
                active_rate = (sample_concepts > 0).float().mean(dim=0)
                n_active = (active_rate > 0.01).sum().item()
                avg_sparsity = (sample_concepts > 0).float().mean().item()
                n_saturated = (active_rate > 0.9).sum().item()

                print(f"  Active concepts: {n_active}/{config['hidden_dim']}")
                print(f"  Saturated (>90%): {n_saturated}")
                print(f"  Avg sparsity: {100 * avg_sparsity:.1f}%")

                # Quick linear probe to measure interpretability
                if action_predictor is None:
                    # Fit quick linear probe to see if concepts are informative
                    probe = torch.nn.Linear(config['hidden_dim'], config['n_actions']).to(device)
                    probe_optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)

                    for _ in range(50):  # Quick 50 step training
                        logits = probe(sample_concepts)
                        loss = F.cross_entropy(logits, sample_actions)
                        probe_optimizer.zero_grad()
                        loss.backward()
                        probe_optimizer.step()

                    # Test accuracy
                    with torch.no_grad():
                        logits = probe(sample_concepts)
                        pred = logits.argmax(dim=-1)
                        probe_acc = (pred == sample_actions).float().mean().item()
                        print(f"  Linear probe acc: {100.0 * probe_acc:.2f}%")

    if action_predictor is not None:
        return model, action_predictor
    return model, None


def save_model(model, action_predictor, features, actions, config, save_dir='./concept_models'):
    """Save model"""
    os.makedirs(save_dir, exist_ok=True)

    # Save SAE
    torch.save({
        'encoder_weight': model.encoder.weight.data,
        'encoder_bias': model.encoder.bias.data,
        'decoder_weight': model.decoder.weight.data,
        'decoder_bias': model.decoder.bias.data,
        'config': config
    }, os.path.join(save_dir, 'sae.pt'))

    # Save action predictor if exists
    if action_predictor is not None:
        torch.save({
            'weight': action_predictor.weight.data,
            'bias': action_predictor.bias.data
        }, os.path.join(save_dir, 'action_predictor.pt'))

    # Save sample data
    torch.save({
        'features': features[:1000],
        'actions': actions[:1000]
    }, os.path.join(save_dir, 'sample_data.pt'))

    print(f"\nModel saved to {save_dir}")


def analyze_concepts(model, features):
    """Analyze learned concepts"""
    device = next(model.parameters()).device
    features = features.to(device)

    print("\n" + "="*60)
    print("CONCEPT ANALYSIS")
    print("="*60)

    with torch.no_grad():
        _, concepts = model(features)

        active_mask = concepts > 0
        active_rate = active_mask.float().mean(dim=0)

        n_active = (active_rate > 0.01).sum().item()
        n_dead = (active_rate < 0.001).sum().item()

        print(f"\nActive concepts (>1%): {n_active}/{concepts.shape[1]}")
        print(f"Dead concepts (<0.1%): {n_dead}/{concepts.shape[1]}")
        print(f"Avg sparsity: {active_mask.float().mean().item()*100:.1f}%")

        # Top concepts
        print(f"\nTop 10 most active concepts:")
        top_indices = active_rate.argsort(descending=True)[:10]
        for rank, idx in enumerate(top_indices):
            print(f"  {rank+1}. Concept {idx.item():3d}: {active_rate[idx].item()*100:5.1f}% active")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='ppo_doorkey_5x5.zip')
    parser.add_argument('--env_name', type=str, default='MiniGrid-DoorKey-5x5-v0')
    parser.add_argument('--n_episodes', type=int, default=800)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--beta', type=float, default=0.0, help='Action loss weight (0=no action predictor)')
    parser.add_argument('--save_dir', type=str, default='./concept_models')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seeds for data collection
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Collect data
    features, actions = collect_rollout_data(args.model_path, args.env_name, args.n_episodes, args.seed)

    # Config
    config = {
        'input_dim': 128,
        'hidden_dim': args.hidden_dim,
        'k': args.k,
        'tau': args.tau,
        'n_actions': 7,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'alpha': args.alpha,
        'beta': args.beta,
        'use_action_predictor': args.beta > 0,
        'env_name': args.env_name,
        'model_path': args.model_path,
        'seed': args.seed
    }

    # Train
    model, action_predictor = train_sae(features, actions, config)

    # Analyze
    analyze_concepts(model, features)

    # Save
    save_model(model, action_predictor, features, actions, config, args.save_dir)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
