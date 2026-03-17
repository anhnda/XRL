# ConceptExtractorS.py
# SAE with temporal consistency, future prediction, and augmentation losses

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
        # nn.init.zeros_(self.encoder.weight)
        # nn.init.zeros_(self.decoder.weight)
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
    """Collect sequential features from PPO rollouts for temporal consistency"""
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
    next_features_list = []
    next_actions_list = []
    episode_mask_list = []  # 1 if next step is in same episode, 0 if episode boundary

    print(f"Collecting {n_episodes} episodes...")
    obs = env.reset()
    episode_count = 0

    with torch.no_grad():
        pbar = tqdm(total=n_episodes)
        episode_features = []
        episode_actions = []

        while episode_count < n_episodes:
            # Get current features and action
            obs_tensor = torch.as_tensor(obs).float().to(model.device)
            features = model.policy.features_extractor(obs_tensor)
            action, _ = model.predict(obs, deterministic=False)

            episode_features.append(features.cpu())
            episode_actions.append(torch.tensor(action))

            # Step environment
            obs, rewards, dones, infos = env.step(action)

            if dones[0]:
                # End of episode - process episode data
                if len(episode_features) > 1:
                    for t in range(len(episode_features) - 1):
                        features_list.append(episode_features[t])
                        actions_list.append(episode_actions[t])
                        next_features_list.append(episode_features[t + 1])
                        next_actions_list.append(episode_actions[t + 1])
                        episode_mask_list.append(torch.tensor([1.0]))  # Valid transition

                # Reset episode buffers
                episode_features = []
                episode_actions = []
                episode_count += 1
                pbar.update(1)
                obs = env.reset()

        pbar.close()

    features = torch.cat(features_list, dim=0)
    actions = torch.cat(actions_list, dim=0)
    next_features = torch.cat(next_features_list, dim=0)
    next_actions = torch.cat(next_actions_list, dim=0)
    episode_mask = torch.cat(episode_mask_list, dim=0)

    print(f"Collected {len(features)} sequential samples")
    return features, actions, next_features, next_actions, episode_mask


def train_sae(features, actions, next_features, next_actions, episode_mask, config):
    """Train the Sparse Autoencoder with temporal consistency"""
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

    # Split data into train/val
    n_samples = len(features)
    val_ratio = config.get('val_ratio', 0.1)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val

    indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_features = features[train_indices]
    train_actions = actions[train_indices]
    train_next_features = next_features[train_indices]
    train_next_actions = next_actions[train_indices]
    train_episode_mask = episode_mask[train_indices]

    val_features = features[val_indices]
    val_actions = actions[val_indices]
    val_next_features = next_features[val_indices]
    val_next_actions = next_actions[val_indices]
    val_episode_mask = episode_mask[val_indices]

    print(f"\nDataset split: {n_train} train, {n_val} val ({val_ratio*100:.0f}% validation)")

    # Create model
    model = SparseAutoencoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        k=config['k'],
        tau=config['tau']
    ).to(device)

    # Optional action predictor
    action_predictor = None
    future_predictor = None

    if config.get('use_action_predictor', False):
        action_predictor = nn.Linear(config['hidden_dim'], config['n_actions']).to(device)
        # Future action predictor for future loss
        future_predictor = nn.Linear(config['hidden_dim'], config['n_actions']).to(device)

        optimizer = torch.optim.Adam(
            list(model.parameters()) +
            list(action_predictor.parameters()) +
            list(future_predictor.parameters()),
            lr=config['lr']
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Dataset with deterministic DataLoader (includes sequential data)
    train_dataset = TensorDataset(train_features, train_actions, train_next_features, train_next_actions, train_episode_mask)
    val_dataset = TensorDataset(val_features, val_actions, val_next_features, val_next_actions, val_episode_mask)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        generator=generator,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Loss weights
    alpha = config.get('alpha', 1.0)  # Reconstruction
    beta = config.get('beta', 0.0)    # Action prediction
    gamma = config.get('gamma', 0.0)  # Action-concept diversity
    delta = config.get('delta', 0.01)  # L1 sparsity on action predictor weights
    epsilon = config.get('epsilon', 0.1)  # Temporal consistency
    zeta = config.get('zeta', 0.1)  # Future prediction
    eta = config.get('eta', 0.05)  # Augmentation invariance

    # Track best model
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience = config.get('patience', 50)
    patience_counter = 0
    best_model_state = None

    print(f"\nTraining for {config['n_epochs']} epochs...")
    print(f"Loss weights: α(recon)={alpha}, β(action)={beta}, γ(diversity)={gamma}, δ(L1)={delta}")
    print(f"              ε(temporal)={epsilon}, ζ(future)={zeta}, η(aug)={eta}")
    print(f"Early stopping patience: {patience}")

    for epoch in range(config['n_epochs']):
        # === TRAINING ===
        model.train()
        if action_predictor is not None:
            action_predictor.train()
            if future_predictor is not None:
                future_predictor.train()

        total_loss = 0
        total_recon = 0
        total_action = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"[TRAIN] Epoch {epoch+1}/{config['n_epochs']}")

        for batch_features, batch_actions, batch_next_features, batch_next_actions, batch_mask in pbar:
            batch_features = batch_features.to(device)
            batch_actions = batch_actions.to(device)
            batch_next_features = batch_next_features.to(device)
            batch_next_actions = batch_next_actions.to(device)
            batch_mask = batch_mask.to(device)

            # Forward
            x_recon, concepts = model(batch_features)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, batch_features)

            # Total loss
            loss = alpha * recon_loss

            # Temporal consistency loss: concepts should be similar at adjacent timesteps
            if epsilon > 0:
                _, next_concepts = model(batch_next_features)
                # L1 distance weighted by episode mask (0 at episode boundaries)
                temporal_loss = (batch_mask * (concepts - next_concepts).abs().sum(dim=1)).mean()
                loss += epsilon * temporal_loss
            else:
                temporal_loss = torch.tensor(0.0)

            # Augmentation invariance loss: add small noise to features, concepts should stay same
            if eta > 0:
                # Add small Gaussian noise to features
                noise = torch.randn_like(batch_features) * 0.01
                aug_features = batch_features + noise
                _, aug_concepts = model(aug_features)
                # L1 distance between original and augmented concepts
                aug_loss = (concepts - aug_concepts).abs().sum(dim=1).mean()
                loss += eta * aug_loss
            else:
                aug_loss = torch.tensor(0.0)

            # Optional action loss
            if action_predictor is not None:
                action_logits = action_predictor(concepts)
                action_loss = F.cross_entropy(action_logits, batch_actions)
                loss += beta * action_loss

                # L1 penalty on action predictor weights: encourage each action to use few concepts
                # This makes the action->concept mapping sparse
                if delta > 0:
                    l1_penalty = action_predictor.weight.abs().mean()
                    loss += delta * l1_penalty

                pred = action_logits.argmax(dim=-1)
                total_correct += (pred == batch_actions).sum().item()
                total_action += action_loss.item()

                # Future prediction loss: concepts should predict next action
                if zeta > 0 and future_predictor is not None:
                    future_logits = future_predictor(concepts)
                    # Mask out episode boundaries
                    future_loss = (batch_mask.squeeze() * F.cross_entropy(
                        future_logits, batch_next_actions, reduction='none'
                    )).mean()
                    loss += zeta * future_loss
                else:
                    future_loss = torch.tensor(0.0)

                # Action-concept diversity loss: encourage different actions to use different concepts
                if gamma > 0 and len(torch.unique(batch_actions)) > 1:
                    # Compute mean concept activation for each action in batch
                    action_concept_patterns = []
                    unique_actions = torch.unique(batch_actions)

                    for action in unique_actions:
                        action_mask = batch_actions == action
                        if action_mask.sum() > 0:
                            # Mean concept activation for this action
                            action_concepts = concepts[action_mask].mean(dim=0)
                            action_concept_patterns.append(action_concepts)

                    if len(action_concept_patterns) > 1:
                        # Stack into matrix: [n_actions_in_batch, hidden_dim]
                        action_patterns = torch.stack(action_concept_patterns)

                        # Normalize to unit vectors
                        action_patterns_norm = F.normalize(action_patterns, p=2, dim=1)

                        # Compute pairwise cosine similarity
                        similarity = torch.mm(action_patterns_norm, action_patterns_norm.t())

                        # Penalize high similarity (encourage diversity)
                        # Extract off-diagonal elements (don't penalize self-similarity)
                        mask = ~torch.eye(similarity.size(0), dtype=torch.bool, device=device)
                        diversity_loss = similarity[mask].abs().mean()

                        loss += gamma * diversity_loss
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
        print(f"\n[TRAIN] Epoch {epoch+1}:")
        print(f"  Recon Loss: {total_recon / len(train_loader):.4f}")
        if action_predictor is not None:
            print(f"  Action Loss: {total_action / len(train_loader):.4f}")
            print(f"  Action Acc: {100.0 * total_correct / total_samples:.2f}%")

            # Per-action accuracy (every 20 epochs)
            if (epoch + 1) % 20 == 0:
                with torch.no_grad():
                    all_preds = []
                    all_targets = []
                    for batch_data in train_loader:
                        batch_features = batch_data[0].to(device)
                        batch_actions = batch_data[1].to(device)
                        _, concepts = model(batch_features)
                        logits = action_predictor(concepts)
                        preds = logits.argmax(dim=-1)
                        all_preds.append(preds.cpu())
                        all_targets.append(batch_actions.cpu())

                    all_preds = torch.cat(all_preds)
                    all_targets = torch.cat(all_targets)

                    action_names = ['TurnLeft', 'TurnRight', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']
                    print(f"  Per-action accuracy (train):")
                    for a in range(config['n_actions']):
                        mask = all_targets == a
                        if mask.sum() > 0:
                            acc = (all_preds[mask] == a).float().mean().item()
                            count = mask.sum().item()
                            print(f"    {action_names[a]:10s}: {acc*100:5.1f}% ({count:5d} samples)")
        else:
            print(f"  Action Acc: (no action predictor)")

        # === VALIDATION ===
        model.eval()
        if action_predictor is not None:
            action_predictor.eval()
            if future_predictor is not None:
                future_predictor.eval()

        val_loss = 0
        val_recon = 0
        val_action = 0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for batch_features, batch_actions, batch_next_features, batch_next_actions, batch_mask in val_loader:
                batch_features = batch_features.to(device)
                batch_actions = batch_actions.to(device)
                batch_next_features = batch_next_features.to(device)
                batch_next_actions = batch_next_actions.to(device)
                batch_mask = batch_mask.to(device)

                # Forward
                x_recon, concepts = model(batch_features)

                # Reconstruction loss
                recon_loss = F.mse_loss(x_recon, batch_features)

                # Total loss
                loss = alpha * recon_loss

                # Temporal consistency loss
                if epsilon > 0:
                    _, next_concepts = model(batch_next_features)
                    temporal_loss = (batch_mask * (concepts - next_concepts).abs().sum(dim=1)).mean()
                    loss += epsilon * temporal_loss

                # Augmentation invariance loss
                if eta > 0:
                    noise = torch.randn_like(batch_features) * 0.01
                    aug_features = batch_features + noise
                    _, aug_concepts = model(aug_features)
                    aug_loss = (concepts - aug_concepts).abs().sum(dim=1).mean()
                    loss += eta * aug_loss

                # Optional action loss
                if action_predictor is not None:
                    action_logits = action_predictor(concepts)
                    action_loss = F.cross_entropy(action_logits, batch_actions)
                    loss += beta * action_loss

                    # L1 penalty
                    if delta > 0:
                        l1_penalty = action_predictor.weight.abs().mean()
                        loss += delta * l1_penalty

                    pred = action_logits.argmax(dim=-1)
                    val_correct += (pred == batch_actions).sum().item()
                    val_action += action_loss.item()

                    # Future prediction loss
                    if zeta > 0 and future_predictor is not None:
                        future_logits = future_predictor(concepts)
                        future_loss = (batch_mask.squeeze() * F.cross_entropy(
                            future_logits, batch_next_actions, reduction='none'
                        )).mean()
                        loss += zeta * future_loss

                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_samples += len(batch_actions)

        print(f"\n[VAL] Epoch {epoch+1}:")
        print(f"  Recon Loss: {val_recon / len(val_loader):.4f}")
        if action_predictor is not None:
            val_acc = 100.0 * val_correct / val_samples
            print(f"  Action Loss: {val_action / len(val_loader):.4f}")
            print(f"  Action Acc: {val_acc:.2f}%")

            # Model selection based on validation accuracy
            metric = val_acc
            if metric > best_val_acc:
                best_val_acc = metric
                best_val_loss = val_loss / len(val_loader)
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = {
                    'model': model.state_dict(),
                    'action_predictor': action_predictor.state_dict(),
                    'future_predictor': future_predictor.state_dict() if future_predictor else None,
                }
                print(f"  ✓ Best model (acc={best_val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"\n⚠ Early stopping at epoch {epoch+1}")
                    break
        else:
            # No action predictor - use validation loss
            val_loss_avg = val_loss / len(val_loader)
            print(f"  Action Acc: (no action predictor)")
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = {
                    'model': model.state_dict(),
                    'action_predictor': None,
                    'future_predictor': None,
                }
                print(f"  ✓ Best model (loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"\n⚠ Early stopping at epoch {epoch+1}")
                    break

        # Sparsity and interpretability analysis
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_features = train_features[:1000].to(device)
                sample_actions = train_actions[:1000].to(device)
                _, sample_concepts = model(sample_features)

                # Sparsity stats
                active_rate = (sample_concepts > 0).float().mean(dim=0)
                n_active = (active_rate > 0.01).sum().item()
                avg_sparsity = (sample_concepts > 0).float().mean().item()
                n_saturated = (active_rate > 0.9).sum().item()

                print(f"\n  Active concepts: {n_active}/{config['hidden_dim']}")
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

    # Restore best model
    if best_model_state is not None:
        print(f"\n{'='*60}")
        print(f"Restoring best model from epoch {best_epoch}")
        if action_predictor is not None:
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
        else:
            print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}")
        model.load_state_dict(best_model_state['model'])
        if action_predictor is not None and best_model_state['action_predictor'] is not None:
            action_predictor.load_state_dict(best_model_state['action_predictor'])
        if future_predictor is not None and best_model_state['future_predictor'] is not None:
            future_predictor.load_state_dict(best_model_state['future_predictor'])

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
    parser.add_argument('--gamma', type=float, default=0.0, help='Action-concept diversity loss weight')
    parser.add_argument('--delta', type=float, default=0.01, help='L1 sparsity penalty on action predictor weights')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Temporal consistency loss weight')
    parser.add_argument('--zeta', type=float, default=0.1, help='Future prediction loss weight')
    parser.add_argument('--eta', type=float, default=0.05, help='Augmentation invariance loss weight')
    parser.add_argument('--save_dir', type=str, default='./concept_models')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio (0.0-1.0)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')

    args = parser.parse_args()

    # Set seeds for data collection
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Collect data
    features, actions, next_features, next_actions, episode_mask = collect_rollout_data(
        args.model_path, args.env_name, args.n_episodes, args.seed
    )

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
        'gamma': args.gamma,
        'delta': args.delta,
        'epsilon': args.epsilon,
        'zeta': args.zeta,
        'eta': args.eta,
        'use_action_predictor': args.beta > 0,
        'env_name': args.env_name,
        'model_path': args.model_path,
        'seed': args.seed,
        'val_ratio': args.val_ratio,
        'patience': args.patience
    }

    # Train
    model, action_predictor = train_sae(features, actions, next_features, next_actions, episode_mask, config)

    # Analyze
    analyze_concepts(model, features)

    # Save
    save_model(model, action_predictor, features, actions, config, args.save_dir)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
