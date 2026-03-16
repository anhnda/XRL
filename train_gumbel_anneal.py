"""
Train with Gumbel-Softmax + Temperature Annealing
Start soft (tau=1.0), gradually decrease to hard (tau=0.1)
"""

import torch
from ConceptExtractor import collect_rollout_data, train_concept_model, save_model, analyze_concepts
import numpy as np

# Collect data
features, actions = collect_rollout_data(
    'ppo_doorkey_5x5.zip',
    'MiniGrid-DoorKey-5x5-v0',
    n_episodes=500,
    seed=42
)

# Training with annealing
config = {
    'input_dim': 128,
    'hidden_dim': 6,
    'n_actions': 7,
    'k': 3,
    'predictor_type': 'linear',
    'gumbel': True,
    'tau': 1.0,  # Will be annealed
    'n_epochs': 150,
    'batch_size': 256,
    'learning_rate': 1e-3,
    'alpha': 1.0,
    'beta': 2.0,
    'gamma': 0.5,
    'env_name': 'MiniGrid-DoorKey-5x5-v0',
    'model_path': 'ppo_doorkey_5x5.zip',
}

# Temperature annealing schedule
def get_tau(epoch, max_epochs):
    """Anneal temperature from 1.0 → 0.3"""
    tau_start = 1.0
    tau_end = 0.3
    return tau_start + (tau_end - tau_start) * (epoch / max_epochs)

# Custom training loop with annealing
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
from ConceptExtractor import ActionConceptModel

device = torch.device('cuda' if torch.cuda.is_available() else
                     'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Training on device: {device}")

model = ActionConceptModel(
    input_dim=config['input_dim'],
    hidden_dim=config['hidden_dim'],
    n_actions=config['n_actions'],
    k=config['k'],
    predictor_type=config['predictor_type'],
    gumbel=config['gumbel'],
    tau=config['tau']
).to(device)

dataset = TensorDataset(features, actions)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

alpha = config['alpha']
beta = config['beta']
gamma = config['gamma']

print(f"\nTraining with Temperature Annealing...")
print(f"Loss weights: α(recon)={alpha}, β(action)={beta}, γ(diversity)={gamma}")

for epoch in range(config['n_epochs']):
    # Update temperature
    tau = get_tau(epoch, config['n_epochs'])
    model.sae.tau = tau

    total_loss = 0
    total_recon_loss = 0
    total_action_loss = 0
    total_diversity_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['n_epochs']} (tau={tau:.3f})")
    for batch_features, batch_actions in pbar:
        batch_features = batch_features.to(device)
        batch_actions = batch_actions.to(device)

        # Forward pass
        x_recon, concepts, action_logits = model(batch_features)

        # Compute losses
        recon_loss = F.mse_loss(x_recon, batch_features)
        action_loss = F.cross_entropy(action_logits, batch_actions)

        # Diversity loss
        concept_usage = (concepts > 0).float().mean(dim=0)
        diversity_loss = -torch.std(concept_usage)

        # Multi-objective loss
        loss = alpha * recon_loss + beta * action_loss + gamma * diversity_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_action_loss += action_loss.item()
        total_diversity_loss += diversity_loss.item()

        # Accuracy
        pred_actions = action_logits.argmax(dim=-1)
        total_correct += (pred_actions == batch_actions).sum().item()
        total_samples += len(batch_actions)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'action': f'{action_loss.item():.4f}',
            'acc': f'{100.0 * total_correct / total_samples:.1f}%'
        })

    # Epoch summary
    accuracy = 100.0 * total_correct / total_samples
    print(f"Epoch {epoch+1} (tau={tau:.3f}):")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Recon Loss: {total_recon_loss / len(dataloader):.4f}")
    print(f"  Action Loss: {total_action_loss / len(dataloader):.4f}")

# Save model
save_model(model, features, actions, config, './concept_models')
analyze_concepts(model, features, actions, top_k=10)

print("\nTraining complete! Concepts are now binary at inference.")
