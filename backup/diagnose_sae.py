import torch
import os

# Load trained SAE
sae_data = torch.load('./concept_models/sae.pt', map_location='cpu')
config = sae_data['config']

print("="*60)
print("SAE DIAGNOSTIC")
print("="*60)

print(f"\nConfig:")
print(f"  hidden_dim: {config['hidden_dim']}")
print(f"  k: {config['k']}")
print(f"  tau: {config['tau']}")

# Load sample data
sample_data = torch.load('./concept_models/sample_data.pt', map_location='cpu')
features = sample_data['features']

print(f"\nFeature statistics:")
print(f"  Shape: {features.shape}")
print(f"  Min: {features.min().item():.4f}")
print(f"  Max: {features.max().item():.4f}")
print(f"  Mean: {features.mean().item():.4f}")
print(f"  Std: {features.std().item():.4f}")

# Reconstruct with SAE
from ConceptExtractor import SparseAutoencoder

sae = SparseAutoencoder(
    input_dim=config['input_dim'],
    hidden_dim=config['hidden_dim'],
    k=config['k'],
    tau=config['tau']
)
sae.encoder.weight.data = sae_data['encoder_weight']
sae.encoder.bias.data = sae_data['encoder_bias']
sae.decoder.weight.data = sae_data['decoder_weight']
sae.decoder.bias.data = sae_data['decoder_bias']
sae.eval()

with torch.no_grad():
    x_recon, concepts = sae(features)

print(f"\nConcept statistics:")
print(f"  Shape: {concepts.shape}")
print(f"  Unique values: {torch.unique(concepts)[:20].tolist()}")
print(f"  Min: {concepts.min().item():.4f}")
print(f"  Max: {concepts.max().item():.4f}")
print(f"  Mean: {concepts.mean().item():.4f}")

print(f"\nReconstruction statistics:")
print(f"  Recon Min: {x_recon.min().item():.4f}")
print(f"  Recon Max: {x_recon.max().item():.4f}")
print(f"  Recon Mean: {x_recon.mean().item():.4f}")
print(f"  Recon Std: {x_recon.std().item():.4f}")

mse = ((x_recon - features) ** 2).mean().item()
print(f"\n  MSE Loss: {mse:.4f}")
print(f"  RMSE: {mse**0.5:.4f}")

# Activation patterns
active_mask = concepts > 0
activation_rate = active_mask.float().mean(dim=0)

print(f"\nActivation patterns:")
print(f"  Concepts always active (>99%): {(activation_rate > 0.99).sum().item()}")
print(f"  Concepts often active (>50%): {(activation_rate > 0.5).sum().item()}")
print(f"  Concepts rarely active (<5%): {(activation_rate < 0.05).sum().item()}")
print(f"  Dead concepts (0%): {(activation_rate == 0).sum().item()}")

print(f"\nTop 10 concepts by activation rate:")
top_indices = activation_rate.argsort(descending=True)[:10]
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. Concept {idx.item():2d}: {activation_rate[idx].item()*100:5.1f}%")

print("\n" + "="*60)
