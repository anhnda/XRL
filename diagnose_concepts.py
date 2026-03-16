# diagnose_concepts.py
# Diagnostic script to check concept activation values

import os
import torch
from ConceptExtractor import ActionConceptModel

# Load model
model_dir = './concept_models'
checkpoint = torch.load(os.path.join(model_dir, 'concept_model.pt'),
                        map_location='cpu')
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
model.eval()

# Load sample data
sample_data = torch.load(os.path.join(model_dir, 'sample_data.pt'),
                         map_location='cpu')
features = sample_data['features'][:100]  # First 100 samples

print("Model configuration:")
print(f"  Gumbel mode: {model.sae.gumbel}")
print(f"  Temperature: {model.sae.tau}")
print(f"  K (sparsity): {model.sae.k}")
print(f"  Hidden dim: {model.sae.hidden_dim}")
print(f"  Model training mode: {model.training}")

print("\n" + "="*60)
print("DIAGNOSTIC: Checking intermediate values")
print("="*60)

with torch.no_grad():
    # Step 1: Raw encoder outputs (logits)
    raw_logits = model.sae.encoder(features)
    print(f"\n1. Raw encoder logits:")
    print(f"   Shape: {raw_logits.shape}")
    print(f"   Min: {raw_logits.min().item():.3f}")
    print(f"   Max: {raw_logits.max().item():.3f}")
    print(f"   Mean: {raw_logits.mean().item():.3f}")
    print(f"   Std: {raw_logits.std().item():.3f}")

    # Step 2: After Gumbel-Sigmoid
    if model.sae.gumbel:
        h = model.sae.gumbel_sigmoid(raw_logits, tau=model.sae.tau, hard=True)
        print(f"\n2. After Gumbel-Sigmoid (hard=True):")
        print(f"   Shape: {h.shape}")
        print(f"   Min: {h.min().item():.3f}")
        print(f"   Max: {h.max().item():.3f}")
        print(f"   Mean: {h.mean().item():.3f}")
        print(f"   Unique values: {torch.unique(h).tolist()[:10]}")  # Should be mostly 0 and 1

    # Step 3: After TopK sparsity
    concepts = model.sae.encode(features)
    print(f"\n3. Final concepts (after TopK):")
    print(f"   Shape: {concepts.shape}")
    print(f"   Min: {concepts.min().item():.3f}")
    print(f"   Max: {concepts.max().item():.3f}")
    print(f"   Mean: {concepts.mean().item():.3f}")
    print(f"   Unique values (first 20): {torch.unique(concepts).tolist()[:20]}")

    # Step 4: Check activation patterns
    active_mask = concepts > 0
    print(f"\n4. Activation statistics:")
    print(f"   % of non-zero values: {active_mask.float().mean().item()*100:.1f}%")
    print(f"   Expected if K={model.sae.k}: {100*model.sae.k/model.sae.hidden_dim:.1f}%")

    # Per-concept activation rate
    activation_rate = active_mask.float().mean(dim=0)
    print(f"\n5. Per-concept activation rates:")
    print(f"   Concepts with 100% activation: {(activation_rate == 1.0).sum().item()}")
    print(f"   Concepts with >50% activation: {(activation_rate > 0.5).sum().item()}")
    print(f"   Concepts with <1% activation: {(activation_rate < 0.01).sum().item()}")

    # Check specific Concept 2
    concept_2_activations = concepts[:, 2]
    print(f"\n6. Concept 2 specifically:")
    print(f"   Min: {concept_2_activations.min().item():.3f}")
    print(f"   Max: {concept_2_activations.max().item():.3f}")
    print(f"   Mean: {concept_2_activations.mean().item():.3f}")
    print(f"   Activation rate: {(concept_2_activations > 0).float().mean().item()*100:.1f}%")
    print(f"   Unique values: {torch.unique(concept_2_activations)[:10].tolist()}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
