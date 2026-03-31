"""
Debug script to investigate why logic network accuracy is 0%
"""

import torch
from train_sae_logic import SAELogicAgent, SAELogicConfig
from neural_logic_layer import binarize_sae_features

# Load data
data = torch.load('./stage1_outputs/collected_data.pt', weights_only=False)
features = data['features'][:100]  # Just first 100 samples
actions = data['actions'][:100]

print("Data shapes:")
print(f"  Features: {features.shape}")
print(f"  Actions: {actions.shape}")
print(f"  Action distribution: {torch.bincount(actions)}")

# Create config
config = SAELogicConfig(
    input_dim=features.shape[1],
    hidden_dim=256,
    k=10,
    n_actions=actions.max().item() + 1,
    n_clauses_per_action=10
)

# Create model
model = SAELogicAgent(config, device='cpu')
model.eval()

# Test forward pass
print("\n" + "="*70)
print("TESTING FORWARD PASS")
print("="*70)

with torch.no_grad():
    # 1. SAE encoding
    batch = features[:10]
    z_sparse, z_pre = model.sae.encode(batch)

    print(f"\n1. SAE Features:")
    print(f"   z_sparse shape: {z_sparse.shape}")
    print(f"   z_sparse range: [{z_sparse.min():.3f}, {z_sparse.max():.3f}]")
    print(f"   z_sparse sparsity: {(z_sparse > 0).float().mean():.3f}")
    print(f"   z_sparse mean: {z_sparse.mean():.3f}")

    # 2. Binarization
    binary_features = binarize_sae_features(
        z_sparse,
        method=config.binarization_method,
        threshold=config.binarization_threshold,
        top_k=config.binarization_topk
    )

    print(f"\n2. Binary Features:")
    print(f"   binary shape: {binary_features.shape}")
    print(f"   binary mean: {binary_features.mean():.3f}")
    print(f"   binary sum per sample: {binary_features.sum(dim=1)[:5]}")
    print(f"   binary example: {binary_features[0][:20]}")

    # 3. Logic layer
    action_logits = model.logic_layer(binary_features)

    print(f"\n3. Logic Layer Output:")
    print(f"   logits shape: {action_logits.shape}")
    print(f"   logits range: [{action_logits.min():.3f}, {action_logits.max():.3f}]")
    print(f"   logits mean: {action_logits.mean():.3f}")
    print(f"   logits std: {action_logits.std():.3f}")
    print(f"   logits[0]: {action_logits[0]}")

    # 4. Predictions
    preds = action_logits.argmax(1)
    print(f"\n4. Predictions:")
    print(f"   predictions: {preds}")
    print(f"   true actions: {actions[:10]}")
    print(f"   accuracy: {(preds == actions[:10]).float().mean():.3f}")

# Check clause weights
print("\n" + "="*70)
print("CLAUSE WEIGHTS ANALYSIS")
print("="*70)

clause_weights = model.logic_layer.get_clause_weights()
print(f"Clause weights shape: {clause_weights.shape}")
print(f"Clause weights range: [{clause_weights.min():.3f}, {clause_weights.max():.3f}]")
print(f"Clause weights mean: {clause_weights.mean():.3f}")
print(f"Non-zero weights: {(clause_weights.abs() > 0.01).sum().item()} / {clause_weights.numel()}")

# Check mask probabilities
mask_probs = torch.sigmoid(model.logic_layer.mask_logits)
print(f"\nMask probabilities:")
print(f"  Range: [{mask_probs.min():.3f}, {mask_probs.max():.3f}]")
print(f"  Mean: {mask_probs.mean():.3f}")
print(f"  Active (>0.5): {(mask_probs > 0.5).sum().item()} / {mask_probs.numel()}")

# Test soft clause evaluation
print("\n" + "="*70)
print("TESTING SOFT CLAUSE EVALUATION")
print("="*70)

model.logic_layer.train()  # Enable soft logic
model.logic_layer.update_temperature(1.0)

with torch.no_grad():
    z_sparse, _ = model.sae.encode(features[:5])
    binary_features = binarize_sae_features(z_sparse, method='threshold', threshold=0.1)

    # Manually evaluate first clause
    clause_weights = model.logic_layer.get_clause_weights()
    clause_0 = clause_weights[:, 0]

    print(f"Clause 0 weights:")
    print(f"  Non-zero count: {(clause_0.abs() > 0.01).sum()}")
    print(f"  Positive literals: {(clause_0 > 0.01).sum()}")
    print(f"  Negative literals: {(clause_0 < -0.01).sum()}")

    # Evaluate it
    sat = model.logic_layer.evaluate_clause_soft(binary_features, clause_0)
    print(f"\nClause 0 satisfaction:")
    print(f"  Values: {sat}")
    print(f"  Mean: {sat.mean():.3f}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Check if issue is in binarization
if binary_features.sum() == 0:
    print("❌ ISSUE: Binary features are all zeros!")
    print("   → SAE features are too small or binarization threshold too high")
elif (mask_probs > 0.5).sum() == 0:
    print("❌ ISSUE: No features are being used (all masks < 0.5)")
    print("   → Mask initialization problem")
elif action_logits.std() < 0.01:
    print("❌ ISSUE: Action logits have no variance")
    print("   → Logic layer producing constant outputs")
    print(f"   → All logits ≈ {action_logits.mean():.3f}")
else:
    print("✓ Data flow looks reasonable")
    print("  Issue might be in training dynamics or gradient flow")
