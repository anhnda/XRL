"""
Simple test of logic layer with synthetic data
"""

import torch
import torch.nn.functional as F
from neural_logic_layer import LearnableNeuralLogicLayer

print("="*70)
print("TESTING LOGIC LAYER WITH SYNTHETIC DATA")
print("="*70)

# Create simple synthetic data with clear patterns
n_features = 10
n_actions = 3
n_samples = 100

# Rule:
# action 0: f0=1 and f1=1
# action 1: f2=1 and f3=0
# action 2: f4=1

features = torch.randint(0, 2, (n_samples, n_features)).float()
actions = torch.zeros(n_samples, dtype=torch.long)

for i in range(n_samples):
    f = features[i]
    if f[0] == 1 and f[1] == 1:
        actions[i] = 0
    elif f[2] == 1 and f[3] == 0:
        actions[i] = 1
    elif f[4] == 1:
        actions[i] = 2
    else:
        actions[i] = torch.randint(0, 3, (1,)).item()

print(f"\nData shape: {features.shape}")
print(f"Action distribution: {torch.bincount(actions)}")

# Create logic layer
model = LearnableNeuralLogicLayer(
    n_features=n_features,
    n_actions=n_actions,
    n_clauses_per_action=5,
    initial_temp=2.0,
    l0_penalty=1e-3
)

print("\n" + "="*70)
print("INITIAL FORWARD PASS")
print("="*70)

model.eval()
with torch.no_grad():
    logits = model(features[:10])
    print(f"Logits shape: {logits.shape}")
    print(f"Logits:\n{logits}")
    print(f"Predictions: {logits.argmax(1)}")
    print(f"True actions: {actions[:10]}")

# Train for a few iterations
print("\n" + "="*70)
print("TRAINING")
print("="*70)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(50):
    model.update_temperature(max(0.2, 2.0 * (0.95 ** epoch)))

    optimizer.zero_grad()

    logits = model(features)
    loss = F.cross_entropy(logits, actions) + model.complexity_penalty()

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(1) == actions).float().mean()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

# Final evaluation
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

model.eval()
with torch.no_grad():
    logits = model(features[:20])
    preds = logits.argmax(1)
    print(f"\nPredictions: {preds}")
    print(f"True actions: {actions[:20]}")
    print(f"Accuracy: {(preds == actions[:20]).float().mean():.3f}")

# Check what was learned
print("\n" + "="*70)
print("LEARNED CLAUSE WEIGHTS")
print("="*70)

clause_weights = model.get_clause_weights()
print(f"Clause weights shape: {clause_weights.shape}")

for a in range(n_actions):
    print(f"\nAction {a} clauses:")
    start = a * 5
    end = start + 5
    for i, clause_idx in enumerate(range(start, end)):
        clause = clause_weights[:, clause_idx]
        active = (clause.abs() > 0.1).nonzero().flatten()
        if len(active) > 0:
            literals = []
            for feat_idx in active:
                w = clause[feat_idx].item()
                if w > 0:
                    literals.append(f"f{feat_idx.item()}")
                else:
                    literals.append(f"¬f{feat_idx.item()}")
            print(f"  Clause {i}: {' ∧ '.join(literals)}")
