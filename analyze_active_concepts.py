import torch
import numpy as np
from ConceptExtractor import ActionConceptModel

# Load model
checkpoint = torch.load('./concept_models/concept_model.pt', map_location='cpu')
sample_data = torch.load('./concept_models/sample_data.pt', map_location='cpu')

config = checkpoint['config']
model = ActionConceptModel(
    input_dim=config['input_dim'],
    hidden_dim=config['hidden_dim'],
    n_actions=config['n_actions'],
    k=config['k'],
    predictor_type=config['predictor_type']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

features = sample_data['features']
actions = sample_data['actions']

# Get concepts
with torch.no_grad():
    _, concepts = model.sae(features)

concepts_np = concepts.numpy()
actions_np = actions.numpy()

# Identify active concepts
activation_rate = (concepts_np > 0).mean(axis=0)
active_concepts = np.where(activation_rate > 0.01)[0]

print(f"Active Concepts Analysis ({len(active_concepts)} concepts)")
print("=" * 80)

action_names = ['Turn Left', 'Turn Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']
weights = model.action_predictor.weight.data.numpy()

for concept_idx in active_concepts:
    print(f"\nConcept {concept_idx}:")
    print(f"  Activation rate: {activation_rate[concept_idx]*100:.1f}%")

    # Mean activation value when active
    active_mask = concepts_np[:, concept_idx] > 0
    if active_mask.sum() > 0:
        mean_val = concepts_np[active_mask, concept_idx].mean()
        max_val = concepts_np[active_mask, concept_idx].max()
        print(f"  Mean value (when active): {mean_val:.1f}")
        print(f"  Max value: {max_val:.1f}")

        # Action distribution when this concept is active
        actions_when_active = actions_np[active_mask]
        unique, counts = np.unique(actions_when_active, return_counts=True)
        print(f"  Action distribution when active:")
        for action_idx, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:3]:
            pct = 100 * count / len(actions_when_active)
            print(f"    {action_names[action_idx]:12s}: {pct:5.1f}%")

        # Top action influences
        concept_weights = weights[:, concept_idx]
        pos_idx = np.where(concept_weights > 0.01)[0]
        neg_idx = np.where(concept_weights < -0.01)[0]

        if len(pos_idx) > 0:
            print(f"  Encourages:")
            for idx in pos_idx[np.argsort(-concept_weights[pos_idx])[:2]]:
                print(f"    {action_names[idx]:12s}: {concept_weights[idx]:+.3f}")

        if len(neg_idx) > 0:
            print(f"  Inhibits:")
            for idx in neg_idx[np.argsort(concept_weights[neg_idx])[:2]]:
                print(f"    {action_names[idx]:12s}: {concept_weights[idx]:+.3f}")

# Concept co-activation matrix
print("\n" + "=" * 80)
print("Concept Co-activation Matrix")
print("=" * 80)
coactivation = np.zeros((len(active_concepts), len(active_concepts)))
for i, c1 in enumerate(active_concepts):
    for j, c2 in enumerate(active_concepts):
        mask1 = concepts_np[:, c1] > 0
        mask2 = concepts_np[:, c2] > 0
        coactivation[i, j] = (mask1 & mask2).sum() / mask1.sum() if mask1.sum() > 0 else 0

print("\nCo-activation rates (how often concepts appear together):")
print("Concept: " + " ".join([f"{c:3d}" for c in active_concepts]))
for i, c1 in enumerate(active_concepts):
    row_str = f"C{c1:2d}    : "
    for j in range(len(active_concepts)):
        if i == j:
            row_str += " -- "
        else:
            row_str += f"{coactivation[i,j]*100:3.0f} "
    print(row_str)

print("\nInterpretation:")
print("- 100% co-activation = concepts always appear together (redundant?)")
print("- Low co-activation = concepts represent different situations")
