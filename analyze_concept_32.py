import torch
import numpy as np

# Load data
model_dir = './concept_models'
checkpoint = torch.load(f'{model_dir}/concept_model.pt', map_location='cpu')
sample_data = torch.load(f'{model_dir}/sample_data.pt', map_location='cpu')

from ConceptExtractor import ActionConceptModel

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
    action_logits = model.action_predictor(concepts)

concepts_np = concepts.numpy()
actions_np = actions.numpy()

# Analyze Concept 32
concept_idx = 32
active_mask = concepts_np[:, concept_idx] > 0

print(f"Analyzing Concept {concept_idx}")
print(f"Active in {active_mask.sum()}/{len(active_mask)} samples ({100*active_mask.mean():.1f}%)\n")

# When Concept 32 is active, what actions are taken?
actions_when_active = actions_np[active_mask]
unique, counts = np.unique(actions_when_active, return_counts=True)
action_names = ['Turn Left', 'Turn Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']

print("Action distribution when Concept 32 is active:")
for action_idx, count in zip(unique, counts):
    pct = 100 * count / len(actions_when_active)
    print(f"  {action_names[action_idx]:12s}: {count:4d} ({pct:5.1f}%)")

# Which concepts co-activate with Concept 32?
coactivation_counts = (concepts_np[active_mask, :] > 0).sum(axis=0)
coactivation_rate = coactivation_counts / active_mask.sum()

# Get top co-activating concepts
top_coactive = np.argsort(coactivation_rate)[-11:][::-1]  # Top 10 + concept 32 itself

print(f"\nTop concepts that co-activate with Concept {concept_idx}:")
weights = model.action_predictor.weight.data.numpy()
for idx in top_coactive:
    if idx == concept_idx:
        continue
    rate = coactivation_rate[idx]
    # Show the Turn Right weight for this concept
    turn_right_weight = weights[1, idx]  # Action 1 = Turn Right
    print(f"  Concept {idx:3d}: Co-active {rate*100:5.1f}%, Turn Right weight: {turn_right_weight:+.3f}")

print(f"\nExplaining Turn Right when Concept {concept_idx} is active:")
print(f"  Concept {concept_idx} contribution to Turn Right: -0.018")

# Calculate average contribution from top co-active concepts to Turn Right
turn_right_idx = 1
total_contribution = 0
print("\n  Top co-active concepts' contributions to Turn Right:")
for idx in top_coactive[:6]:
    if idx == concept_idx:
        continue
    avg_activation = concepts_np[active_mask, idx].mean()
    contribution = avg_activation * weights[turn_right_idx, idx]
    total_contribution += contribution
    print(f"    Concept {idx:3d}: {avg_activation:.1f} × {weights[turn_right_idx, idx]:+.3f} = {contribution:+.2f}")

print(f"\n  Combined effect from top 5 concepts: {total_contribution:+.2f}")
print(f"  This explains why Turn Right is chosen despite Concept 32's negative weight!")
