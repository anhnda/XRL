# check_saturated_concepts.py
# Check which concepts are always active (saturated)

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
features = sample_data['features']

print("Analyzing concept saturation...")
print(f"Total concepts: {config['hidden_dim']}")
print(f"Top-K sparsity: {config['k']}")
print(f"Expected activation rate: ~{100*config['k']/config['hidden_dim']:.1f}%")
print("="*60)

with torch.no_grad():
    _, concepts = model.sae(features)

    # Activation statistics
    activation_rate = (concepts > 0).float().mean(dim=0)
    mean_when_active = []

    for i in range(concepts.shape[1]):
        active_mask = concepts[:, i] > 0
        if active_mask.sum() > 0:
            mean_when_active.append(concepts[active_mask, i].mean().item())
        else:
            mean_when_active.append(0)

    # Find saturated concepts (always active)
    saturated = (activation_rate > 0.99).nonzero(as_tuple=True)[0]
    print(f"\nSATURATED CONCEPTS (>99% activation rate):")
    print(f"Count: {len(saturated)}/{config['hidden_dim']}")

    if len(saturated) > 0:
        print("\nDetails:")
        for idx in saturated:
            idx_val = idx.item()
            rate = activation_rate[idx].item()
            mean_val = mean_when_active[idx_val]
            print(f"  Concept {idx_val:3d}: {rate*100:5.1f}% active, mean={mean_val:.3f}")

    # Find frequently active concepts (>50%)
    frequent = (activation_rate > 0.5).nonzero(as_tuple=True)[0]
    print(f"\nFREQUENTLY ACTIVE CONCEPTS (>50% activation rate):")
    print(f"Count: {len(frequent)}/{config['hidden_dim']}")

    # Find rarely active concepts (<5%)
    rare = (activation_rate < 0.05).nonzero(as_tuple=True)[0]
    print(f"\nRARELY ACTIVE CONCEPTS (<5% activation rate):")
    print(f"Count: {len(rare)}/{config['hidden_dim']}")

    # Find dead concepts (never active)
    dead = (activation_rate == 0).nonzero(as_tuple=True)[0]
    print(f"\nDEAD CONCEPTS (0% activation rate):")
    print(f"Count: {len(dead)}/{config['hidden_dim']}")

    # Distribution summary
    print("\n" + "="*60)
    print("DISTRIBUTION SUMMARY:")
    print("="*60)
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.99, 1.0]
    labels = ['0%', '1-5%', '5-10%', '10-20%', '20-50%', '50-80%', '80-99%', '99-100%']

    for i in range(len(bins)-1):
        count = ((activation_rate >= bins[i]) & (activation_rate < bins[i+1])).sum().item()
        print(f"  {labels[i]:10s}: {count:3d} concepts")

    # Last bin is inclusive
    count = (activation_rate == 1.0).sum().item()
    print(f"  {labels[-1]:10s}: {count:3d} concepts")

    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)

    if len(saturated) > config['k'] * 0.3:
        print("⚠️  Too many saturated concepts!")
        print(f"   {len(saturated)} concepts are always active (>30% of K={config['k']})")
        print("\nSuggestions:")
        print("  1. Increase hidden_dim (try 128 or 256 instead of 64)")
        print("  2. Decrease beta (try 1.0-2.0 instead of 5.0)")
        print("  3. Increase alpha (try 0.5-1.0 instead of 0.3)")
        print("  4. Add L1 regularization on concepts during training")

    if len(dead) > config['hidden_dim'] * 0.3:
        print("⚠️  Too many dead concepts!")
        print(f"   {len(dead)} concepts never activate (>30% of hidden_dim)")
        print("\nSuggestions:")
        print("  1. Decrease hidden_dim")
        print("  2. Increase K (top-k sparsity)")
        print("  3. Increase gamma (diversity loss)")
