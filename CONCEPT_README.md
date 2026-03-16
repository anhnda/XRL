# Concept Extraction and Visualization Guide

This guide explains how to extract and visualize interpretable concepts from your trained PPO policy.

## Workflow

### 1. Train PPO Model
```bash
python training_minigrid.py
# Outputs: ppo_doorkey_5x5.zip
```

### 2. Extract Concepts with SAE
```bash
# Basic usage (256 concepts, k=25 sparsity)
python ConceptExtractor.py --model_path ppo_doorkey_5x5.zip

# Custom configuration
python ConceptExtractor.py \
    --model_path ppo_doorkey_5x5.zip \
    --hidden_dim 256 \
    --k 25 \
    --n_episodes 200 \
    --n_epochs 50 \
    --alpha 1.0 \
    --beta 1.0
```

**Key Parameters:**
- `--hidden_dim`: Number of concepts to learn (default: 256)
- `--k`: Top-K sparsity (default: 25 = ~10% active)
- `--alpha`: Reconstruction loss weight (default: 1.0)
- `--beta`: Action prediction loss weight (default: 1.0)
  - Increase `beta` for more action-relevant concepts
  - Decrease `beta` for more general perceptual concepts
- `--predictor_type`: 'linear' or 'mlp'

**Outputs:**
- `concept_models/concept_model.pt` - Full model
- `concept_models/sae.pt` - Standalone SAE
- `concept_models/sample_data.pt` - Sample features/actions

### 3. Visualize Concepts
```bash
# Full visualization (may take time due to observation rendering)
python VisualizeConceptsmd.py

# Quick visualization (skip observation rendering)
python VisualizeConceptsmd.py --skip_obs_viz

# Visualize specific concepts
python VisualizeConceptsmd.py --visualize_concepts "0,5,10,15,20"

# Custom paths
python VisualizeConceptsmd.py \
    --model_dir ./concept_models \
    --ppo_path ppo_doorkey_5x5.zip \
    --save_dir ./my_visualizations
```

**Generated Visualizations:**

1. **`activation_distributions.png`**
   - Histogram of concept activation rates
   - Distribution of activation values
   - Sparsity statistics

2. **`concept_action_correlation.png`**
   - Top concepts for each action
   - Shows which concepts predict which actions

3. **`action_predictor_weights.png`**
   - Heatmap of learned weights
   - Red = positive influence, Blue = negative influence

4. **`concept_summary.txt`**
   - Text summary of top 20 concepts
   - Activation statistics
   - Action correlations

5. **`concept_XXX_examples.png`** (if not using `--skip_obs_viz`)
   - Visual examples that maximally activate each concept
   - Shows what the concept "sees"

## Interpretation Tips

### Understanding Concepts

1. **High activation rate + strong max activation** = Important perceptual feature
   - Example: "Door visible in front", "Wall on left"

2. **Low activation rate + strong max activation** = Rare but critical concept
   - Example: "Holding key", "At goal"

3. **Strong correlation with specific action** = Behavioral concept
   - Example: If concept correlates with "Turn Left", it might detect "wall on right"

### Reading the Visualizations

**Action Predictor Weights:**
- Large positive weight: Concept encourages this action
- Large negative weight: Concept inhibits this action
- Near-zero weight: Concept irrelevant for this action

**Concept-Action Correlation:**
- High bars = concept is very active when this action is taken
- Helps identify: "Concept X is the 'forward movement' detector"

### Common Concept Patterns in MiniGrid

Expected concepts you might find:
- **Directional**: Wall-left, Wall-right, Wall-front, Empty-ahead
- **Object Detection**: Door-visible, Key-visible, Goal-visible
- **State**: Holding-key, Door-open, At-door
- **Strategic**: Path-blocked, Need-to-turn, Near-goal

## Advanced Usage

### Fine-tuning Hyperparameters

**If concepts are too general (not predicting actions well):**
```bash
python ConceptExtractor.py --beta 2.0  # Increase action loss weight
```

**If concepts are too action-specific (poor reconstruction):**
```bash
python ConceptExtractor.py --beta 0.5  # Decrease action loss weight
```

**If concepts are polysemantic (one concept = multiple meanings):**
```bash
python ConceptExtractor.py --hidden_dim 384  # Increase capacity
```

**If too many dead neurons:**
```bash
python ConceptExtractor.py --hidden_dim 192  # Decrease capacity
```

**If concepts are not sparse enough:**
```bash
python ConceptExtractor.py --k 15  # Reduce sparsity (fewer active concepts)
```

### Loading Models Programmatically

```python
import torch
from ConceptExtractor import ActionConceptModel

# Determine device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load full model with device mapping
checkpoint = torch.load('concept_models/concept_model.pt', map_location=device)
config = checkpoint['config']

model = ActionConceptModel(
    input_dim=128,
    hidden_dim=config['hidden_dim'],
    n_actions=7,
    k=config['k']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract concepts from new features
with torch.no_grad():
    x_recon, concepts = model.sae(features)

# Predict actions from concepts
action_logits = model.action_predictor(concepts)
```

### Exporting Concepts for Decision Trees

```python
# Binarize concepts for rule extraction
binary_concepts = (concepts > 0.1).float()  # Threshold at 0.1

# Now you can train decision trees:
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=5)
clf.fit(binary_concepts.numpy(), actions.numpy())

# Extract rules
from sklearn.tree import export_text
rules = export_text(clf, feature_names=[f'C{i}' for i in range(256)])
print(rules)
```

## Troubleshooting

**Issue: Poor action prediction accuracy (<70%)**
- Increase `--beta` to make concepts more action-relevant
- Increase `--n_episodes` to collect more diverse data
- Check if PPO model itself is well-trained

**Issue: Most concepts are dead (never activate)**
- Decrease `--hidden_dim` (too many concepts)
- Decrease `--k` (too sparse)
- Increase `--n_epochs` (undertrained)

**Issue: Concepts seem random/uninterpretable**
- This is a research challenge! Some concepts may be genuinely hard to interpret
- Try increasing `--beta` for more task-relevant concepts
- Look at multiple examples per concept
- Some concepts might be combinatorial (AND/OR of basic features)

## Citation

This implementation is based on:
- Sparse Autoencoder techniques from Anthropic's interpretability research
- Concept Bottleneck Models (Koh et al., 2020)
- MiniGrid environment (Chevalier-Boisvert et al., 2018)
