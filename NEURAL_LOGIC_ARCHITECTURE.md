# Neural Logic Network Architecture

## Overview

This implementation adds a **learnable neural logic network** to the XRL codebase, providing fully interpretable decision rules on top of SAE features.

```
Input → Encoder → SAE Features → Binarization → Logic Rules → Actions
                      ↓                              ↓
                 Interpretable                 Interpretable
                  (concepts)                     (rules)
```

## Key Components

### 1. Sparse Autoencoder (SAE)
- **File**: `sparse_concept_autoencoder.py` (existing)
- **Purpose**: Extract sparse, interpretable features from neural network activations
- **Output**: Sparse feature vectors (e.g., 256-dim with top-k=10 active)

### 2. Neural Logic Layer
- **File**: `neural_logic_layer.py` (new)
- **Purpose**: Learn interpretable logic rules over SAE features
- **Architecture**: Disjunctive Normal Form (DNF)
  ```
  action_i ← (c1 ∧ c2 ∧ ¬c3) ∨ (c5 ∧ ¬c7) ∨ ...
  ```

### 3. Training Script
- **File**: `train_sae_logic.py` (new)
- **Purpose**: Train SAE + Logic Layer together or separately
- **Modes**:
  - **Joint**: Train both together (end-to-end)
  - **Two-stage**: Pre-train SAE, then train logic layer

### 4. Visualization Tools
- **File**: `visualize_logic_rules.py` (new)
- **Purpose**: Analyze and visualize learned rules
- **Outputs**: Rule structure, statistics, fidelity metrics

## How It Works

### Architecture Details

#### 1. Feature Binarization
SAE produces continuous sparse features. We binarize them for logic operations:

```python
# Method 1: Threshold
binary_features = (sae_features > threshold).float()

# Method 2: Top-k
top_k_indices = topk(sae_features, k)
binary_features = one_hot(top_k_indices)
```

#### 2. Logic Layer Structure

**Learnable Components:**
- **Clause weights**: `W ∈ {-1, 0, 1}^{n_features × n_clauses}`
  - `+1`: feature must be active (positive literal)
  - `-1`: feature must be inactive (negative literal)
  - `0`: feature not used in this clause

- **Feature mask**: Which features participate in each clause (L0 regularization)

**Inference:**
```python
# Each clause is a conjunction (AND)
clause_j = f1 ∧ f2 ∧ ¬f3

# Each action is a disjunction of clauses (OR)
action_i = clause_1 ∨ clause_2 ∨ clause_3
```

#### 3. Differentiable Training

**Challenge**: Logic operations (AND, OR, NOT) are discrete and non-differentiable.

**Solution**: Use soft relaxations during training:

```python
# Soft AND (product t-norm)
def soft_and(a, b):
    return a * b

# Soft OR (probabilistic sum)
def soft_or(a, b):
    return a + b - a * b

# NOT
def soft_not(a):
    return 1 - a
```

**Temperature Annealing**: Gradually transition from soft to hard logic
```python
temperature = max(min_temp, initial_temp * (decay ** epoch))
```

#### 4. Sparsity Regularization

**L0 Regularization** encourages simple rules:
```python
# Penalize number of features used
mask_prob = sigmoid(mask_logits)
l0_penalty = λ * sum(mask_prob)
```

## Training Strategies

### Joint Training (Recommended)

Train SAE and logic layer together for best performance:

```python
loss = α * reconstruction_loss +    # SAE quality
       β * action_loss +            # Task performance
       λ1 * sae_sparsity +          # SAE sparsity
       λ2 * logic_complexity        # Rule simplicity
```

**Advantages:**
- End-to-end optimization
- Better task performance
- Features optimized for logical reasoning

**Schedule:**
1. Epochs 0-50: Train both SAE and logic
2. Epochs 50+: Freeze SAE, fine-tune logic only

### Two-Stage Training

Pre-train SAE, then train logic layer:

```python
# Stage 1: Pre-train SAE
for epoch in epochs_1:
    loss = reconstruction_loss + sparsity_loss
    update(sae_params)

# Stage 2: Train logic (freeze SAE)
freeze(sae_params)
for epoch in epochs_2:
    loss = action_loss + logic_complexity
    update(logic_params)
```

**Advantages:**
- More stable
- SAE features are task-agnostic
- Can reuse pre-trained SAE

## Interpretability

### Rule Extraction

The learned rules are fully interpretable:

```python
rules = model.extract_rules(
    feature_names=["door_visible", "has_key", "goal_near", ...],
    action_names=["TurnLeft", "TurnRight", "Forward", ...]
)
```

**Example Output:**
```
TurnRight ←
    (obstacle_ahead ∧ ¬wall_right) ∨
    (goal_right ∧ ¬door_locked)

Forward ←
    (¬obstacle_ahead ∧ goal_visible) ∨
    (path_clear)

Pickup ←
    (object_visible ∧ ¬carrying_object ∧ object_near)
```

### Metrics

**Rule Statistics:**
- Number of clauses per action
- Average literals per clause
- Sparsity (% of unused features)
- Positive vs. negative literal ratio

**Fidelity:**
- Accuracy on original task
- Per-action accuracy
- Coverage (% samples matching at least one rule)

## Comparison: Neural Logic vs. CNLNet

| Aspect | CNLNet (paper) | Our Implementation |
|--------|----------------|-------------------|
| **Weights** | Confidence values (real) | Binary structure only |
| **Reasoning** | Weighted sum of formulas | Hard/soft logic (temp-based) |
| **Output** | Weighted satisfaction | OR of AND clauses (DNF) |
| **Interpretability** | Confidence = reliability | Pure logical rules |
| **Simplicity** | More complex | Simpler, cleaner |

## Usage Example

### 1. Train the Model

```bash
# Joint training (recommended)
python train_sae_logic.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --mode joint \
    --hidden_dim 256 \
    --k 10 \
    --n_clauses_per_action 10 \
    --n_epochs 200 \
    --save_dir ./sae_logic_outputs
```

### 2. Visualize Results

```bash
python visualize_logic_rules.py \
    --model_path ./sae_logic_outputs/sae_logic_model.pt \
    --features_path ./stage1_outputs/collected_data.pt \
    --save_dir ./rule_visualizations
```

### 3. Use in Code

```python
from train_sae_logic import SAELogicAgent

# Load model
checkpoint = torch.load("sae_logic_model.pt")
model = SAELogicAgent(config, device="cuda")
model.load_state_dict(checkpoint['model_state'])

# Get predictions
features = extract_features(observation)
action_logits = model(features)
action = action_logits.argmax()

# Extract interpretable rules
rules = model.extract_rules(
    feature_names=concept_labels,
    action_names=action_names
)
```

## Key Hyperparameters

### SAE Parameters
- `hidden_dim`: SAE feature dimension (overcomplete, e.g., 256-512)
- `k`: Top-k sparsity (e.g., 6-10)

### Logic Parameters
- `n_clauses_per_action`: Number of clauses per action (e.g., 5-15)
- `initial_temp`: Starting temperature (e.g., 5.0)
- `min_temp`: Final temperature (e.g., 0.1)
- `temp_decay`: Decay rate (e.g., 0.98)

### Loss Weights
- `alpha`: SAE reconstruction weight (e.g., 1.0)
- `beta`: Action prediction weight (e.g., 2.0)
- `lambda1`: SAE sparsity (e.g., 5e-3)
- `lambda2`: Logic complexity (e.g., 1e-4)

## Files Created

```
XRL/
├── neural_logic_layer.py          # Core logic network implementation
├── train_sae_logic.py             # Training script (SAE + Logic)
├── visualize_logic_rules.py       # Visualization and analysis
├── example_logic_network.py       # Simple toy example
└── NEURAL_LOGIC_ARCHITECTURE.md   # This documentation
```

## Expected Results

### Performance
- **Accuracy**: 80-95% (depending on task complexity)
- **Rule Fidelity**: 90-98% (how well rules match learned policy)
- **Sparsity**: 3-8 literals per clause on average

### Interpretability
- **Clear rules**: Human-readable logical formulas
- **Sparse**: Few features per rule
- **Verifiable**: Can manually check if rules make sense

## Next Steps

1. **Run toy example**: `python example_logic_network.py`
2. **Train on MiniGrid**: Use existing stage 1 features
3. **Analyze rules**: Use visualization script
4. **Compare with original**: Evaluate fidelity vs. original policy

## References

- **CNLNet Paper**: "Deep Confidence Neural Logic Networks" (CNLNet.pdf)
- **SAE**: Sparse Autoencoders for interpretable features
- **Logic Programming**: DNF (Disjunctive Normal Form) representation
- **Differentiable Logic**: Temperature-based soft relaxations
