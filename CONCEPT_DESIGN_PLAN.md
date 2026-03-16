# Concept Extraction Design Analysis & Improvement Plan

## Current Problems

### 1. **Too Many Concepts (256)**
- For a simple 5x5 MiniGrid, 256 concepts is excessive
- Hard to interpret and understand
- Many dead/redundant concepts
- Human can't reason about 256 features

### 2. **Too Many Active Concepts (TopK=25)**
- 25 simultaneous active concepts = not sparse enough
- Each action decision uses 25 concepts → hard to interpret
- "Why did agent turn right?" requires understanding 25 concepts together
- Defeats the purpose of sparse interpretability

### 3. **Dense Action Predictor**
- Linear layer: 256 concepts × 7 actions = 1,792 weights
- Every concept influences every action (dense connections)
- Weights are all small (±0.06 range)
- Distributed representations → NOT interpretable

### 4. **Overlapping Concepts**
- Many concepts co-activate 100% together (see Concept 32 analysis)
- Concepts 35, 93, 172, 191, 225 always fire together
- Redundancy → wasted capacity

### 5. **No Semantic Grounding**
- Concepts learned purely from reconstruction + action prediction
- No connection to MiniGrid semantics (door, key, walls, agent direction)
- Hard to name concepts ("Concept 32" vs "See door ahead")

---

## Design Principles for Interpretable Concepts

### What Makes a Good Concept?

1. **Sparse**: Only 3-7 concepts active at once (not 25!)
2. **Semantic**: Aligned with human-understandable features
3. **Disentangled**: Each concept represents ONE thing
4. **Actionable**: Clear connection between concept → action
5. **Few Total**: 20-50 concepts max for simple games

---

## Proposed Improvements

### Strategy 1: **Reduce Concept Count**
```
Current: 256 concepts, TopK=25 (9.8% sparsity)
Proposed: 64 concepts, TopK=5 (7.8% sparsity)
Or even: 32 concepts, TopK=3 (9.4% sparsity)
```

**Benefits:**
- Easier to analyze and interpret
- Fewer dead concepts
- Forced to learn more meaningful features

**Implementation:**
```python
model = ActionConceptModel(
    input_dim=128,
    hidden_dim=32,  # Down from 256
    k=3,           # Down from 25
    ...
)
```

---

### Strategy 2: **Stronger Sparsity Constraints**

**Option A: Binary Concepts**
- Force concepts to be 0 or 1 (present/absent)
- Use Gumbel-Softmax or straight-through estimator
```python
def binary_concepts(logits, tau=0.5):
    probs = torch.sigmoid(logits / tau)
    # Straight-through: hard in forward, soft in backward
    hard = (probs > 0.5).float()
    return hard - probs.detach() + probs
```

**Option B: Stricter TopK with Diversity Loss**
- Keep TopK but add diversity regularization
- Penalize concepts that always co-activate
```python
# Diversity loss: encourage different concepts per sample
diversity_loss = -torch.std(concepts.mean(dim=0))
```

**Option C: L1 Sparsity (Classic SAE)**
- Replace TopK with L1 penalty on activations
```python
sparsity_loss = lambda_sparse * torch.abs(concepts).sum()
```

---

### Strategy 3: **Sparse Action Predictor**

**Current Problem:**
- Dense 256→7 linear layer
- Every concept votes on every action

**Solution A: Top-K Action Predictor**
- Each action uses only top-5 most relevant concepts
- Zero out other weights during inference
```python
# Sparsify action predictor weights
mask = torch.zeros_like(weights)
for action_idx in range(n_actions):
    top_concepts = weights[action_idx].abs().topk(5).indices
    mask[action_idx, top_concepts] = 1
sparse_weights = weights * mask
```

**Solution B: Structured Sparsity**
- Group concepts by "type" (spatial, object, direction)
- Each action only uses certain groups
```python
# Example structure:
# Concepts 0-15: Spatial (where is agent?)
# Concepts 16-31: Objects (what's visible?)
# Concepts 32-47: Direction (which way facing?)

# Turn actions only use direction concepts
# Pickup only uses object concepts
```

**Solution C: Separate Predictors per Action**
- Instead of one big linear layer, 7 small independent predictors
```python
self.action_predictors = nn.ModuleList([
    nn.Linear(hidden_dim, 1) for _ in range(n_actions)
])
# Add L1 regularization to make each predictor sparse
```

---

### Strategy 4: **Semantic Grounding via Auxiliary Tasks**

**Add MiniGrid-specific supervision:**

```python
# Extract ground-truth features from MiniGrid env
def get_semantic_labels(obs, env):
    labels = {
        'has_key': env.carrying is not None,
        'door_visible': check_door_in_view(obs),
        'door_open': env.door.is_open if door_visible else None,
        'wall_ahead': check_wall_ahead(obs, env.agent_dir),
        'wall_left': check_wall_left(obs, env.agent_dir),
        'wall_right': check_wall_right(obs, env.agent_dir),
        'agent_direction': env.agent_dir,  # 0-3
        'distance_to_key': compute_distance_to_key(env),
        'distance_to_door': compute_distance_to_door(env),
    }
    return labels

# Multi-task loss
loss = (alpha * recon_loss +
        beta * action_loss +
        gamma * semantic_loss)
```

**Benefits:**
- Concepts align with meaningful game features
- Can label concepts: "Concept 5 = has key"
- Better interpretability

---

### Strategy 5: **Hierarchical Concepts**

**Idea:** Learn concepts at different abstraction levels

```python
class HierarchicalSAE(nn.Module):
    def __init__(self):
        # Level 1: Low-level features (spatial)
        self.sae_l1 = SparseAutoencoder(input_dim=128, hidden_dim=16, k=2)

        # Level 2: Mid-level features (objects)
        self.sae_l2 = SparseAutoencoder(input_dim=16, hidden_dim=12, k=2)

        # Level 3: High-level features (intentions)
        self.sae_l3 = SparseAutoencoder(input_dim=12, hidden_dim=8, k=2)

    def forward(self, x):
        # Encode hierarchically
        c1 = self.sae_l1.encode(x)
        c2 = self.sae_l2.encode(c1)
        c3 = self.sae_l3.encode(c2)

        # Decode hierarchically
        x_recon = self.sae_l1.decode(
            self.sae_l2.decode(
                self.sae_l3.decode(c3)))

        return x_recon, (c1, c2, c3)
```

**Total active:** 2+2+2 = 6 concepts (highly sparse!)

---

### Strategy 6: **Concept Pruning & Merging**

**After training, post-process to improve interpretability:**

1. **Remove dead concepts** (activation rate < 1%)
2. **Merge redundant concepts** (correlation > 0.95)
3. **Reorder by importance**
4. **Retrain action predictor** with reduced concepts

```python
def prune_concepts(model, features, threshold=0.01):
    with torch.no_grad():
        _, concepts = model.sae(features)
        activation_rate = (concepts > 0).float().mean(dim=0)

    # Keep only active concepts
    active_mask = activation_rate > threshold
    n_active = active_mask.sum().item()

    print(f"Keeping {n_active}/{len(activation_rate)} concepts")

    # Create new pruned model
    # ... (implementation)
```

---

## Recommended Approach for MiniGrid 5x5

### Phase 1: Quick Fix (Minimal Changes)
```python
# In ConceptExtractor.py, change defaults:
parser.add_argument('--hidden_dim', type=int, default=64)   # Was 256
parser.add_argument('--k', type=int, default=5)             # Was 25
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=2.0)      # Emphasize action prediction
```

**Retrain and check if concepts improve**

---

### Phase 2: Add Semantic Grounding
1. Extract MiniGrid state features (door visible, has key, etc.)
2. Add auxiliary loss to predict these features
3. Analyze which concepts align with which features

---

### Phase 3: Sparse Action Predictor
1. Add L1 regularization to action predictor
2. Or use TopK masking during inference
3. Report which concepts are used for each action

---

### Phase 4: Interactive Concept Labeling
1. Visualize top activating states for each concept
2. Manually label concepts based on patterns
3. Validate labels with held-out data

---

## Questions to Answer

1. **How many concepts do we really need for DoorKey-5x5?**
   - Expected: ~20-40 concepts
   - Objects: key, door, wall, goal (4)
   - Spatial: ahead, left, right, behind (4)
   - States: has_key, door_open (2)
   - Directions: facing N/S/E/W (4)
   - Distances: near/far (2)
   - Total: ~16 semantic concepts → use 32 with redundancy

2. **What's the right sparsity level?**
   - Test: k=3, 5, 7, 10
   - Measure: Reconstruction quality vs interpretability
   - Goal: Minimal k that preserves action accuracy

3. **Should we use binary or continuous concepts?**
   - Binary: Easier to interpret ("has key" = 1 or 0)
   - Continuous: More expressive but harder to understand
   - Hybrid: Binary activation + continuous strength?

4. **How to validate interpretability?**
   - Human evaluation: Can you explain agent behavior?
   - Intervention: Manually set concepts, check actions
   - Counterfactual: What if Concept X was different?

---

## Next Steps

1. **Decide on architecture changes** (see recommendations above)
2. **Retrain with new hyperparameters**
3. **Analyze concepts** (visualization + metrics)
4. **Iterate** based on interpretability

What would you like to try first?
