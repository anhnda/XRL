# XRL — Explainable Reinforcement Learning with Logic Rules

Extract interpretable logic rules from a frozen PPO policy using Sparse Autoencoders and Product T-Norm Neural Logic.

Tested on `MiniGrid-DoorKey-5x5-v0` and `MiniGrid-DoorKey-6x6-v0`. Achieves **90%+ success rate** on 5x5 with fully human-readable DNF rules.

---

## Architecture

```
Raw observation (MiniGrid)
        │
        ▼
PPO CNN feature extractor  (frozen — never trained)
        │  (N, 128)
        ▼
Stage 1: SVD + ICA analysis
        │  signal subspace V_k, ICA directions, normalization stats
        ▼
Normalize  →  (x - mean) / std
        │  (N, 128)
        ▼
┌─────────────────────────────────────────────────┐
│  Stage 2A: SAE Pre-training (recon only)        │
│                                                 │
│  Sparse Autoencoder (SAE)                       │
│    encoder: Linear(128 → 300) + TopK(k=50)     │
│    decoder: Linear(300 → 128) [unit-norm cols]  │
│    loss: MSE reconstruction + L1 sparsity       │
│                                                 │
│  ► SAE frozen after convergence                 │
│  ► Activation stats computed (fixed buffers)    │
└─────────────────────────────────────────────────┘
        │  sparse z  (N, 300),  only k=50 active per sample
        ▼
Fixed Normalization  →  (z - z_mean) / z_std
        │  per-feature stats from non-zero activations (buffers, not learned)
        ▼
┌─────────────────────────────────────────────────┐
│  Stage 2B: Logic Training (SAE frozen)          │
│                                                 │
│  Sigmoid Bottleneck                             │
│    output_i = sigmoid(α_i · (z_i − β_i))       │
│    α, β learnable per feature                   │
│    bimodality loss pushes outputs toward {0, 1} │
│            │                                    │
│            ▼                                    │
│  Product T-Norm Logic Layer (DNF)               │
│    For each action:  OR(clause_1, ..., clause_n)│
│    Each clause:      AND over soft literals     │
│    literal = p·f + n·(1−f) + (1−p−n)·1         │
│    p, n from 3-way softmax → differentiable     │
└─────────────────────────────────────────────────┘
        │  (N, 7)  action logits
        ▼
argmax  →  action
```

The two-stage design prevents gradient wars between SAE reconstruction and action prediction. The logic layer always sees features from the same distribution — no moving targets.

**Loss terms:**

| Stage | Term | Purpose |
|---|---|---|
| 2A | `alpha_recon · MSE(x̂, x)` | SAE reconstruction |
| 2A | `lambda_sparsity · \|z_pre\|` | SAE L1 sparsity |
| 2B | `beta_action · CrossEntropy` | Action imitation (main objective) |
| 2B | `bimodal_weight · mean(z(1−z))` | Pushes bottleneck toward {0,1} |
| 2B | `l0_penalty · mean(p+n)` | Encourages sparse rules |

---

## Pipeline


DoorKey5x5

python feature_space_analysis.py \
    --model_path ppo_doorkey_5x5.zip \
    --env_name MiniGrid-DoorKey-5x5-v0 \
    --n_episodes 2000 \
    --save_dir ./stage1_outputs
python train_sae_logic.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --hidden_dim 300 --k 50 \
    --n_clauses_per_action 10 \
    --sae_pretrain_epochs 50 \
    --n_epochs 300 \
    --max_grad_norm 5.0
python check_success_rules.py \
    --model_path ./sae_logic_v3_outputs/sae_logic_v3_model.pt \
    --ppo_path ppo_doorkey_5x5.zip \
    --n_episodes 100 --print_rules
    
### Step 1 — Feature space analysis

Collects rollouts from the frozen PPO, runs SVD and ICA on the feature space, and saves normalization stats and ICA directions for SAE initialization.

```bash
python feature_space_analysis.py \
    --model_path ppo_doorkey_5x5.zip \
    --env_name   MiniGrid-DoorKey-5x5-v0 \
    --n_episodes 800 \
    --save_dir   ./stage1_outputs
```

**Key outputs:**

- `stage1_outputs/collected_data.pt` — raw features and action labels
- `stage1_outputs/stage1_outputs.pt` — SVD subspace, ICA directions, normalization stats
- `stage1_outputs/stage1_diagnostics.png` — singular value spectrum, ICA stability
- `stage1_outputs/action_distribution.png` — action frequency in collected data

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | Path to PPO `.zip` |
| `--env_name` | `MiniGrid-DoorKey-5x5-v0` | Gymnasium environment |
| `--n_episodes` | 800 | Episodes to collect |
| `--explore_eps` | 0.3 | Fraction of random steps for state diversity |
| `--save_dir` | `./stage1_outputs` | Output directory |

---

### Step 2 — Train SAE + logic rules

Trains the neuro-symbolic pipeline in two stages. First the SAE is pre-trained for reconstruction only, then it is frozen and the logic layer trains on the stable feature space.

```bash
python train_sae_logic_v3.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path   ./stage1_outputs/stage1_outputs.pt \
    --hidden_dim 300 --k 50 \
    --n_clauses_per_action 10 \
    --sae_pretrain_epochs 50 \
    --n_epochs 400 \
    --save_dir ./sae_logic_v3_outputs
```

**Key outputs:**

- `sae_logic_v3_outputs/sae_logic_v3_model.pt` — full model checkpoint
- `sae_logic_v3_outputs/learned_rules.json` — extracted DNF rules
- `sae_logic_v3_outputs/training_curves.png` — loss, accuracy, binarization plots

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--features_path` | required | Path to `collected_data.pt` |
| `--stage1_path` | `None` | Path to `stage1_outputs.pt` (recommended) |
| `--hidden_dim` | 300 | SAE hidden dimension |
| `--k` | 50 | TopK sparsity — active features per sample |
| `--n_clauses_per_action` | 10 | DNF clauses per action |
| `--sae_pretrain_epochs` | 50 | Epochs for SAE pre-training (Stage 2A) |
| `--n_epochs` | 400 | Total training epochs (Stage 2B = total − pretrain) |
| `--beta_action` | 5.0 | Action prediction loss weight |
| `--bimodal_warmup` | 30 | Epochs into Stage 2B before bimodality starts |
| `--bimodal_ramp` | 80 | Epochs to ramp bimodality to max |
| `--bimodal_max` | 0.3 | Maximum bimodality loss weight |
| `--l0_penalty` | 1e-4 | Rule sparsity penalty |
| `--lambda_sparsity` | 5e-3 | SAE L1 sparsity weight |
| `--max_grad_norm` | 5.0 | Gradient clipping norm |
| `--action_class_weights` | all 1.0 | Per-action loss weights (7 values) |
| `--sae_lr` | 1e-3 | SAE learning rate (Stage 2A) |
| `--logic_lr` | 3e-3 | Logic layer learning rate (Stage 2B) |
| `--bottleneck_lr` | 1e-3 | Bottleneck learning rate (Stage 2B) |
| `--save_dir` | `./sae_logic_v3_outputs` | Output directory |

**Boosting rare actions** — if an action appears rarely in training data, boost its weight so the logic layer learns rules for it. Example boosting Done (index 6):

```bash
--action_class_weights 1.0 1.0 1.0 1.0 1.0 1.0 10.0
```

---

### Step 3 — Evaluate by playing the game

Loads the trained model and plays MiniGrid episodes using the learned logic rules. Optionally compares against the original PPO baseline.

```bash
# Rules agent only
python check_success_rules.py \
    --model_path ./sae_logic_v3_outputs/sae_logic_v3_model.pt \
    --ppo_path   ppo_doorkey_5x5.zip \
    --n_episodes 100 \
    --print_rules

# Full comparison with failure diagnostics
python check_success_rules.py \
    --model_path ./sae_logic_v3_outputs/sae_logic_v3_model.pt \
    --ppo_path   ppo_doorkey_5x5.zip \
    --n_episodes 100 \
    --compare_ppo \
    --verbose_failures
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | Path to trained model checkpoint |
| `--ppo_path` | `ppo_doorkey_5x5.zip` | PPO model (feature extractor + baseline) |
| `--env_name` | `MiniGrid-DoorKey-5x5-v0` | Environment to evaluate on |
| `--n_episodes` | 100 | Number of evaluation episodes |
| `--max_steps` | 500 | Max steps per episode before forced termination |
| `--compare_ppo` | off | Also run PPO baseline for comparison |
| `--print_rules` | off | Print learned DNF rules before evaluation |
| `--verbose_failures` | off | Print last 10 actions of failed episodes |
| `--render` | off | Render episodes visually |

---

## Example results

```
RULES AGENT — EVALUATION RESULTS
============================================================
  Episodes     : 100
  Successes    : 90/100
  Success rate : 90.00%
  Avg reward   : 0.8657 ± 0.2886
  Avg length   : 34.5 ± 71.8

  Action usage:
    TurnRight :    403 ( 11.7%)
    Forward   :    409 ( 11.8%)
    Pickup    :    100 (  2.9%)
    Toggle    :   2541 ( 73.6%)
```

**Example extracted rules (DNF form):**

```
Forward ←
    (¬f_67 ∧ ¬f_109 ∧ ¬f_184 ∧ ¬f_206) [bias=3.86]

Toggle ←
    (f_68 ∧ f_87 ∧ ¬f_89 ∧ ¬f_109 ∧ f_206 ∧ f_230) [bias=3.96]

Pickup ←
    (f_56 ∧ f_67 ∧ f_87 ∧ f_89 ∧ ¬f_109 ∧ f_172 ∧ f_183 ∧ f_184 ∧ f_230) [bias=5.08]

TurnRight ←
    (f_4 ∧ f_10 ∧ f_28 ∧ f_32 ∧ f_36 ∧ f_44 ∧ f_56 ∧ f_68 ∧ ¬f_71 ∧ f_109 ∧ f_118
     ∧ f_144 ∧ f_145 ∧ f_173 ∧ f_204 ∧ f_206 ∧ f_247 ∧ f_248)
```

Each rule is a conjunction of SAE feature activations. Positive literals (`f_i`) mean the feature must be active; negative literals (`¬f_i`) mean it must be inactive.

---

## Design notes

**Why two-stage training?** Joint training of SAE + logic causes gradient wars: the SAE keeps reshaping its features to minimize reconstruction, which destroys the feature mapping the logic layer has learned. Accuracy peaks early (~93% at epoch 30) then collapses to ~70%. Freezing the SAE first gives the logic layer a stable target.

**Why fixed normalization instead of a learned scaler?** SAE activations can be very large (mean ~180, max ~1500) due to top-k sparsity. The sigmoid bottleneck needs inputs in the [-3, 3] range to be discriminative. A learned scaler couples back to the SAE via gradients, recreating the instability problem. Fixed per-feature `(z - mean) / std` computed once from converged SAE activations is simple and stable.

**Linear probe diagnostic.** After training, a linear probe is fitted on bottleneck features. If it achieves high accuracy (>70%), the SAE+bottleneck preserves enough action-relevant information and the logic layer expressiveness is the limiting factor. If it's low, the SAE lost critical information.

---

## File structure

```
XRL/
├── feature_space_analysis.py      # Step 1: SVD, ICA, data collection
├── train_sae_logic_v3.py          # Step 2: two-stage SAE + logic training
├── check_success_rules.py         # Step 3: game evaluation
├── sparse_concept_autoencoder.py  # SAE implementation
├── ppo_doorkey_5x5.zip            # Pre-trained PPO policy
├── stage1_outputs/
│   ├── collected_data.pt          # Raw features + action labels
│   ├── stage1_outputs.pt          # SVD/ICA/normalization results
│   ├── stage1_diagnostics.png
│   └── action_distribution.png
└── sae_logic_v3_outputs/
    ├── sae_logic_v3_model.pt      # Trained model checkpoint
    ├── learned_rules.json         # Extracted DNF rules
    └── training_curves.png        # Training diagnostics
```

---

## Dependencies

```bash
pip install torch gymnasium minigrid stable-baselines3
pip install scikit-learn scipy matplotlib seaborn
```