# XRL — Explainable Reinforcement Learning with Logic Rules

Extract interpretable logic rules from a frozen PPO policy using Sparse Autoencoders and Product T-Norm Neural Logic.

Tested on `MiniGrid-DoorKey-5x5-v0`. Achieves **90%+ success rate** with fully human-readable DNF rules.

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
Sparse Autoencoder (SAE)
  encoder: Linear(128 → 256) + TopK(k=10)
  decoder: Linear(256 → 128)  [unit-norm columns]
        │  sparse z  (N, 256),  only k=10 active per sample
        ▼
Sigmoid Bottleneck
  output_i = sigmoid(α_i · (z_i − β_i))
  α, β learnable per feature
  bimodality loss pushes outputs toward {0, 1}
        │  (N, 256)  ≈ binary
        ▼
Product T-Norm Logic Layer  (DNF)
  For each action:  OR(clause_1, ..., clause_n)
  Each clause:      AND over soft literals
  literal_ji = p_ji·f_i + n_ji·(1−f_i) + (1−p_ji−n_ji)·1
  p, n from 3-way softmax  →  differentiable rule selection
        │  (N, 7)  action logits
        ▼
argmax  →  action
```

**Loss terms during training:**

| Term | Purpose |
|---|---|
| `alpha_recon · MSE(x̂, x)` | SAE reconstruction — keeps features meaningful |
| `beta_action · CrossEntropy` | Action imitation — main objective |
| `lambda_sparsity · \|z_pre\|` | SAE L1 — encourages sparse activations |
| `bimodal_weight · mean(z(1−z))` | Pushes bottleneck toward {0,1} |
| `l0_penalty · mean(p+n)` | Encourages sparse rules |
| `lambda_diversity · cosine_sim` | Prevents duplicate clauses per action |

---

## Pipeline

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

Trains the full neuro-symbolic pipeline end-to-end. SAE and bottleneck are trained jointly with the logic layer, then the SAE is frozen and the logic layer fine-tunes.

```bash
python train_sae_logic.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path   ./stage1_outputs/stage1_outputs.pt \
    --mode joint \
    --n_epochs 400 --sae_freeze_epoch 200 --k 20 \
    --save_dir ./sae_logic_v2_outputs
```

**Key outputs:**

- `sae_logic_v2_outputs/sae_logic_v2_model.pt` — full model checkpoint
- `sae_logic_v2_outputs/learned_rules.json` — extracted DNF rules
- `sae_logic_v2_outputs/training_curves.png` — loss, accuracy, binarization plots

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--features_path` | required | Path to `collected_data.pt` |
| `--stage1_path` | `None` | Path to `stage1_outputs.pt` (recommended) |
| `--hidden_dim` | 256 | SAE hidden dimension |
| `--k` | 10 | TopK sparsity — active features per sample |
| `--n_clauses_per_action` | 10 | DNF clauses per action |
| `--mode` | `joint` | `joint` or `two_stage` |
| `--n_epochs` | 200 | Total training epochs |
| `--sae_freeze_epoch` | 80 | Epoch to freeze SAE weights |
| `--alpha_recon` | 1.0 | SAE reconstruction loss weight |
| `--beta_action` | 2.0 | Action prediction loss weight |
| `--bimodal_warmup` | 30 | Epochs before bimodality loss starts |
| `--bimodal_ramp` | 50 | Epochs to ramp bimodality to max |
| `--bimodal_max` | 1.0 | Maximum bimodality loss weight |
| `--l0_penalty` | 1e-4 | Rule sparsity penalty |
| `--lambda_sparsity` | 5e-3 | SAE L1 sparsity weight |
| `--action_class_weights` | all 1.0 | Per-action loss weights (7 values) |
| `--sae_lr` | 1e-3 | SAE learning rate |
| `--logic_lr` | 3e-3 | Logic layer learning rate |
| `--save_dir` | `./sae_logic_v2_outputs` | Output directory |

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
    --model_path ./sae_logic_v2_outputs/sae_logic_v2_model.pt \
    --ppo_path   ppo_doorkey_5x5.zip \
    --n_episodes 100 \
    --print_rules

# Full comparison with failure diagnostics
python check_success_rules.py \
    --model_path ./sae_logic_v2_outputs/sae_logic_v2_model.pt \
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
TurnRight←                                                
    (f_4 ∧ f_10 ∧ f_28 ∧ f_32 ∧ f_36 ∧ f_44 ∧ f_56 ∧ f_68 ∧ ¬f_71 ∧ f_109 ∧ f_118 ∧ f_144 ∧ f_145 ∧ f_173 ∧ f_204 ∧ f_206 ∧ f_247 ∧ f_248)                                         

```

Each rule is a conjunction of SAE feature activations. Positive literals (`f_i`) mean the feature must be active; negative literals (`¬f_i`) mean it must be inactive.

---

## File structure

```
XRL/
├── feature_space_analysis.py   # Stage 1: SVD, ICA, data collection
├── train_sae_logic.py          # Stage 2: SAE + logic training
├── check_success_rules.py      # Stage 3: game evaluation
├── sparse_concept_autoencoder.py  # SAE implementation
├── ppo_doorkey_5x5.zip         # Pre-trained PPO policy
├── stage1_outputs/
│   ├── collected_data.pt       # Raw features + action labels
│   ├── stage1_outputs.pt       # SVD/ICA/normalization results
│   ├── stage1_diagnostics.png
│   └── action_distribution.png
└── sae_logic_v2_outputs/
    ├── sae_logic_v2_model.pt   # Trained model checkpoint
    ├── learned_rules.json      # Extracted DNF rules
    └── training_curves.png     # Training diagnostics
```

---

## Dependencies

```bash
pip install torch gymnasium minigrid stable-baselines3
pip install scikit-learn scipy matplotlib seaborn
```