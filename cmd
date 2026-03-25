python feature_space_analysis.py \
    --model_path ppo_doorkey_5x5.zip \
    --env_name MiniGrid-DoorKey-5x5-v0 \
    --n_episodes 800 \
    --save_dir ./stage1_outputs


Or if you've already collected features:
pythonpython feature_space_analysis.py \
    --features_path ./collected_data.pt \
    --save_dir ./stage1_outputs


# Single run
python sparse_concept_autoencoder.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --hidden_dim 512 --k 10 --n_epochs 100

# Multi-run for Stage 3 consensus
python sparse_concept_autoencoder.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --hidden_dim 512 --k 10 --n_epochs 100 --n_runs 10


python sparse_concept_autoencoder.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --hidden_dim 32 \
    --k 4 \
    --n_epochs 200 \
    --batch_size 256 \
    --lr 3e-4 \
    --alpha 1.0 \
    --beta 2.0 \
    --lambda1 5e-3 \
    --lambda2 1e-3 \
    --lambda3 0.5 \
    --interaction_top_n 30 \
    --dead_threshold 5e-3 \
    --resample_every 300 \
    --log_every 10 \
    --n_runs 5 \
    --save_dir ./stage2_outputs

# Without env grounding (faster)
python consensus_concepts.py \
    --stage2_dir ./stage2_outputs \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --m_min_frac 0.8 \
    --save_dir ./stage3_outputs

# With env grounding (recommended for paper results)
python consensus_concepts.py \
    --stage2_dir ./stage2_outputs \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --model_path ppo_doorkey_5x5.zip \
    --env_name MiniGrid-DoorKey-5x5-v0 \
    --m_min_frac 0.8 \
    --save_dir ./stage3_outputs

python rule_extraction.py \
    --stage2_dir ./stage2_outputs \
    --stage3_path ./stage3_outputs/stage3_outputs.pt \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --weight_threshold 0.1 \
    --save_dir ./stage4_outputs