# Stage 1
python feature_space_analysis.py \
    --model_path ppo_doorkey_5x5.zip \
    --env_name MiniGrid-DoorKey-5x5-v0 \
    --n_episodes 800 \
    --save_dir ./stage1_outputs


Or if you've already collected features:
pythonpython feature_space_analysis.py \
    --features_path ./collected_data.pt \
    --save_dir ./stage1_outputs

# Stage 2  
python sparse_concept_autoencoder.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --hidden_dim 128 --k 6 \
    --n_epochs 200 --lr 3e-4 \
    --alpha 1.0 --beta 2.0 \
    --lambda1 5e-3 --lambda2 1e-3 --lambda3 0.5 \
    --n_runs 10 \
    --save_dir ./stage2_outputs_v2

# Stage 3 Relaxed clustering + consensus
python consensus_concepts.py \
    --stage2_dir ./stage2_outputs_v2 \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --model_path ppo_doorkey_5x5.zip \
    --env_name MiniGrid-DoorKey-5x5-v0 \
    --distance_threshold 0.4 \
    --m_min_frac 0.6 \
    --save_dir ./stage3_outputs_v2

# Stage 4
python rule_extraction.py \
    --stage2_dir ./stage2_outputs_v2 \
    --stage3_path ./stage3_outputs_v2/stage3_outputs.pt \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --weight_threshold 0.05 \
    --consensus_retrain_epochs 500 \
    --save_dir ./stage4_outputs_v3