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