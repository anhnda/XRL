Pong
python feature_space_analysis.py \
    --model_path ppo_pong.zip \
    --n_episodes 200 \
    --save_dir ./pong_stage1



python train_sae_logic.py --features_path ./pong_stage1/collected_data.pt   --stage1_path ./pong_stage1/stage1_outputs.pt --hidden_dim 512 --k 60     --n_clauses_per_action 40     --sae_pretrain_epochs 150     --n_epochs 500     --logic_lr 3e-3 --bottleneck_lr 1e-2     --beta_action 5.0 --l0_penalty 5e-6     --lambda_sparsity 5e-4     --bimodal_max 0.2 --bimodal_warmup 60 --bimodal_ramp 150 --save_dir ./sae_logic_pong_A

    --action_class_weights 1.0 2.0 1.2 1.0 \

Atari Next good 94%
python feature_space_analysis.py \
    --model_path ppo_atari_breakout.zip \
    --env_name "ALE/Breakout-v5" \
    --n_episodes 4000 \
    --save_dir ./stage1_atari

python train_sae_logic.py \                                                                                                 
    --features_path ./stage1_atari/collected_data.pt \                                                                      
    --stage1_path ./stage1_atari/stage1_outputs.pt \                                                                        
    --hidden_dim 256 --k 60 \                                                                                               
    --n_clauses_per_action 40 \                                                                                             
    --sae_pretrain_epochs 150 \                                                                                             
    --n_epochs 500 \                                                                                                        
    --logic_lr 3e-3 --bottleneck_lr 1e-2 \
    --beta_action 5.0 --l0_penalty 5e-6 \
    --lambda_sparsity 5e-4 \
    --bimodal_max 0.2 --bimodal_warmup 60 --bimodal_ramp 150 \ 
    --action_class_weights 1.0 2.0 1.2 1.0 \
    --save_dir ./sae_logic_atari_E
python check_success_rules.py \
    --model_path ./sae_logic_atari_E/sae_logic_v3_model.pt \
    --ppo_path ppo_atari_breakout.zip \
    --env_name "ALE/Breakout-v5" \
    --n_episodes 100 --print_rules

python train_sae_logic.py \
    --features_path ./stage1_atari/collected_data.pt \
    --stage1_path ./stage1_atari/stage1_outputs.pt \
    --hidden_dim 256 --k 60 \
    --n_clauses_per_action 40 \
    --sae_pretrain_epochs 150 \
    --n_epochs 500 \
    --logic_lr 3e-3 --bottleneck_lr 1e-2 \
    --beta_action 5.0 --l0_penalty 5e-6 \
    --lambda_sparsity 5e-4 \
    --bimodal_max 0.2 --bimodal_warmup 60 --bimodal_ramp 150 \
    --action_class_weights 1.0 2.0 1.2 1.0 \
    --save_dir ./sae_logic_atari_E, 

python check_success_rules.py \
    --model_path ./sae_logic_atari_E/sae_logic_v3_model.pt \
    --ppo_path ppo_atari_breakout.zip \
    --env_name "ALE/Breakout-v5" \
    --n_episodes 100 --print_rules



Atari Good 8x
python feature_space_analysis.py \
    --model_path ppo_atari_breakout.zip \
    --env_name "ALE/Breakout-v5" \
    --n_episodes 200 \
    --save_dir ./stage1_atari

python train_sae_logic.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --hidden_dim 256 --k 60 \
    --n_clauses_per_action 40 \
    --sae_pretrain_epochs 150 \
    --n_epochs 500 \
    --logic_lr 3e-3 --bottleneck_lr 1e-2 \
    --beta_action 5.0 --l0_penalty 5e-6 \
    --lambda_sparsity 5e-4 \
    --bimodal_max 0.2 --bimodal_warmup 60 --bimodal_ramp 150 \
    --action_class_weights 1.0 2.0 1.2 1.0 \
    --save_dir ./sae_logic_atari_E

python check_success_rules.py \
    --model_path ./sae_logic_atari_E/sae_logic_v3_model.pt \
    --ppo_path ppo_atari_breakout.zip \
    --env_name "ALE/Breakout-v5" \
    --n_episodes 100 --print_rules

Rules agent: 100%|██████████████████████████████| 100/100 [00:12<00:00,  7.70it/s, success=84.0%, avg_r=3.180, avg_len=33.1]

============================================================
RULES AGENT — EVALUATION RESULTS
============================================================
  Environment:      ALE/Breakout-v5 (atari)
  Episodes:         100
  Successes:        84/100
  Success rate:     84.00%
  Avg reward:       3.1800 ± 2.8049
  Avg ep length:    33.1 ± 24.8

  Action usage during play:
    NOOP          :    977 ( 29.5%) ███████████
    FIRE          :    385 ( 11.6%) ████
    RIGHT         :    855 ( 25.8%) ██████████
    LEFT          :   1091 ( 33.0%) █████████████
============================================================

python feature_space_analysis.py \
    --model_path ppo_doorkey_6x6.zip \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --n_episodes 2000 \
    --save_dir ./stage1_outputs
python train_sae_logic.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --hidden_dim 300 --k 50 \
    --n_clauses_per_action 10 \
    --sae_pretrain_epochs 80 \
    --n_epochs 300 \
    --max_grad_norm 5.0 --save_training_data
python check_success_rules.py \
    --model_path ./sae_logic_v3_outputs/sae_logic_v3_model.pt \
    --ppo_path ppo_doorkey_5x5.zip \
    --n_episodes 100 --print_rules
    
# Step 1: Re-collect with observations (same model, same env, same episodes)
python collect_with_observations.py \
    --model_path ppo_doorkey_5x5.zip \
    --env_name MiniGrid-DoorKey-5x5-v0 \
    --n_episodes 2000 \
    --save_path ./stage1_outputs/collected_data_with_obs.pt

# Step 2: Run visualization (point at the new file)
python visualize_rule_features.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --model_path ./sae_logic_v3_outputs/sae_logic_v3_model.pt

python train_sae_logic.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --hidden_dim 300 --k 50 \
    --n_clauses_per_action 10 \
    --sae_pretrain_epochs 50 \
    --n_epochs 400 \
    --max_grad_norm 5.0
Or if you've already collected features:
pythonpython feature_space_analysis.py \
    --features_path ./collected_data.pt \
    --save_dir ./stage1_outputs

# More overcomplete, more runs, relaxed consensus
python sparse_concept_autoencoder.py \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --hidden_dim 128 --k 6 \
    --n_epochs 200 --lr 3e-4 \
    --alpha 1.0 --beta 2.0 \
    --lambda1 5e-3 --lambda2 1e-3 --lambda3 0.5 \
    --n_runs 10 \
    --save_dir ./stage2_outputs_v2

# Relaxed clustering + consensus
python consensus_concepts.py \
    --stage2_dir ./stage2_outputs_v2 \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --model_path ppo_doorkey_5x5.zip \
    --env_name MiniGrid-DoorKey-5x5-v0 \
    --distance_threshold 0.4 \
    --m_min_frac 0.6 \
    --save_dir ./stage3_outputs_v2
python rule_extraction.py \
    --stage2_dir ./stage2_outputs_v2 \
    --stage3_path ./stage3_outputs_v2/stage3_outputs.pt \
    --features_path ./stage1_outputs/collected_data.pt \
    --stage1_path ./stage1_outputs/stage1_outputs.pt \
    --weight_threshold 0.05 \
    --consensus_retrain_epochs 500 \
    --save_dir ./stage4_outputs_v3