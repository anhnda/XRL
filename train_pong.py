# train_pong.py
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage


# ------------------------------------------------------------
# Nature DQN-style CNN — the standard for Atari
# Input: (N, 4, 84, 84) stacked grayscale frames
# ------------------------------------------------------------
class AtariCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 4 stacked frames

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),  # → 20×20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),                # → 9×9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),                # → 7×7
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]  # 64 * 7 * 7 = 3136

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.float()))


def main():
    seed = 42
    n_envs = 8
    game = "ALE/Pong-v5"

    # make_atari_env handles: NoopReset, MaxAndSkip, EpisodicLife,
    # FireReset, WarpFrame (84×84 grayscale), ClipReward
    train_env = make_atari_env(game, n_envs=n_envs, seed=seed)
    eval_env  = make_atari_env(game, n_envs=1,      seed=1000)

    # Stack 4 consecutive frames → (4, 84, 84)
    train_env = VecFrameStack(train_env, n_stack=4)
    eval_env  = VecFrameStack(eval_env,  n_stack=4)

    # SB3 CNN expects channel-first (already true after VecFrameStack)
    train_env = VecTransposeImage(train_env)
    eval_env  = VecTransposeImage(eval_env)

    policy_kwargs = dict(
        features_extractor_class=AtariCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[512], vf=[512]),
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=128,          # per env per rollout (128 * 8 envs = 1024 total)
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tb_pong/",
        seed=seed,
        device="auto",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_pong/",
        log_path="./eval_logs_pong/",
        eval_freq=50_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=10_000_000, callback=eval_callback)
    model.save("ppo_pong")


if __name__ == "__main__":
    main()