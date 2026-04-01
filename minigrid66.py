# train_minigrid_doorkey_6x6.py

import gymnasium as gym
import minigrid  # noqa: F401, needed so envs are registered
import numpy as np
import torch
import torch.nn as nn

from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


# ------------------------------------------------------------
# Custom CNN extractor for MiniGrid image observations
# Official MiniGrid training page shows this pattern.
# ------------------------------------------------------------
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # After VecTransposeImage, observation_space is in CHW format: (3, 7, 7)
        n_input_channels = observation_space.shape[0]  # First dimension is channels
        print(f"MiniGrid observation shape: {observation_space.shape}, using {n_input_channels} input channels")

        # Slightly deeper architecture for the larger 6x6 grid
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()  # Already NCHW
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # VecTransposeImage ensures observations are already in NCHW format
        return self.linear(self.cnn(observations.float()))


def make_env(render_mode=None, seed=0):
    def _init():
        env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode=render_mode)
        env = ImgObsWrapper(env)   # dict -> image only
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    seed = 42
    n_envs = 8

    train_env = DummyVecEnv([make_env(seed=seed + i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(seed=1000)])

    # SB3 CNN expects channel-first images
    train_env = VecTransposeImage(train_env)
    eval_env = VecTransposeImage(eval_env)

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tb_doorkey_6x6/",
        seed=seed,
        device="auto",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_doorkey_6x6/",
        log_path="./eval_logs_doorkey_6x6/",
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=500_000, callback=eval_callback)
    model.save("ppo_doorkey_6x6")


if __name__ == "__main__":
    main()