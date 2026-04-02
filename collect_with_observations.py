#!/usr/bin/env python3
"""
Re-collect PPO rollout data WITH raw observations for feature visualization.

This is a patched version of the collect_features() from feature_space_analysis.py
that additionally saves the rendered observation images.

Usage:
    python collect_with_observations.py \
        --model_path ppo_doorkey_5x5.zip \
        --env_name MiniGrid-DoorKey-5x5-v0 \
        --n_episodes 800 \
        --save_path ./stage1_outputs/collected_data_with_obs.pt

Then point visualize_rule_features.py at the new file:
    python visualize_rule_features.py \
        --features_path ./stage1_outputs/collected_data_with_obs.pt \
        ...
"""

import argparse
import os

import numpy as np
import torch
import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from tqdm import tqdm


def collect_with_observations(
    model_path: str,
    env_name: str,
    n_episodes: int = 800,
    seed: int = 42,
    tile_size: int = 8,
    save_obs_mode: str = "both",
):
    """
    Collect features, actions, AND observations from a frozen PPO policy.

    Args:
        model_path: Path to PPO .zip model
        env_name: MiniGrid environment name
        n_episodes: Number of episodes to collect
        seed: Random seed
        tile_size: Pixel size per grid cell for rendered images
        save_obs_mode: What to save:
            "grid"   — raw grid encoding (H, W, 3) with object/color/state channels
            "pixel"  — rendered RGB pixel images
            "both"   — save both (recommended, ~2x storage but most flexible)

    Returns:
        dict with keys: features, actions, observations_grid, observations_pixel
    """
    print(f"Loading PPO model from {model_path}...")
    model = PPO.load(model_path)

    # We need TWO envs:
    #   1. A wrapped env for the PPO policy (VecTransposeImage expects CHW)
    #   2. Access to the raw MiniGrid env for grid observations
    # 
    # The trick: DummyVecEnv wraps a single env, so we can reach into it.

    raw_env_holder = [None]  # mutable container for the raw env reference

    def make_env():
        def _init():
            env = gym.make(env_name)
            env = ImgObsWrapper(env)
            raw_env_holder[0] = env
            return env
        return _init

    vec_env = DummyVecEnv([make_env()])
    vec_env = VecTransposeImage(vec_env)

    features_list = []
    actions_list = []
    obs_grid_list = []
    obs_pixel_list = []

    print(f"Collecting {n_episodes} episodes from {env_name}...")
    obs = vec_env.reset()
    episode_count = 0
    step_count = 0

    with torch.no_grad():
        pbar = tqdm(total=n_episodes, desc="Episodes")
        while episode_count < n_episodes:
            # --- Get raw observation from the underlying MiniGrid env ---
            raw_env = raw_env_holder[0]

            if save_obs_mode in ("grid", "both"):
                # Grid encoding: (H, W, 3) with channels [object_type, color, state]
                # This is what ImgObsWrapper returns before any transposing
                # Access directly from the underlying env
                try:
                    grid_obs = raw_env.unwrapped.gen_obs()['image']
                except AttributeError:
                    # Fallback: grab from the vec env's buffer (already HWC)
                    grid_obs = vec_env.get_attr('get_wrapper_attr', indices=[0])
                    grid_obs = raw_env.observation(raw_env.unwrapped.gen_obs())
                obs_grid_list.append(grid_obs.copy())

            if save_obs_mode in ("pixel", "both"):
                # Rendered pixel image
                try:
                    # MiniGrid's get_obs_render gives a pixel rendering of the agent's view
                    pixel_obs = raw_env.unwrapped.get_obs_render(
                        raw_env.unwrapped.gen_obs()['image'], tile_size=tile_size
                    )
                except (AttributeError, TypeError):
                    # Fallback: full render
                    try:
                        pixel_obs = raw_env.unwrapped.render()
                        if pixel_obs is None:
                            pixel_obs = np.zeros((tile_size * 7, tile_size * 7, 3), dtype=np.uint8)
                    except Exception:
                        pixel_obs = np.zeros((tile_size * 7, tile_size * 7, 3), dtype=np.uint8)
                obs_pixel_list.append(pixel_obs)

            # --- Get features from PPO's feature extractor ---
            action, _ = model.predict(obs, deterministic=False)
            obs_tensor = torch.as_tensor(obs).float().to(model.device)
            features = model.policy.features_extractor(obs_tensor)

            features_list.append(features.cpu())
            actions_list.append(torch.tensor(action))

            # --- Step ---
            obs, rewards, dones, infos = vec_env.step(action)
            step_count += 1

            if dones[0]:
                episode_count += 1
                pbar.update(1)

        pbar.close()

    vec_env.close()

    # Stack everything
    result = {
        'features': torch.cat(features_list, dim=0),
        'actions': torch.cat(actions_list, dim=0),
    }

    if obs_grid_list:
        result['observations'] = torch.tensor(np.stack(obs_grid_list))
        print(f"  Grid observations: {result['observations'].shape}")

    if obs_pixel_list:
        result['observations_pixel'] = torch.tensor(np.stack(obs_pixel_list))
        print(f"  Pixel observations: {result['observations_pixel'].shape}")

    print(f"  Features: {result['features'].shape}")
    print(f"  Actions: {result['actions'].shape}")
    print(f"  Total steps: {step_count}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Collect PPO rollout data with raw observations"
    )
    parser.add_argument("--model_path", type=str, default="ppo_doorkey_5x5.zip",
                        help="Path to trained PPO model")
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-5x5-v0",
                        help="MiniGrid environment name")
    parser.add_argument("--n_episodes", type=int, default=800,
                        help="Number of episodes to collect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile_size", type=int, default=8,
                        help="Pixel size per grid cell for rendered images")
    parser.add_argument("--save_obs_mode", type=str, default="both",
                        choices=["grid", "pixel", "both"],
                        help="What observation format to save")
    parser.add_argument("--save_path", type=str,
                        default="./stage1_outputs/collected_data_with_obs.pt")
    args = parser.parse_args()

    data = collect_with_observations(
        model_path=args.model_path,
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        seed=args.seed,
        tile_size=args.tile_size,
        save_obs_mode=args.save_obs_mode,
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(data, args.save_path)
    print(f"\nSaved to {args.save_path}")
    print(f"  File size: {os.path.getsize(args.save_path) / 1e6:.1f} MB")

    # Verify the observations look right
    if 'observations' in data:
        obs = data['observations']
        print(f"\n  Grid obs sample [0,0]: obj_type={obs[0, 0, 0, 0].item()}, "
              f"color={obs[0, 0, 0, 1].item()}, state={obs[0, 0, 0, 2].item()}")
        print(f"  Grid obs range: [{obs.min().item()}, {obs.max().item()}]")
        print(f"  Unique object types: {torch.unique(obs[:, :, :, 0]).tolist()}")


if __name__ == "__main__":
    main()