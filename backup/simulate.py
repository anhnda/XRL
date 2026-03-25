# eval_minigrid_doorkey_5x5.py

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


def make_env():
    def _init():
        env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")
        env = ImgObsWrapper(env)
        return env
    return _init


def main():
    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    model = PPO.load("ppo_doorkey_5x5.zip")

    obs = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if dones[0]:
            obs = env.reset()


if __name__ == "__main__":
    main()