# check_success.py
# Test SAE + action predictor by playing the game and measuring success rate

import argparse
import os

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from tqdm import tqdm

from ConceptExtractor import SparseAutoencoder


class SAEAgent:
    """Agent that uses SAE concepts + action predictor to play"""

    def __init__(self, ppo_model, sae, action_predictor, device='cpu'):
        self.ppo_model = ppo_model
        self.sae = sae
        self.action_predictor = action_predictor
        self.device = device

        self.sae.eval()
        self.action_predictor.eval()

    def predict(self, obs):
        """Predict action from observation"""
        with torch.no_grad():
            # Extract features using PPO's feature extractor
            obs_tensor = torch.as_tensor(obs).float().to(self.device)
            features = self.ppo_model.policy.features_extractor(obs_tensor)

            # Get concepts from SAE
            _, concepts = self.sae(features)

            # Predict action from concepts
            action_logits = self.action_predictor(concepts)
            action = action_logits.argmax(dim=-1)

            return action.cpu().numpy()


def load_models(model_dir, ppo_path, device='cpu'):
    """Load SAE and action predictor"""
    print(f"Loading models from {model_dir}...")

    # Load PPO model for feature extraction
    ppo_model = PPO.load(ppo_path, device=device)

    # Load SAE
    sae_data = torch.load(os.path.join(model_dir, 'sae.pt'), map_location=device)
    config = sae_data['config']

    sae = SparseAutoencoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        k=config['k'],
        tau=config['tau']
    ).to(device)

    sae.encoder.weight.data = sae_data['encoder_weight']
    sae.encoder.bias.data = sae_data['encoder_bias']
    sae.decoder.weight.data = sae_data['decoder_weight']
    sae.decoder.bias.data = sae_data['decoder_bias']

    # Load action predictor
    action_pred_path = os.path.join(model_dir, 'action_predictor.pt')
    if not os.path.exists(action_pred_path):
        raise FileNotFoundError(f"Action predictor not found at {action_pred_path}")

    action_pred_data = torch.load(action_pred_path, map_location=device)
    action_predictor = nn.Linear(config['hidden_dim'], config['n_actions']).to(device)
    action_predictor.weight.data = action_pred_data['weight']
    action_predictor.bias.data = action_pred_data['bias']

    print(f"Loaded SAE with {config['hidden_dim']} concepts, k={config['k']}")

    return ppo_model, sae, action_predictor, config


def evaluate_agent(agent, env_name, n_episodes=100, max_steps=1000, seed=42, render=False):
    """Evaluate agent by playing episodes"""
    print(f"\nEvaluating agent on {env_name} for {n_episodes} episodes...")

    # Create environment
    def make_env():
        def _init():
            if render:
                env = gym.make(env_name, render_mode='human')
            else:
                env = gym.make(env_name)
            env = ImgObsWrapper(env)
            return env
        return _init

    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)
    env.seed(seed)

    successes = 0
    total_rewards = []
    episode_lengths = []

    pbar = tqdm(total=n_episodes, desc="Playing episodes")

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        for step in range(max_steps):
            # Predict action using SAE agent
            action = agent.predict(obs)

            # Step environment
            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]
            episode_length += 1

            if done[0]:
                # Check if episode was successful
                if reward[0] > 0:  # Success in MiniGrid gives positive reward
                    successes += 1
                break

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        pbar.update(1)
        pbar.set_postfix({
            'success_rate': f'{100.0 * successes / (episode + 1):.1f}%',
            'avg_reward': f'{np.mean(total_rewards):.2f}',
            'avg_length': f'{np.mean(episode_lengths):.1f}'
        })

    pbar.close()

    # Compute statistics
    success_rate = 100.0 * successes / n_episodes
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average episode length: {avg_length:.1f} ± {std_length:.1f}")
    print(f"Successful episodes: {successes}/{n_episodes}")
    print("="*60)

    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'std_length': std_length,
        'successes': successes,
        'n_episodes': n_episodes
    }


def compare_with_ppo(ppo_model, env_name, n_episodes=100, max_steps=1000, seed=42):
    """Compare SAE agent with original PPO agent"""
    print(f"\n{'='*60}")
    print("BASELINE: Original PPO Agent")
    print("="*60)

    # Create environment
    def make_env():
        def _init():
            env = gym.make(env_name)
            env = ImgObsWrapper(env)
            return env
        return _init

    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)
    env.seed(seed)

    successes = 0
    total_rewards = []
    episode_lengths = []

    pbar = tqdm(total=n_episodes, desc="PPO baseline")

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(max_steps):
            action, _ = ppo_model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]
            episode_length += 1

            if done[0]:
                if reward[0] > 0:
                    successes += 1
                break

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        pbar.update(1)
        pbar.set_postfix({
            'success_rate': f'{100.0 * successes / (episode + 1):.1f}%',
            'avg_reward': f'{np.mean(total_rewards):.2f}'
        })

    pbar.close()

    success_rate = 100.0 * successes / n_episodes
    avg_reward = np.mean(total_rewards)

    print(f"\nPPO Success rate: {success_rate:.2f}%")
    print(f"PPO Average reward: {avg_reward:.2f}")

    return success_rate, avg_reward


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAE agent by playing the game')
    parser.add_argument('--model_dir', type=str, default='./concept_models',
                        help='Directory containing SAE model')
    parser.add_argument('--ppo_path', type=str, default='ppo_doorkey_5x5.zip',
                        help='Path to PPO model')
    parser.add_argument('--env_name', type=str, default='MiniGrid-DoorKey-5x5-v0',
                        help='Environment name')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--compare_ppo', action='store_true',
                        help='Also evaluate original PPO agent for comparison')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes (only first 5)')

    args = parser.parse_args()

    # Set device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Load models
    ppo_model, sae, action_predictor, config = load_models(
        args.model_dir, args.ppo_path, device
    )

    # Create SAE agent
    agent = SAEAgent(ppo_model, sae, action_predictor, device)

    # Evaluate SAE agent
    sae_results = evaluate_agent(
        agent,
        args.env_name,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        render=args.render
    )

    # Compare with PPO if requested
    if args.compare_ppo:
        ppo_success, ppo_reward = compare_with_ppo(
            ppo_model,
            args.env_name,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            seed=args.seed
        )

        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"SAE Agent:  {sae_results['success_rate']:.2f}% success")
        print(f"PPO Agent:  {ppo_success:.2f}% success")
        print(f"Difference: {sae_results['success_rate'] - ppo_success:+.2f}%")
        print("="*60)


if __name__ == '__main__':
    main()
