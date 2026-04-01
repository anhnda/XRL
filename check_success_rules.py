"""
check_success_rules.py
======================
Evaluate the SAELogicAgentV2 (trained logic rules) by playing MiniGrid
and measuring success rate. Compares against the original PPO baseline.

Usage:
    # Rules agent only
    python check_success_rules.py \
        --model_path ./sae_logic_v2_outputs/sae_logic_v2_model.pt \
        --ppo_path   ppo_doorkey_5x5.zip \
        --env_name   MiniGrid-DoorKey-5x5-v0 \
        --n_episodes 100

    # Rules agent + PPO baseline comparison
    python check_success_rules.py \
        --model_path ./sae_logic_v2_outputs/sae_logic_v2_model.pt \
        --ppo_path   ppo_doorkey_5x5.zip \
        --compare_ppo
"""

import argparse
import os

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from dataclasses import asdict
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from tqdm import tqdm

from train_sae_logic import SAELogicAgentV2, SAELogicConfig

ACTION_NAMES = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]


# ============================================================================
# Rules-based agent wrapper
# ============================================================================
class RulesAgent:
    """
    Wraps SAELogicAgentV2 for game-play evaluation.

    Pipeline per step:
        raw obs → PPO feature extractor → normalize → SAE encode
                → sigmoid bottleneck → product t-norm logic → remap → MiniGrid action
    """

    def __init__(
        self,
        ppo_model: PPO,
        logic_model: SAELogicAgentV2,
        device: str,
        remap_back: dict,
    ):
        self.ppo_model   = ppo_model
        self.logic_model = logic_model
        self.device      = device
        self.remap_back  = remap_back  # model 0-3 → MiniGrid action index

        self.logic_model.eval()

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Args:
            obs: raw observation from VecEnv, shape (1, C, H, W)

        Returns:
            MiniGrid action array of shape (1,)
        """
        with torch.no_grad():
            obs_tensor    = torch.as_tensor(obs).float().to(self.device)
            features      = self.ppo_model.policy.features_extractor(obs_tensor)
            features_norm = self.logic_model.normalize(features)
            action_logits = self.logic_model(features_norm)
            action_model  = action_logits.argmax(dim=-1).item()

        # Remap from model's compressed action space back to MiniGrid's full space
        action_minigrid = self.remap_back[action_model]
        return np.array([action_minigrid])

    def print_rules(self):
        """Print the learned rules being used."""
        action_names = list(self.remap_back.values())
        # Build display names from remap
        minigrid_names = {
            0: "TurnLeft", 1: "TurnRight", 2: "Forward",
            3: "Pickup",   4: "Drop",      5: "Toggle",  6: "Done"
        }
        display_names = [minigrid_names.get(v, str(v)) for v in sorted(self.remap_back.values())]
        rules = self.logic_model.extract_rules(action_names=display_names)

        print("\n" + "=" * 60)
        print("ACTIVE LOGIC RULES")
        print("=" * 60)
        for action_name, clauses in rules.items():
            print(f"\n{action_name} ←")
            unique = list(dict.fromkeys(clauses))
            for i, clause in enumerate(unique):
                print(f"    {clause}")
                if i < len(unique) - 1:
                    print("  ∨")


def load_rules_agent(model_path: str, ppo_path: str, device: str) -> RulesAgent:
    """Load SAELogicAgentV2 checkpoint and PPO feature extractor."""
    print(f"Loading PPO from {ppo_path}...")
    ppo_model = PPO.load(ppo_path, device=device)

    print(f"Loading logic model from {model_path}...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    config      = SAELogicConfig(**ckpt['config'])
    logic_model = SAELogicAgentV2(config, device=device)
    logic_model.load_state_dict(ckpt['model_state'])
    logic_model.to(device)
    logic_model.eval()

    # Normalization stats — already on device after .to(device)
    feat_mean = ckpt['feature_mean'].to(device)
    feat_std  = ckpt['feature_std'].to(device)
    logic_model.set_normalization(feat_mean, feat_std)

    # Load action remap saved during training, invert for game-play
    # remap      : MiniGrid action → model index  e.g. {1:0, 2:1, 3:2, 5:3}
    # remap_back : model index → MiniGrid action  e.g. {0:1, 1:2, 2:3, 3:5}
    remap      = ckpt.get('action_remap', {1: 0, 2: 1, 3: 2, 5: 3})
    remap_back = {v: k for k, v in remap.items()}

    action_names = ckpt.get('action_names', ["TurnRight", "Forward", "Pickup", "Toggle"])

    print(f"  Val accuracy at save : {ckpt['best_val_acc']:.3f}")
    print(f"  Active actions       : {action_names}")
    print(f"  Model→MiniGrid remap : {remap_back}")
    print(f"  Architecture         : {config.input_dim}d → SAE({config.hidden_dim},"
          f" k={config.k}) → Logic({config.n_clauses_per_action} clauses/action)"
          f" → {config.n_actions} actions")

    return RulesAgent(ppo_model, logic_model, device, remap_back)

# ============================================================================
# Environment factory
# ============================================================================

def make_vec_env(env_name: str, seed: int, render: bool = False):
    def _make():
        def _init():
            e = gym.make(env_name, render_mode="human" if render else None)
            return ImgObsWrapper(e)
        return _init

    env = DummyVecEnv([_make()])
    env = VecTransposeImage(env)
    env.seed(seed)
    return env


# ============================================================================
# Evaluation loop
# ============================================================================

def evaluate_rules_agent(
    agent: RulesAgent,
    env_name: str,
    n_episodes: int = 100,
    max_steps: int = 500,
    seed: int = 42,
    render: bool = False,
    verbose_failures: bool = False,
) -> dict:
    """
    Play n_episodes and collect success rate, reward, and episode length.

    A MiniGrid episode is successful when the final reward > 0.
    """
    env = make_vec_env(env_name, seed, render)

    successes      = 0
    total_rewards  = []
    episode_lengths = []
    action_counts  = np.zeros(len(ACTION_NAMES), dtype=int)

    pbar = tqdm(total=n_episodes, desc="Rules agent")

    for ep in range(n_episodes):
        obs          = env.reset()
        ep_reward    = 0.0
        ep_length    = 0
        ep_actions   = []

        for _ in range(max_steps):
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)

            ep_reward  += reward[0]
            ep_length  += 1
            ep_actions.append(int(action[0]))
            action_counts[int(action[0])] += 1

            if done[0]:
                if reward[0] > 0:
                    successes += 1
                elif verbose_failures:
                    action_seq = [ACTION_NAMES[a] for a in ep_actions[-10:]]
                    print(f"\n  [Ep {ep+1}] FAILED — last 10 actions: {action_seq}")
                break

        total_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        pbar.update(1)
        pbar.set_postfix({
            'success': f'{100.0 * successes / (ep + 1):.1f}%',
            'avg_r':   f'{np.mean(total_rewards):.3f}',
            'avg_len': f'{np.mean(episode_lengths):.1f}',
        })

    pbar.close()
    env.close()

    success_rate = 100.0 * successes / n_episodes

    print("\n" + "=" * 60)
    print("RULES AGENT — EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes:         {n_episodes}")
    print(f"  Successes:        {successes}/{n_episodes}")
    print(f"  Success rate:     {success_rate:.2f}%")
    print(f"  Avg reward:       {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
    print(f"  Avg ep length:    {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"\n  Action usage during play:")
    total_actions = action_counts.sum()
    for name, cnt in zip(ACTION_NAMES, action_counts):
        bar = "█" * int(40 * cnt / max(total_actions, 1))
        print(f"    {name:10s}: {cnt:6d} ({100.0*cnt/max(total_actions,1):5.1f}%) {bar}")
    print("=" * 60)

    return {
        'success_rate':    success_rate,
        'successes':       successes,
        'n_episodes':      n_episodes,
        'avg_reward':      float(np.mean(total_rewards)),
        'std_reward':      float(np.std(total_rewards)),
        'avg_length':      float(np.mean(episode_lengths)),
        'std_length':      float(np.std(episode_lengths)),
        'action_counts':   action_counts.tolist(),
    }


# ============================================================================
# PPO baseline
# ============================================================================

def evaluate_ppo_baseline(
    ppo_model: PPO,
    env_name: str,
    n_episodes: int = 100,
    max_steps: int = 500,
    seed: int = 42,
) -> dict:
    """Evaluate original PPO policy as baseline."""
    env = make_vec_env(env_name, seed)

    successes      = 0
    total_rewards  = []
    episode_lengths = []

    pbar = tqdm(total=n_episodes, desc="PPO baseline")

    for ep in range(n_episodes):
        obs       = env.reset()
        ep_reward = 0.0
        ep_length = 0

        for _ in range(max_steps):
            action, _ = ppo_model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)

            ep_reward += reward[0]
            ep_length += 1

            if done[0]:
                if reward[0] > 0:
                    successes += 1
                break

        total_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        pbar.update(1)
        pbar.set_postfix({
            'success': f'{100.0 * successes / (ep + 1):.1f}%',
            'avg_r':   f'{np.mean(total_rewards):.3f}',
        })

    pbar.close()
    env.close()

    success_rate = 100.0 * successes / n_episodes

    print("\n" + "=" * 60)
    print("PPO BASELINE — EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes:      {n_episodes}")
    print(f"  Successes:     {successes}/{n_episodes}")
    print(f"  Success rate:  {success_rate:.2f}%")
    print(f"  Avg reward:    {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
    print(f"  Avg ep length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 60)

    return {
        'success_rate': success_rate,
        'successes':    successes,
        'n_episodes':   n_episodes,
        'avg_reward':   float(np.mean(total_rewards)),
        'std_reward':   float(np.std(total_rewards)),
        'avg_length':   float(np.mean(episode_lengths)),
        'std_length':   float(np.std(episode_lengths)),
    }


# ============================================================================
# Comparison summary
# ============================================================================

def print_comparison(rules_results: dict, ppo_results: dict):
    diff = rules_results['success_rate'] - ppo_results['success_rate']
    print("\n" + "=" * 60)
    print("COMPARISON: RULES vs PPO")
    print("=" * 60)
    print(f"  {'Metric':<22} {'Rules':>10} {'PPO':>10} {'Diff':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Success rate (%)':22} {rules_results['success_rate']:>10.2f} "
          f"{ppo_results['success_rate']:>10.2f} {diff:>+10.2f}")
    print(f"  {'Avg reward':22} {rules_results['avg_reward']:>10.4f} "
          f"{ppo_results['avg_reward']:>10.4f} "
          f"{rules_results['avg_reward']-ppo_results['avg_reward']:>+10.4f}")
    print(f"  {'Avg ep length':22} {rules_results['avg_length']:>10.1f} "
          f"{ppo_results['avg_length']:>10.1f} "
          f"{rules_results['avg_length']-ppo_results['avg_length']:>+10.1f}")
    print("=" * 60)

    if diff >= 0:
        print(f"  Rules agent matches or exceeds PPO by {diff:+.2f}%")
    else:
        print(f"  Rules agent is {abs(diff):.2f}% below PPO — "
              f"check action coverage and class weights")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAELogicAgentV2 logic rules by playing MiniGrid"
    )
    parser.add_argument("--model_path",  type=str,
                        default="./sae_logic_v2_outputs/sae_logic_v2_model.pt",
                        help="Path to trained SAELogicAgentV2 checkpoint")
    parser.add_argument("--ppo_path",    type=str,
                        default="ppo_doorkey_5x5.zip",
                        help="Path to PPO model (for feature extraction and baseline)")
    parser.add_argument("--env_name",    type=str,
                        default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--n_episodes",  type=int,  default=100)
    parser.add_argument("--max_steps",   type=int,  default=500,
                        help="Max steps per episode before forced termination")
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--compare_ppo", action="store_true",
                        help="Also run PPO baseline for comparison")
    parser.add_argument("--render",      action="store_true",
                        help="Render episodes visually")
    parser.add_argument("--print_rules", action="store_true",
                        help="Print learned rules before evaluation")
    parser.add_argument("--verbose_failures", action="store_true",
                        help="Print last 10 actions of failed episodes")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load rules agent
    agent = load_rules_agent(args.model_path, args.ppo_path, device)

    if args.print_rules:
        agent.print_rules()

    # Evaluate rules agent
    rules_results = evaluate_rules_agent(
        agent,
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        render=args.render,
        verbose_failures=args.verbose_failures,
    )

    # Optionally compare with PPO
    if args.compare_ppo:
        ppo_results = evaluate_ppo_baseline(
            agent.ppo_model,
            env_name=args.env_name,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        print_comparison(rules_results, ppo_results)


if __name__ == "__main__":
    main()