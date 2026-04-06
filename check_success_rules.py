"""
check_success_rules.py
======================
Evaluate the SAELogicAgentV3 (trained logic rules) by playing MiniGrid
or Atari environments and measuring success rate. Compares against the
original PPO baseline.

Usage (MiniGrid):
    python check_success_rules.py \
        --model_path ./sae_logic_v3_outputs/sae_logic_v3_model.pt \
        --ppo_path   ppo_doorkey_5x5.zip \
        --n_episodes 100

Usage (Atari):
    python check_success_rules.py \
        --model_path ./sae_logic_v3_outputs/sae_logic_v3_model.pt \
        --ppo_path   ppo_atari_breakout.zip \
        --n_episodes 100 --print_rules

Usage (with PPO comparison):
    python check_success_rules.py \
        --model_path ./sae_logic_v3_outputs/sae_logic_v3_model.pt \
        --ppo_path   ppo_atari_breakout.zip \
        --compare_ppo --n_episodes 100
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from dataclasses import asdict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from tqdm import tqdm

from train_sae_logic import SAELogicAgentV3, SAELogicConfig, get_action_names


# ============================================================================
# Environment detection
# ============================================================================

def _get_env_name_from_ppo(ppo_model: PPO) -> str:
    """Try to extract the environment name from a loaded PPO model."""
    # SB3 stores env metadata in several places
    try:
        # Method 1: get_env() wrapper
        env = ppo_model.get_env()
        if env is not None:
            spec = getattr(env, 'spec', None)
            if spec and hasattr(spec, 'id'):
                return spec.id
    except Exception:
        pass

    try:
        # Method 2: model's stored env name in _custom_objects or kwargs
        if hasattr(ppo_model, '_custom_objects') and ppo_model._custom_objects:
            env_name = ppo_model._custom_objects.get('env_name', None)
            if env_name:
                return env_name
    except Exception:
        pass

    return None


def detect_env_type(env_name: str) -> str:
    """Infer env type from the environment name string."""
    name_lower = env_name.lower()
    if "minigrid" in name_lower:
        return "minigrid"
    # Common Atari env name patterns
    atari_keywords = [
        "breakout", "pong", "spaceinvaders", "seaquest", "qbert",
        "mspacman", "asteroids", "beamrider", "enduro", "freeway",
        "frostbite", "montezuma", "pitfall", "riverraid", "tennis",
        "assault", "atlantis", "bankheist", "battlezone", "bowling",
        "boxing", "centipede", "demonattack", "gravitar", "hero",
        "icehockey", "jamesbond", "kangaroo", "krull", "kungfumaster",
        "phoenix", "robotank", "skiing", "solaris", "venture",
        "videopinball", "wizardofwor", "zaxxon",
        "atari", "ale/", "noframeskip", "deterministic",
    ]
    if any(kw in name_lower for kw in atari_keywords):
        return "atari"
    return "unknown"


# ============================================================================
# Rules-based agent wrapper
# ============================================================================

class RulesAgent:
    """
    Wraps SAELogicAgentV3 for game-play evaluation.

    Pipeline per step:
        raw obs → PPO feature extractor → normalize_input → SAE encode
                → fixed z-norm → sigmoid bottleneck → product t-norm logic → action
    """

    def __init__(
        self,
        ppo_model: PPO,
        logic_model: SAELogicAgentV3,
        device: str = "cpu",
    ):
        self.ppo_model   = ppo_model
        self.logic_model = logic_model
        self.device      = device

        self.logic_model.eval()

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Args:
            obs: raw observation from VecEnv, shape (1, C, H, W)

        Returns:
            action array of shape (1,)
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float().to(self.device)

            # PPO CNN feature extractor (frozen, same as training)
            features = self.ppo_model.policy.features_extractor(obs_tensor)

            # Normalize using Stage 1 stats stored in the logic model
            features_norm = self.logic_model.normalize_input(features)

            # SAE → fixed z-norm → bottleneck → logic → action
            action_logits = self.logic_model(features_norm)
            action = action_logits.argmax(dim=-1)

        return action.cpu().numpy()

    def print_rules(self, action_names=None):
        """Print the learned rules being used."""
        if action_names is None:
            action_names = get_action_names(
                self.logic_model.config.n_actions,
                env_type=detect_env_type(self.logic_model.config.env_name),
            )
        rules = self.logic_model.extract_rules(action_names=action_names, threshold=0.5)
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


# ============================================================================
# Model loading
# ============================================================================

def load_rules_agent(model_path: str, ppo_path: str, device: str) -> tuple:
    """
    Load SAELogicAgentV3 checkpoint and PPO feature extractor.

    Returns:
        (RulesAgent, env_name, env_type, action_names)
    """
    print(f"Loading PPO from {ppo_path}...")
    ppo_model = PPO.load(ppo_path, device=device)

    print(f"Loading logic model from {model_path}...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Handle tuple fields stored as lists in the checkpoint
    config_dict = ckpt['config']
    if 'action_class_weights' in config_dict and isinstance(config_dict['action_class_weights'], list):
        config_dict['action_class_weights'] = tuple(config_dict['action_class_weights'])

    config = SAELogicConfig(**config_dict)
    logic_model = SAELogicAgentV3(config, device=device)
    logic_model.load_state_dict(ckpt['model_state'])
    logic_model.to(device)
    logic_model.eval()

    # Restore input normalization stats
    feat_mean = ckpt['feature_mean'].to(device)
    feat_std  = ckpt['feature_std'].to(device)
    logic_model.set_normalization(feat_mean, feat_std)

    # Restore SAE activation normalization stats (V3-specific)
    if 'z_mean' in ckpt and 'z_std' in ckpt:
        logic_model.z_mean.copy_(ckpt['z_mean'].to(device))
        logic_model.z_std.copy_(ckpt['z_std'].to(device))

    # Resolve env metadata from checkpoint
    env_name = ckpt.get('env_name', config.env_name)
    env_type = ckpt.get('env_type', config.env_type)
    action_names_stored = ckpt.get('action_names', None)
    action_names = get_action_names(config.n_actions, action_names_stored, env_type)

    print(f"  Val accuracy at save: {ckpt['best_val_acc']:.3f}")
    print(f"  Env: {env_name} (type={env_type})")
    print(f"  Actions: {action_names}")
    print(f"  Architecture: {config.input_dim}d → SAE({config.hidden_dim}, k={config.k})"
          f" → FixedNorm → Sigmoid → Logic({config.n_clauses_per_action} clauses/action)"
          f" → {config.n_actions} actions")

    agent = RulesAgent(ppo_model, logic_model, device)
    return agent, env_name, env_type, action_names


# ============================================================================
# Environment factory
# ============================================================================

def make_vec_env(env_name: str, env_type: str, seed: int, render: bool = False):
    """
    Create a VecEnv with the appropriate wrappers for MiniGrid or Atari.
    """
    if env_type == "minigrid":
        import minigrid  # noqa: F401
        from minigrid.wrappers import ImgObsWrapper

        def _make():
            def _init():
                e = gym.make(env_name, render_mode="human" if render else None)
                return ImgObsWrapper(e)
            return _init

        env = DummyVecEnv([_make()])
        env = VecTransposeImage(env)
        env.seed(seed)
        return env

    elif env_type == "atari":
        import ale_py  # noqa: F401
        gym.register_envs(ale_py)
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecFrameStack

        env = make_atari_env(
            env_name,
            n_envs=1,
            seed=seed,
        )
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
        return env

    else:
        # Generic fallback: try to create without special wrappers
        print(f"  [Warning] Unknown env_type '{env_type}', using generic wrapper.")
        def _make():
            def _init():
                return gym.make(env_name, render_mode="human" if render else None)
            return _init

        env = DummyVecEnv([_make()])
        # Only transpose if obs is image-like (4D)
        sample = env.observation_space.shape
        if len(sample) == 3:
            env = VecTransposeImage(env)
        env.seed(seed)
        return env


# ============================================================================
# Success criteria per env type
# ============================================================================

def is_success(reward: float, info: dict, env_type: str) -> bool:
    """
    Determine whether an episode was successful.

    - MiniGrid: final reward > 0 means the agent reached the goal.
    - Atari: any positive total episode reward counts as success.
      (This is checked at episode end using cumulative reward.)
    """
    if env_type == "minigrid":
        return reward > 0
    # For Atari, success is determined by total episode reward (checked externally)
    return False  # placeholder — Atari uses cumulative reward


# ============================================================================
# Evaluation loop
# ============================================================================

def evaluate_rules_agent(
    agent: RulesAgent,
    env_name: str,
    env_type: str,
    action_names: list,
    n_episodes: int = 100,
    max_steps: int = 500,
    seed: int = 42,
    render: bool = False,
    verbose_failures: bool = False,
) -> dict:
    """
    Play n_episodes and collect success rate, reward, and episode length.

    Success criteria:
      - MiniGrid: episode ends with reward > 0
      - Atari: total episode reward > 0
    """
    env = make_vec_env(env_name, env_type, seed, render)

    successes       = 0
    total_rewards   = []
    episode_lengths = []
    action_counts   = np.zeros(len(action_names), dtype=int)

    # Atari may report episode stats through info['episode']
    is_atari = (env_type == "atari")

    pbar = tqdm(total=n_episodes, desc="Rules agent")

    ep = 0
    obs = env.reset()
    ep_reward  = 0.0
    ep_length  = 0
    ep_actions = []

    while ep < n_episodes:
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)

        ep_reward += reward[0]
        ep_length += 1
        act_idx = int(action[0])
        if act_idx < len(action_names):
            ep_actions.append(act_idx)
            action_counts[act_idx] += 1

        if done[0]:
            # Determine success
            if is_atari:
                # For Atari with SB3 wrappers, true episode reward may be in info
                true_reward = ep_reward
                if isinstance(info, (list, tuple)) and len(info) > 0:
                    ep_info = info[0].get('episode', None)
                    if ep_info is not None:
                        true_reward = ep_info.get('r', ep_reward)
                if true_reward > 0:
                    successes += 1
            else:
                # MiniGrid: final step reward > 0
                if reward[0] > 0:
                    successes += 1
                elif verbose_failures:
                    last_acts = [action_names[a] for a in ep_actions[-10:]]
                    print(f"\n  [Ep {ep+1}] FAILED — last 10 actions: {last_acts}")

            total_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            ep += 1
            pbar.update(1)
            pbar.set_postfix({
                'success': f'{100.0 * successes / ep:.1f}%',
                'avg_r':   f'{np.mean(total_rewards):.3f}',
                'avg_len': f'{np.mean(episode_lengths):.1f}',
            })

            # Reset episode tracking (env auto-resets in VecEnv)
            ep_reward  = 0.0
            ep_length  = 0
            ep_actions = []

        # Safety: for non-VecEnv auto-reset setups, cap steps
        if not done[0] and ep_length >= max_steps:
            total_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            ep += 1
            pbar.update(1)
            pbar.set_postfix({
                'success': f'{100.0 * successes / ep:.1f}%',
                'avg_r':   f'{np.mean(total_rewards):.3f}',
                'avg_len': f'{np.mean(episode_lengths):.1f}',
            })
            obs = env.reset()
            ep_reward  = 0.0
            ep_length  = 0
            ep_actions = []

    pbar.close()
    env.close()

    success_rate = 100.0 * successes / n_episodes

    print("\n" + "=" * 60)
    print("RULES AGENT — EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Environment:      {env_name} ({env_type})")
    print(f"  Episodes:         {n_episodes}")
    print(f"  Successes:        {successes}/{n_episodes}")
    print(f"  Success rate:     {success_rate:.2f}%")
    print(f"  Avg reward:       {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
    print(f"  Avg ep length:    {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"\n  Action usage during play:")
    total_actions = action_counts.sum()
    for name, cnt in zip(action_names, action_counts):
        bar = "█" * int(40 * cnt / max(total_actions, 1))
        print(f"    {name:14s}: {cnt:6d} ({100.0*cnt/max(total_actions,1):5.1f}%) {bar}")
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
    env_type: str,
    n_episodes: int = 100,
    max_steps: int = 500,
    seed: int = 42,
) -> dict:
    """Evaluate original PPO policy as baseline."""
    env = make_vec_env(env_name, env_type, seed)

    is_atari = (env_type == "atari")

    successes       = 0
    total_rewards   = []
    episode_lengths = []

    pbar = tqdm(total=n_episodes, desc="PPO baseline")

    ep = 0
    obs = env.reset()
    ep_reward = 0.0
    ep_length = 0

    while ep < n_episodes:
        action, _ = ppo_model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)

        ep_reward += reward[0]
        ep_length += 1

        if done[0]:
            if is_atari:
                true_reward = ep_reward
                if isinstance(info, (list, tuple)) and len(info) > 0:
                    ep_info = info[0].get('episode', None)
                    if ep_info is not None:
                        true_reward = ep_info.get('r', ep_reward)
                if true_reward > 0:
                    successes += 1
            else:
                if reward[0] > 0:
                    successes += 1

            total_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            ep += 1
            pbar.update(1)
            pbar.set_postfix({
                'success': f'{100.0 * successes / ep:.1f}%',
                'avg_r':   f'{np.mean(total_rewards):.3f}',
            })
            ep_reward = 0.0
            ep_length = 0

        if not done[0] and ep_length >= max_steps:
            total_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            ep += 1
            pbar.update(1)
            pbar.set_postfix({
                'success': f'{100.0 * successes / ep:.1f}%',
                'avg_r':   f'{np.mean(total_rewards):.3f}',
            })
            obs = env.reset()
            ep_reward = 0.0
            ep_length = 0

    pbar.close()
    env.close()

    success_rate = 100.0 * successes / n_episodes

    print("\n" + "=" * 60)
    print("PPO BASELINE — EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Environment:   {env_name} ({env_type})")
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
        description="Evaluate SAELogicAgentV3 logic rules by playing MiniGrid or Atari",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MiniGrid
  python check_success_rules.py \\
      --model_path ./sae_logic_v3_outputs/sae_logic_v3_model.pt \\
      --ppo_path ppo_doorkey_5x5.zip \\
      --n_episodes 100

  # Atari Breakout
  python check_success_rules.py \\
      --model_path ./sae_logic_v3_outputs/sae_logic_v3_model.pt \\
      --ppo_path ppo_atari_breakout.zip \\
      --n_episodes 100 --print_rules

  # Override auto-detected env (if checkpoint lacks metadata)
  python check_success_rules.py \\
      --model_path ./model.pt \\
      --ppo_path ppo.zip \\
      --env_name BreakoutNoFrameskip-v4 \\
      --env_type atari
        """
    )
    parser.add_argument("--model_path",  type=str,
                        default="./sae_logic_v3_outputs/sae_logic_v3_model.pt",
                        help="Path to trained SAELogicAgentV3 checkpoint")
    parser.add_argument("--ppo_path",    type=str,
                        default="ppo_doorkey_5x5.zip",
                        help="Path to PPO model (for feature extraction and baseline)")
    parser.add_argument("--env_name",    type=str, default=None,
                        help="Override env name (auto-detected from checkpoint if not set)")
    parser.add_argument("--env_type",    type=str, default=None,
                        choices=["minigrid", "atari"],
                        help="Override env type (auto-detected from checkpoint if not set)")
    parser.add_argument("--n_episodes",  type=int,  default=100)
    parser.add_argument("--max_steps",   type=int,  default=500,
                        help="Max steps per episode before forced termination "
                             "(default 500; Atari may need more, e.g. 10000)")
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

    # Load rules agent + metadata from checkpoint
    agent, ckpt_env_name, ckpt_env_type, action_names = load_rules_agent(
        args.model_path, args.ppo_path, device
    )

    # CLI overrides take priority over checkpoint metadata
    env_name = args.env_name or ckpt_env_name
    env_type = args.env_type or ckpt_env_type

    # If checkpoint had "unknown", try to extract env name from PPO model
    if env_name in (None, "unknown", ""):
        ppo_env_name = _get_env_name_from_ppo(agent.ppo_model)
        if ppo_env_name:
            env_name = ppo_env_name
            print(f"  [Info] Detected env from PPO model: {env_name}")

    # Validate env_name before proceeding
    if env_name in (None, "unknown", ""):
        parser.error(
            "Could not determine environment name from checkpoint or PPO model.\n"
            "Please specify --env_name explicitly, e.g.:\n"
            "  --env_name MiniGrid-DoorKey-5x5-v0\n"
            "  --env_name BreakoutNoFrameskip-v4"
        )

    # If env_type is still unknown, try to detect from env_name
    if env_type in (None, "unknown", ""):
        env_type = detect_env_type(env_name)
        if env_type == "unknown":
            print(f"  [Warning] Could not detect env type for '{env_name}'. "
                  f"Use --env_type to specify. Defaulting to 'minigrid'.")
            env_type = "minigrid"

    # For Atari, bump default max_steps if user didn't explicitly set it
    if env_type == "atari" and args.max_steps == 500:
        args.max_steps = 10000
        print(f"  [Info] Atari detected — using max_steps={args.max_steps}")

    print(f"\n  Environment: {env_name}")
    print(f"  Type:        {env_type}")
    print(f"  Actions:     {action_names}")

    if args.print_rules:
        agent.print_rules(action_names=action_names)

    # Evaluate rules agent
    rules_results = evaluate_rules_agent(
        agent,
        env_name=env_name,
        env_type=env_type,
        action_names=action_names,
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
            env_name=env_name,
            env_type=env_type,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        print_comparison(rules_results, ppo_results)


if __name__ == "__main__":
    main()