#!/usr/bin/env python3
"""
Checkpoint Evaluator & Ranker
==============================
Headless multi-trial evaluation of trained checkpoints + checkpoint ranking by metric.

Replaces: _eval_section011.py, _eval_stage3.py, _evaluate_checkpoint.py, _find_best_checkpoint.py

Usage:
    # Evaluate a checkpoint (multi-trial, headless)
    uv run starter_kit_schedule/scripts/eval_checkpoint.py --env vbot_navigation_section011 --ckpt path/to/agent.pt

    # Evaluate with custom settings
    uv run starter_kit_schedule/scripts/eval_checkpoint.py --env vbot_navigation_section011 --ckpt path/to/agent.pt --num-trials 5 --max-steps 3000

    # Rank checkpoints in a run by best metric
    uv run starter_kit_schedule/scripts/eval_checkpoint.py --rank runs/vbot_navigation_section011/26-02-12_PPO --top 10

    # Find best checkpoint across all runs for an env
    uv run starter_kit_schedule/scripts/eval_checkpoint.py --rank-env vbot_navigation_section011 --top 5
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]  # d:\MotrixLab

# Environment import table — maps env name to its registration module
ENV_IMPORTS = {
    "vbot_navigation_section001": ("starter_kit.navigation1", "vbot"),
    "vbot_navigation_section011": ("starter_kit.navigation2", "vbot"),
    "vbot_navigation_section012": ("starter_kit.navigation2", "vbot"),
    "vbot_navigation_section013": ("starter_kit.navigation2", "vbot"),
    "vbot_navigation_long_course": ("starter_kit.navigation2", "vbot"),
}


def _register_env(env_name: str):
    """Register environment by importing its module."""
    if env_name in ENV_IMPORTS:
        pkg, mod = ENV_IMPORTS[env_name]
        parent = str(WORKSPACE / pkg.replace(".", "/").rsplit(".", 1)[0]) if "." in pkg else str(WORKSPACE)
        # For starter_kit paths, add the parent to sys.path
        kit_path = str(WORKSPACE / pkg.replace(".", "/"))
        parent_path = str(Path(kit_path).parent)
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)
        __import__(f"{pkg.split('.')[-1]}.{mod}")
    else:
        # Try generic import
        import importlib
        try:
            importlib.import_module(env_name)
        except ImportError:
            print(f"WARNING: Could not register env '{env_name}'. Ensure it's imported.")


# ============================================================
# Checkpoint Evaluation
# ============================================================

def evaluate(env_name: str, ckpt_path: str, num_envs: int = 2048,
             num_trials: int = 3, max_steps: int = 2000, seed: int = 42):
    """Run headless multi-trial evaluation and print metrics."""
    import torch
    from motrix_envs import registry

    _register_env(env_name)
    env = registry.make(env_name, num_envs=num_envs)

    # Build agent from RL config
    from motrix_rl.skrl.torch.train.ppo import build_agent
    from motrix_rl.registry import default_rl_cfg

    rlcfg = default_rl_cfg(env_name, "ppo", "torch")
    agent = build_agent(env, rlcfg)
    agent.load(ckpt_path)
    agent.set_running_mode("eval")

    print(f"=== Evaluation: {env_name} ===")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Envs: {num_envs}, Trials: {num_trials}, Max steps: {max_steps}")
    print()

    all_reached = []
    all_distances = []
    all_ep_lens = []
    all_wp_idx = []

    for trial in range(num_trials):
        trial_seed = seed + trial * 100
        np.random.seed(trial_seed)

        obs, info = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device="cuda")

        steps_done = np.zeros(num_envs, dtype=np.int32)
        episode_reached = np.zeros(num_envs, dtype=bool)
        episode_min_dist = np.full(num_envs, 999.0, dtype=np.float32)
        episode_max_wp = np.zeros(num_envs, dtype=np.float32)
        episode_lens = np.zeros(num_envs, dtype=np.int32)
        completed = np.zeros(num_envs, dtype=bool)

        for step in range(max_steps):
            with torch.no_grad():
                action = agent.act(obs_tensor, timestep=0, timesteps=0)[0]
            action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action)

            obs, reward, terminated, truncated, info = env.step(action_np)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device="cuda")
            steps_done += 1

            # Track metrics
            if "metrics" in info:
                metrics = info["metrics"]
                dist = metrics.get("distance_to_target", np.full(num_envs, 999.0))
                reached = metrics.get("reached_fraction", np.zeros(num_envs)) > 0.5
                wp_idx = metrics.get("wp_idx_mean", np.zeros(num_envs))

                active = ~completed
                episode_min_dist = np.where(active & (dist < episode_min_dist), dist, episode_min_dist)
                episode_reached = np.where(active & reached, True, episode_reached)
                episode_max_wp = np.where(active & (wp_idx > episode_max_wp), wp_idx, episode_max_wp)

            done = terminated | truncated
            newly_done = done & ~completed
            episode_lens = np.where(newly_done, steps_done, episode_lens)
            completed = completed | done

        # Mark still-running episodes
        episode_lens = np.where(~completed, max_steps, episode_lens)

        reached_frac = episode_reached.mean()
        avg_dist = episode_min_dist[~episode_reached].mean() if (~episode_reached).any() else 0.0
        avg_ep_len = episode_lens.mean()
        avg_wp = episode_max_wp.mean()

        print(f"Trial {trial+1}/{num_trials} (seed={trial_seed}):")
        print(f"  reached: {reached_frac:.4f} ({episode_reached.sum()}/{num_envs})")
        print(f"  avg_min_dist (non-reached): {avg_dist:.3f}m")
        print(f"  avg_wp_idx: {avg_wp:.3f}")
        print(f"  avg_ep_len: {avg_ep_len:.0f} steps")
        print()

        all_reached.append(reached_frac)
        all_distances.append(avg_dist)
        all_ep_lens.append(avg_ep_len)
        all_wp_idx.append(avg_wp)

    env.close()

    print("=" * 60)
    print(f"SUMMARY ({num_trials} trials):")
    print(f"  reached: {np.mean(all_reached):.4f} ± {np.std(all_reached):.4f}")
    print(f"  avg_min_dist: {np.mean(all_distances):.3f}m")
    print(f"  avg_wp_idx: {np.mean(all_wp_idx):.3f}")
    print(f"  avg_ep_len: {np.mean(all_ep_lens):.0f}")

    return {
        "reached_mean": float(np.mean(all_reached)),
        "reached_std": float(np.std(all_reached)),
        "distance_mean": float(np.mean(all_distances)),
        "wp_idx_mean": float(np.mean(all_wp_idx)),
        "ep_len_mean": float(np.mean(all_ep_lens)),
    }


# ============================================================
# Checkpoint Ranking (via TensorBoard)
# ============================================================

def rank_checkpoints(run_dir: str, metric: str = "metrics / wp_idx_mean (mean)",
                     top_n: int = 10):
    """Rank checkpoints by a TensorBoard metric, show which ckpt files exist."""
    try:
        # Use existing evaluate.py helpers if available
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluate import read_tb_scalars
    except ImportError:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        def read_tb_scalars(run_dir, tag, max_entries=0):
            ea = EventAccumulator(str(run_dir))
            ea.Reload()
            try:
                events = ea.Scalars(tag)
                return [(e.step, e.value) for e in events]
            except Exception:
                return []

    data = read_tb_scalars(run_dir, metric)
    if not data:
        # Try fuzzy match
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(run_dir))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        metric_lower = metric.lower()
        for tag in tags:
            if metric_lower in tag.lower():
                data = read_tb_scalars(run_dir, tag)
                if data:
                    print(f"Fuzzy matched: {tag}")
                    break
    if not data:
        print(f"No data for metric: {metric}")
        return []

    # Sort by value descending
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

    # Map steps to checkpoint files
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    # Also get reward and ep_len for context
    reward_data = dict(read_tb_scalars(run_dir, "Reward / Instantaneous (mean)") or
                       read_tb_scalars(run_dir, "Reward / Instantaneous reward (mean)") or [])
    ep_data = dict(read_tb_scalars(run_dir, "Episode / Total length (mean)") or
                   read_tb_scalars(run_dir, "Episode / Total timesteps (mean)") or [])

    print(f"\n{'='*75}")
    print(f"Best Checkpoints: {os.path.basename(run_dir)}")
    print(f"Metric: {metric}")
    print(f"{'='*75}")
    print(f"\n{'Rank':<6} {'Step':>8} {'Metric':>10} {'Reward':>10} {'EpLen':>8} {'Status':>10}")
    print("-" * 60)

    results = []
    for i, (step, value) in enumerate(sorted_data[:top_n], 1):
        ckpt_path = os.path.join(ckpt_dir, f"agent_{step}.pt")
        exists = os.path.exists(ckpt_path)
        reward = reward_data.get(step, 0)
        ep_len = ep_data.get(step, 0)
        status = "EXISTS" if exists else "MISSING"
        print(f"{i:<6} {step:>8} {value:>10.4f} {reward:>10.2f} {ep_len:>8.0f} {status:>10}")
        results.append((step, value, ckpt_path, exists))

    # Show best existing checkpoint
    for step, value, path, exists in results:
        if exists:
            print(f"\nBest available checkpoint: {path}")
            print(f"  Step {step}, {metric.split('/')[-1].strip()} = {value:.4f}")
            break

    return results


def rank_env_checkpoints(env_name: str, top_n: int = 5):
    """Find the best checkpoint across all runs for an environment."""
    runs_dir = WORKSPACE / "runs" / env_name
    if not runs_dir.exists():
        print(f"No runs directory: {runs_dir}")
        return

    run_dirs = sorted(runs_dir.iterdir(), key=lambda d: d.name)
    print(f"Scanning {len(list(run_dirs))} runs for {env_name}...")

    all_results = []
    for run_dir in sorted(runs_dir.iterdir(), key=lambda d: d.name):
        if not run_dir.is_dir():
            continue
        results = rank_checkpoints(str(run_dir), top_n=1)
        if results:
            step, value, path, exists = results[0]
            all_results.append((value, str(run_dir), step, path, exists))

    if all_results:
        all_results.sort(key=lambda x: x[0], reverse=True)
        print(f"\n{'='*75}")
        print(f"TOP {top_n} CHECKPOINTS ACROSS ALL RUNS")
        print(f"{'='*75}")
        for i, (value, run_dir, step, path, exists) in enumerate(all_results[:top_n], 1):
            run_name = os.path.basename(run_dir)
            status = "EXISTS" if exists else "MISSING"
            print(f"  {i}. {run_name} step={step} metric={value:.4f} [{status}]")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint Evaluator & Ranker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a checkpoint
  uv run starter_kit_schedule/scripts/eval_checkpoint.py --env vbot_navigation_section011 --ckpt path/to/agent.pt

  # Rank checkpoints in a run
  uv run starter_kit_schedule/scripts/eval_checkpoint.py --rank runs/.../26-02-12_PPO

  # Find best across all runs
  uv run starter_kit_schedule/scripts/eval_checkpoint.py --rank-env vbot_navigation_section011
        """
    )
    parser.add_argument("--env", type=str, help="Environment name")
    parser.add_argument("--ckpt", type=str, help="Checkpoint path for evaluation")
    parser.add_argument("--rank", type=str, metavar="RUN_DIR", help="Rank checkpoints in a specific run")
    parser.add_argument("--rank-env", type=str, metavar="ENV_NAME", help="Rank best checkpoints across all runs")
    parser.add_argument("--metric", type=str, default="metrics / wp_idx_mean (mean)",
                        help="Metric for ranking (default: wp_idx_mean)")
    parser.add_argument("--top", type=int, default=10, help="Number of top results to show")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rank:
        rank_checkpoints(args.rank, metric=args.metric, top_n=args.top)
    elif args.rank_env:
        rank_env_checkpoints(args.rank_env, top_n=args.top)
    elif args.ckpt:
        if not args.env:
            parser.error("--env is required for evaluation")
        evaluate(args.env, args.ckpt, args.num_envs, args.num_trials, args.max_steps, args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
