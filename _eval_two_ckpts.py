"""Evaluate two section001 checkpoints and compare."""
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Register environment
sys.path.insert(0, str(Path(__file__).resolve().parent / "starter_kit" / "navigation1"))
import vbot

from motrix_envs import registry as env_registry
from motrix_rl import registry as rl_registry
from motrix_rl.skrl.torch.train.ppo import Trainer, _get_cfg
from motrix_rl.skrl.torch import wrap_env
from skrl.utils import set_seed
from skrl import config

config.torch.backend = "torch"

ENV_NAME = "vbot_navigation_section001"
NUM_ENVS = 2048
MAX_STEPS = 700
NUM_TRIALS = 1
SEED = 42

CKPTS = {
    "best_agent": r"D:\MotrixLab\starter_kit_schedule\checkpoints\vbot_navigation_section001\best_agent.pt",
    "stage3_continue_1600": r"D:\MotrixLab\starter_kit_schedule\checkpoints\vbot_navigation_section001\stage3_continue_agent1600_reached100_4608.pt",
}

def evaluate_checkpoint(name, ckpt_path):
    print(f"\n{'='*70}")
    print(f"Evaluating: {name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Envs: {NUM_ENVS}, Trials: {NUM_TRIALS}, Max steps: {MAX_STEPS}")
    print(f"{'='*70}")

    trainer = Trainer(ENV_NAME, sim_backend=None, enable_render=False,
                      cfg_override={"play_num_envs": NUM_ENVS, "seed": SEED})
    rlcfg = trainer._rlcfg

    env = env_registry.make(ENV_NAME, sim_backend=None, num_envs=NUM_ENVS)
    set_seed(SEED)
    skrl_env = wrap_env(env, False)
    models = trainer._make_model(skrl_env, rlcfg)
    ppo_cfg = _get_cfg(rlcfg, skrl_env)
    agent = trainer._make_agent(models, skrl_env, ppo_cfg)
    agent.load(ckpt_path)

    all_reached = []
    all_distances = []
    all_ep_lens = []
    all_min_dists_reached = []

    for trial in range(NUM_TRIALS):
        trial_seed = SEED + trial * 100
        np.random.seed(trial_seed)

        obs, info = skrl_env.reset()

        steps_done = np.zeros(NUM_ENVS, dtype=np.int32)
        episode_reached = np.zeros(NUM_ENVS, dtype=bool)
        episode_min_dist = np.full(NUM_ENVS, 999.0, dtype=np.float32)
        episode_lens = np.zeros(NUM_ENVS, dtype=np.int32)
        completed = np.zeros(NUM_ENVS, dtype=bool)
        total_reward = np.zeros(NUM_ENVS, dtype=np.float32)

        for step in range(MAX_STEPS):
            with torch.inference_mode():
                outputs = agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])

            obs, reward, terminated, truncated, infos = skrl_env.step(actions)
            steps_done += 1
            total_reward += reward.cpu().numpy().flatten()

            # Extract metrics from raw env
            raw_info = env._state.info
            if "metrics" in raw_info:
                metrics = raw_info["metrics"]
                dist = metrics.get("distance_to_target", np.full(NUM_ENVS, 999.0))
                reached = metrics.get("reached_fraction", np.zeros(NUM_ENVS)) > 0.5

                active = ~completed
                episode_min_dist = np.where(active & (dist < episode_min_dist), dist, episode_min_dist)
                episode_reached = np.where(active & reached, True, episode_reached)

            done = (terminated | truncated).cpu().numpy().flatten().astype(bool)
            newly_done = done & ~completed
            episode_lens = np.where(newly_done, steps_done, episode_lens)
            completed = completed | done

        # Mark still-running episodes
        episode_lens = np.where(~completed, MAX_STEPS, episode_lens)

        reached_frac = episode_reached.mean()
        unreached_mask = ~episode_reached
        avg_dist_unreached = episode_min_dist[unreached_mask].mean() if unreached_mask.any() else 0.0
        avg_dist_reached = episode_min_dist[episode_reached].mean() if episode_reached.any() else 999.0
        avg_ep_len = episode_lens.mean()
        avg_reward = total_reward.mean()

        print(f"  Trial {trial+1}/{NUM_TRIALS} (seed={trial_seed}):")
        print(f"    reached: {reached_frac:.4f} ({episode_reached.sum()}/{NUM_ENVS})")
        print(f"    avg_min_dist (unreached): {avg_dist_unreached:.3f}m")
        print(f"    avg_min_dist (reached):   {avg_dist_reached:.3f}m")
        print(f"    avg_ep_len: {avg_ep_len:.0f} steps")
        print(f"    avg_reward: {avg_reward:.1f}")

        all_reached.append(reached_frac)
        all_distances.append(avg_dist_unreached)
        all_ep_lens.append(avg_ep_len)
        all_min_dists_reached.append(avg_dist_reached)

    summary = {
        "name": name,
        "reached_mean": float(np.mean(all_reached)),
        "reached_std": float(np.std(all_reached)),
        "dist_unreached_mean": float(np.mean(all_distances)),
        "dist_reached_mean": float(np.mean(all_min_dists_reached)),
        "ep_len_mean": float(np.mean(all_ep_lens)),
    }

    print(f"\n  SUMMARY ({NUM_TRIALS} trials):")
    print(f"    reached:   {summary['reached_mean']:.4f} ± {summary['reached_std']:.4f}")
    print(f"    dist (unreached): {summary['dist_unreached_mean']:.3f}m")
    print(f"    dist (reached):   {summary['dist_reached_mean']:.3f}m")
    print(f"    ep_len:    {summary['ep_len_mean']:.0f}")

    return summary

if __name__ == "__main__":
    results = {}
    for name, path in CKPTS.items():
        results[name] = evaluate_checkpoint(name, path)

    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'best_agent':>18} {'stage3_cont_1600':>18}")
    print("-" * 70)
    r1 = results["best_agent"]
    r2 = results["stage3_continue_1600"]
    print(f"{'Reached %':<30} {r1['reached_mean']*100:>17.2f}% {r2['reached_mean']*100:>17.2f}%")
    print(f"{'Reached ± std':<30} {r1['reached_std']*100:>17.2f}% {r2['reached_std']*100:>17.2f}%")
    print(f"{'Dist unreached (m)':<30} {r1['dist_unreached_mean']:>18.3f} {r2['dist_unreached_mean']:>18.3f}")
    print(f"{'Dist reached (m)':<30} {r1['dist_reached_mean']:>18.3f} {r2['dist_reached_mean']:>18.3f}")
    print(f"{'Ep len (steps)':<30} {r1['ep_len_mean']:>18.0f} {r2['ep_len_mean']:>18.0f}")

    # Determine winner
    if r1['reached_mean'] > r2['reached_mean'] + 0.01:
        winner = "best_agent"
    elif r2['reached_mean'] > r1['reached_mean'] + 0.01:
        winner = "stage3_continue_1600"
    else:
        # Tie on reached �?compare distance
        if r1['dist_unreached_mean'] < r2['dist_unreached_mean']:
            winner = "best_agent"
        else:
            winner = "stage3_continue_1600"

    print(f"\n  WINNER: {winner}")

