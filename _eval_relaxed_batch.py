"""
Batch headless evaluation of Stage 2 T0-T7 checkpoints with relaxed termination.
Monkey-patches termination to remove base_contact, soft_tilt, stagnation — keeps only
extreme physics failures (85° tilt, OOB, NaN, joint explosions).

Does NOT modify any existing files or AutoML state.
"""
import sys
import os
import json
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent

# --- Register environment ---
sys.path.insert(0, str(WORKSPACE / "starter_kit" / "navigation2"))
import vbot as navigation2_vbot  # noqa: F401, E402

from motrix_envs import registry as env_registry

# --- Monkey-patch termination ---
_meta = env_registry._envs.get("vbot_navigation_section011")
env_cls = _meta.env_cls_dict.get("np")

_original_compute_terminated = env_cls._compute_terminated


def _relaxed_compute_terminated(self, state, projected_gravity=None, joint_vel=None, robot_xy=None, current_z=None):
    """Relaxed termination: only extreme physics failures."""
    n = self._num_envs
    hard_terminated = np.zeros(n, dtype=bool)

    # Only 85° tilt (nearly upside-down)
    if projected_gravity is not None:
        gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
        gz = projected_gravity[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        hard_terminated |= tilt_angle > np.deg2rad(85)

    # OOB (competition bounds)
    bounds = getattr(self._cfg, 'course_bounds', None)
    if bounds is not None and robot_xy is not None and current_z is not None:
        oob_x = (robot_xy[:, 0] < bounds.x_min) | (robot_xy[:, 0] > bounds.x_max)
        oob_y = (robot_xy[:, 1] < bounds.y_min) | (robot_xy[:, 1] > bounds.y_max)
        oob_z = current_z < bounds.z_min
        hard_terminated |= oob_x | oob_y | oob_z
        state.info["oob_terminated"] = oob_x | oob_y | oob_z

    # Joint velocity / NaN explosions
    if joint_vel is not None:
        vel_max = np.abs(joint_vel).max(axis=1)
        hard_terminated |= vel_max > self._cfg.max_dof_vel
        hard_terminated |= np.isnan(joint_vel).any(axis=1) | np.isinf(joint_vel).any(axis=1)
        last_dof_vel = state.info.get("last_dof_vel", np.zeros_like(joint_vel))
        dof_acc_max = np.abs(joint_vel - np.clip(last_dof_vel, -100.0, 100.0)).max(axis=1)
        hard_terminated |= dof_acc_max > 80.0
    nan_terminated = state.info.get("nan_terminated", np.zeros(n, dtype=bool))
    hard_terminated |= nan_terminated

    return state.replace(terminated=hard_terminated)


env_cls._compute_terminated = _relaxed_compute_terminated

# Disable stagnation truncation
_original_update_truncate = env_cls._update_truncate


def _relaxed_update_truncate(self):
    from motrix_envs.np.env import NpEnv
    NpEnv._update_truncate(self)  # max episode length only


env_cls._update_truncate = _relaxed_update_truncate

print("[RELAXED EVAL] Termination patched: base_contact=OFF, soft_tilt=OFF, hard_tilt=85°, stagnation=OFF")

# --- Trial configs ---
AUTOML_DIR = WORKSPACE / "starter_kit_log" / "automl_20260222_124457"
EXPERIMENTS_DIR = AUTOML_DIR / "experiments"

TRIALS = {}
for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
    if not exp_dir.is_dir():
        continue
    summary_path = exp_dir / "summary.yaml"
    if not summary_path.exists():
        continue
    # Parse trial index from dir name  (... _i{N}_...)
    name = exp_dir.name
    parts = name.split("_i")
    if len(parts) >= 2:
        idx_str = parts[-1].split("_")[0]
        try:
            trial_idx = int(idx_str)
        except ValueError:
            continue
        if trial_idx > 7:
            continue
        # Read run_dir from summary
        import yaml
        with open(summary_path) as f:
            summary = yaml.safe_load(f)
        run_dir = summary.get("results", {}).get("run_dir", "")
        if not run_dir:
            run_dir = summary.get("run_dir", "")
        ckpt = WORKSPACE / run_dir / "checkpoints" / "best_agent.pt"
        if ckpt.exists():
            TRIALS[trial_idx] = {
                "run_dir": run_dir,
                "ckpt": str(ckpt),
                "original_wp": summary.get("results", {}).get("final_metrics", {}).get("wp_idx_mean", 0),
                "original_success": summary.get("results", {}).get("final_metrics", {}).get("success_rate", 0),
            }

print(f"Found {len(TRIALS)} trials with checkpoints (T0-T{max(TRIALS.keys()) if TRIALS else '?'})")


# --- Evaluation ---
def evaluate_checkpoint(ckpt_path: str, num_envs: int = 2048, max_steps: int = 3000):
    """Run one headless evaluation episode and return metrics."""
    import torch
    from motrix_envs import registry
    from motrix_rl.skrl.torch.train.ppo import Trainer as PpoTrainer, _get_cfg

    # Use Trainer to build agent with correct model architecture
    trainer = PpoTrainer("vbot_navigation_section011", cfg_override={"play_num_envs": num_envs})
    env = registry.make("vbot_navigation_section011", num_envs=num_envs)

    from motrix_rl.skrl.torch import wrap_env
    skrl_env = wrap_env(env, enable_render=False)
    rlcfg = trainer._rlcfg
    models = trainer._make_model(skrl_env, rlcfg)
    ppo_cfg = _get_cfg(rlcfg, skrl_env)
    agent = trainer._make_agent(models, skrl_env, ppo_cfg)
    agent.load(ckpt_path)
    agent.set_running_mode("eval")

    obs, _ = skrl_env.reset()

    max_wp = np.zeros(num_envs, dtype=np.float32)
    max_y = np.full(num_envs, -999.0, dtype=np.float32)
    max_nav_phase = np.zeros(num_envs, dtype=np.float32)
    ep_count = np.zeros(num_envs, dtype=np.int32)
    cumul_wp = np.zeros(num_envs, dtype=np.float32)
    term_count = 0
    total_done = 0

    with torch.inference_mode():
        for step in range(max_steps):
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated_t, truncated_t, infos = skrl_env.step(actions)

            # Get raw info from underlying env
            raw_info = env._state.info if hasattr(env, '_state') else {}
            raw_metrics = raw_info.get("metrics", {})

            wp_idx = raw_metrics.get("wp_idx_mean", np.zeros(num_envs))
            y_prog = raw_metrics.get("max_y_progress", np.full(num_envs, -999.0))
            nav_phase = raw_metrics.get("nav_phase_mean", np.zeros(num_envs))

            max_wp = np.maximum(max_wp, wp_idx)
            max_y = np.maximum(max_y, y_prog)
            max_nav_phase = np.maximum(max_nav_phase, nav_phase)

            # Track terminations
            terminated_np = terminated_t.cpu().numpy().flatten() if isinstance(terminated_t, torch.Tensor) else np.asarray(terminated_t).flatten()
            truncated_np = truncated_t.cpu().numpy().flatten() if isinstance(truncated_t, torch.Tensor) else np.asarray(truncated_t).flatten()
            done = terminated_np | truncated_np
            term_count += terminated_np.sum()
            total_done += done.sum()

            # On reset, accumulate and restart per-env tracking
            if done.any():
                ep_count += done.astype(np.int32)
                cumul_wp += np.where(done, max_wp, 0.0)
                # Reset per-episode trackers for done envs
                max_wp = np.where(done, 0.0, max_wp)
                max_y = np.where(done, -999.0, max_y)

    # Final episode (not yet reset)
    cumul_wp += max_wp
    ep_count_safe = np.maximum(ep_count + 1, 1)  # +1 for the ongoing episode

    return {
        "wp_idx_mean": float((cumul_wp / ep_count_safe).mean()),
        "wp_idx_max": float(max_wp.max()),
        "wp_idx_p75": float(np.percentile(cumul_wp / ep_count_safe, 75)),
        "max_y_mean": float(max_y[max_y > -900].mean() if (max_y > -900).any() else 0.0),
        "max_y_max": float(max_y.max()),
        "max_nav_phase": float(max_nav_phase.max()),
        "avg_nav_phase": float(max_nav_phase.mean()),
        "term_rate": float(term_count / max(total_done, 1)),
        "avg_episodes": float(ep_count.mean()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch eval Stage 2 T0-T7 with relaxed termination")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--trials", type=str, default="0-7", help="Trial range, e.g. '0-7' or '0,2,4'")
    args = parser.parse_args()

    # Parse trial range
    if "-" in args.trials:
        start, end = args.trials.split("-")
        trial_indices = list(range(int(start), int(end) + 1))
    else:
        trial_indices = [int(x) for x in args.trials.split(",")]

    results = {}
    for t_idx in trial_indices:
        if t_idx not in TRIALS:
            print(f"T{t_idx}: checkpoint not found, skipping")
            continue
        trial = TRIALS[t_idx]
        print(f"\n{'='*70}")
        print(f"EVALUATING T{t_idx} (relaxed termination)")
        print(f"  Checkpoint: {trial['ckpt']}")
        print(f"  Original wp_idx: {trial['original_wp']:.3f}, success: {trial['original_success']:.4f}")
        print(f"{'='*70}")

        metrics = evaluate_checkpoint(trial["ckpt"], args.num_envs, args.max_steps)
        metrics["original_wp"] = trial["original_wp"]
        metrics["original_success"] = trial["original_success"]
        results[t_idx] = metrics

        print(f"  wp_idx: {metrics['wp_idx_mean']:.3f} (orig: {trial['original_wp']:.3f}, delta: {metrics['wp_idx_mean'] - trial['original_wp']:+.3f})")
        print(f"  wp_idx_max: {metrics['wp_idx_max']:.1f}, p75: {metrics['wp_idx_p75']:.3f}")
        print(f"  max_y: {metrics['max_y_mean']:.2f}m (max: {metrics['max_y_max']:.2f}m)")
        print(f"  nav_phase: avg={metrics['avg_nav_phase']:.2f}, max={metrics['max_nav_phase']:.0f}")
        print(f"  term_rate: {metrics['term_rate']:.4f}")
        print(f"  avg_episodes: {metrics['avg_episodes']:.1f}")

    # --- Summary table ---
    if results:
        print(f"\n\n{'='*100}")
        print("COMPARISON TABLE: Original vs Relaxed Termination")
        print(f"{'='*100}")
        header = f"{'Trial':>6} | {'Orig wp':>8} | {'Relax wp':>9} | {'Delta':>7} | {'wp_max':>7} | {'max_y':>7} | {'navPh':>6} | {'term%':>7} | {'ep/env':>7}"
        print(header)
        print("-" * len(header))
        for t_idx in sorted(results.keys()):
            m = results[t_idx]
            delta = m["wp_idx_mean"] - m["original_wp"]
            print(f"  T{t_idx:>3} | {m['original_wp']:>8.3f} | {m['wp_idx_mean']:>9.3f} | {delta:>+7.3f} | {m['wp_idx_max']:>7.1f} | {m['max_y_mean']:>7.2f} | {m['avg_nav_phase']:>6.2f} | {m['term_rate']*100:>6.2f}% | {m['avg_episodes']:>7.1f}")

        # Find best trial
        best_t = max(results.keys(), key=lambda t: results[t]["wp_idx_mean"])
        print(f"\nBest trial (relaxed): T{best_t} with wp_idx={results[best_t]['wp_idx_mean']:.3f}")

    # Save results to JSON
    out_path = WORKSPACE / "_eval_relaxed_results.json"
    with open(out_path, "w") as f:
        json.dump({f"T{k}": v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
