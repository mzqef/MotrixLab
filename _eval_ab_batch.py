"""
Batch headless A/B evaluation: Standard vs Relaxed termination for Stage 2 T0-T7.
Captures metrics BEFORE auto-reset to avoid post-reset data corruption.

Does NOT modify any existing files or AutoML state.
"""
import sys
import json
import copy
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent

# --- Register environment ---
sys.path.insert(0, str(WORKSPACE / "starter_kit" / "navigation2"))
import vbot as navigation2_vbot  # noqa: F401, E402

from motrix_envs import registry as env_registry

# --- Get env class for monkey patching ---
_meta = env_registry._envs.get("vbot_navigation_section011")
env_cls = _meta.env_cls_dict.get("np")

# Save originals
_original_compute_terminated = env_cls._compute_terminated
_original_update_truncate = env_cls._update_truncate


def _relaxed_compute_terminated(self, state, projected_gravity=None, joint_vel=None, robot_xy=None, current_z=None):
    """Relaxed termination: only extreme physics failures."""
    n = self._num_envs
    hard_terminated = np.zeros(n, dtype=bool)
    if projected_gravity is not None:
        gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
        gz = projected_gravity[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        hard_terminated |= tilt_angle > np.deg2rad(85)
    bounds = getattr(self._cfg, 'course_bounds', None)
    if bounds is not None and robot_xy is not None and current_z is not None:
        oob_x = (robot_xy[:, 0] < bounds.x_min) | (robot_xy[:, 0] > bounds.x_max)
        oob_y = (robot_xy[:, 1] < bounds.y_min) | (robot_xy[:, 1] > bounds.y_max)
        oob_z = current_z < bounds.z_min
        hard_terminated |= oob_x | oob_y | oob_z
        state.info["oob_terminated"] = oob_x | oob_y | oob_z
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


def _relaxed_update_truncate(self):
    from motrix_envs.np.env import NpEnv
    NpEnv._update_truncate(self)  # time limit only


def set_relaxed_mode():
    env_cls._compute_terminated = _relaxed_compute_terminated
    env_cls._update_truncate = _relaxed_update_truncate


def set_standard_mode():
    env_cls._compute_terminated = _original_compute_terminated
    env_cls._update_truncate = _original_update_truncate


# --- Find trial checkpoints ---
import yaml

AUTOML_DIR = WORKSPACE / "starter_kit_log" / "automl_20260222_124457"
EXPERIMENTS_DIR = AUTOML_DIR / "experiments"

TRIALS = {}
for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
    if not exp_dir.is_dir():
        continue
    summary_path = exp_dir / "summary.yaml"
    if not summary_path.exists():
        continue
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
        with open(summary_path) as f:
            summary = yaml.safe_load(f)
        run_dir = summary.get("results", {}).get("run_dir", "") or summary.get("run_dir", "")
        ckpt = WORKSPACE / run_dir / "checkpoints" / "best_agent.pt"
        if ckpt.exists():
            TRIALS[trial_idx] = {
                "ckpt": str(ckpt),
                "original_wp": summary.get("results", {}).get("final_metrics", {}).get("wp_idx_mean", 0),
                "original_success": summary.get("results", {}).get("final_metrics", {}).get("success_rate", 0),
            }

print(f"Found {len(TRIALS)} trials (T0-T{max(TRIALS.keys()) if TRIALS else '?'})")


# --- Core evaluation ---
def evaluate_checkpoint(ckpt_path: str, num_envs: int = 2048, max_steps: int = 12000):
    """
    Run one full-episode evaluation.
    Hooks into env to capture pre-reset metrics.
    Returns per-episode-aggregated metrics.
    """
    import torch
    from motrix_envs import registry
    from motrix_rl.skrl.torch.train.ppo import Trainer as PpoTrainer, _get_cfg
    from motrix_rl.skrl.torch import wrap_env
    from motrix_envs.np.env import NpEnv

    trainer = PpoTrainer("vbot_navigation_section011", cfg_override={"play_num_envs": num_envs})
    env = registry.make("vbot_navigation_section011", num_envs=num_envs)

    # === Hook: capture pre-reset metrics ===
    # We intercept _reset_done_envs to save metrics before they get overwritten
    _episode_peaks = {
        "wp_idx": np.zeros(num_envs, dtype=np.float32),      # max wp_idx this episode
        "max_y": np.full(num_envs, -999.0, dtype=np.float32), # max y-forward this episode
        "nav_phase": np.zeros(num_envs, dtype=np.float32),
    }
    _completed_episodes = []  # list of per-episode peak dicts (one per done env)
    _term_events = []  # (terminated_bool, truncated_bool) for each done env

    _original_reset_done = env._reset_done_envs

    def _hooked_reset_done_envs():
        state = env._state
        done = state.done
        if np.any(done):
            # Capture pre-reset metrics for done envs
            metrics = state.info.get("metrics", {})
            wp_idx = metrics.get("wp_idx_mean", np.zeros(num_envs))
            y_prog = metrics.get("max_y_progress", np.full(num_envs, -999.0))
            nav_phase = metrics.get("nav_phase_mean", np.zeros(num_envs))

            # Update episode peaks
            _episode_peaks["wp_idx"] = np.maximum(_episode_peaks["wp_idx"], wp_idx)
            _episode_peaks["max_y"] = np.maximum(_episode_peaks["max_y"], y_prog)
            _episode_peaks["nav_phase"] = np.maximum(_episode_peaks["nav_phase"], nav_phase)

            done_indices = np.where(done)[0]
            for idx in done_indices:
                _completed_episodes.append({
                    "wp_idx": float(_episode_peaks["wp_idx"][idx]),
                    "max_y": float(_episode_peaks["max_y"][idx]),
                    "nav_phase": float(_episode_peaks["nav_phase"][idx]),
                    "terminated": bool(state.terminated[idx]),
                    "truncated": bool(state.truncated[idx]),
                })

            # Reset episode peaks for done envs
            _episode_peaks["wp_idx"] = np.where(done, 0.0, _episode_peaks["wp_idx"])
            _episode_peaks["max_y"] = np.where(done, -999.0, _episode_peaks["max_y"])
            _episode_peaks["nav_phase"] = np.where(done, 0.0, _episode_peaks["nav_phase"])

        _original_reset_done()

    env._reset_done_envs = _hooked_reset_done_envs

    skrl_env = wrap_env(env, enable_render=False)
    rlcfg = trainer._rlcfg
    models = trainer._make_model(skrl_env, rlcfg)
    ppo_cfg = _get_cfg(rlcfg, skrl_env)
    agent = trainer._make_agent(models, skrl_env, ppo_cfg)
    agent.load(ckpt_path)
    agent.set_running_mode("eval")

    obs, _ = skrl_env.reset()

    with torch.inference_mode():
        for step in range(max_steps):
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, infos = skrl_env.step(actions)

            # Also track peaks for still-running envs
            metrics = infos.get("metrics", {})
            wp_idx = metrics.get("wp_idx_mean", np.zeros(num_envs))
            y_prog = metrics.get("max_y_progress", np.full(num_envs, -999.0))
            nav_phase = metrics.get("nav_phase_mean", np.zeros(num_envs))
            _episode_peaks["wp_idx"] = np.maximum(_episode_peaks["wp_idx"], wp_idx)
            _episode_peaks["max_y"] = np.maximum(_episode_peaks["max_y"], y_prog)
            _episode_peaks["nav_phase"] = np.maximum(_episode_peaks["nav_phase"], nav_phase)

    # Include still-running episodes (not yet terminated)
    for idx in range(num_envs):
        _completed_episodes.append({
            "wp_idx": float(_episode_peaks["wp_idx"][idx]),
            "max_y": float(_episode_peaks["max_y"][idx]),
            "nav_phase": float(_episode_peaks["nav_phase"][idx]),
            "terminated": False,
            "truncated": False,
        })

    # Aggregate
    n_eps = len(_completed_episodes)
    wp_vals = np.array([e["wp_idx"] for e in _completed_episodes])
    y_vals = np.array([e["max_y"] for e in _completed_episodes])
    phase_vals = np.array([e["nav_phase"] for e in _completed_episodes])
    term_count = sum(1 for e in _completed_episodes if e["terminated"])
    trunc_count = sum(1 for e in _completed_episodes if e["truncated"])
    alive_count = sum(1 for e in _completed_episodes if not e["terminated"] and not e["truncated"])

    return {
        "n_episodes": n_eps,
        "wp_idx_mean": float(wp_vals.mean()),
        "wp_idx_median": float(np.median(wp_vals)),
        "wp_idx_max": float(wp_vals.max()),
        "wp_idx_p75": float(np.percentile(wp_vals, 75)),
        "wp_idx_p90": float(np.percentile(wp_vals, 90)),
        "wp_ge3": float((wp_vals >= 3).mean()),  # fraction that collected all 3 smileys
        "wp_ge4": float((wp_vals >= 4).mean()),  # fraction that reached red packets
        "wp_ge6": float((wp_vals >= 6).mean()),  # fraction that collected all red packets
        "wp_eq7": float((wp_vals >= 7).mean()),   # fraction that reached platform
        "max_y_mean": float(y_vals[y_vals > -900].mean() if (y_vals > -900).any() else 0),
        "max_y_max": float(y_vals.max()),
        "max_y_p75": float(np.percentile(y_vals[y_vals > -900], 75) if (y_vals > -900).any() else 0),
        "nav_phase_mean": float(phase_vals.mean()),
        "nav_phase_max": float(phase_vals.max()),
        "terminated_frac": float(term_count / n_eps),
        "truncated_frac": float(trunc_count / n_eps),
        "alive_frac": float(alive_count / n_eps),
        "episodes_per_env": float((n_eps - num_envs) / num_envs),  # subtract the still-running ones
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=12000, help="12000 = one full episode at 120s")
    parser.add_argument("--trials", type=str, default="0-7")
    args = parser.parse_args()

    if "-" in args.trials:
        start, end = args.trials.split("-")
        trial_indices = list(range(int(start), int(end) + 1))
    else:
        trial_indices = [int(x) for x in args.trials.split(",")]

    all_results = {}

    for t_idx in trial_indices:
        if t_idx not in TRIALS:
            print(f"T{t_idx}: not found, skipping")
            continue
        trial = TRIALS[t_idx]
        print(f"\n{'='*80}")
        print(f"TRIAL T{t_idx}")
        print(f"  Checkpoint: {trial['ckpt']}")
        print(f"  Training wp_idx_mean: {trial['original_wp']:.3f}")
        print(f"{'='*80}")

        # --- Standard termination ---
        set_standard_mode()
        print(f"\n  [STANDARD TERMINATION]")
        std_metrics = evaluate_checkpoint(trial["ckpt"], args.num_envs, args.max_steps)
        print(f"    Episodes: {std_metrics['n_episodes']} ({std_metrics['episodes_per_env']:.1f}/env)")
        print(f"    wp_idx: mean={std_metrics['wp_idx_mean']:.3f} median={std_metrics['wp_idx_median']:.1f} p75={std_metrics['wp_idx_p75']:.1f} p90={std_metrics['wp_idx_p90']:.1f} max={std_metrics['wp_idx_max']:.0f}")
        print(f"    wp>=3: {std_metrics['wp_ge3']*100:.1f}%  wp>=4: {std_metrics['wp_ge4']*100:.1f}%  wp>=6: {std_metrics['wp_ge6']*100:.1f}%  wp=7: {std_metrics['wp_eq7']*100:.1f}%")
        print(f"    max_y: mean={std_metrics['max_y_mean']:.2f}m  p75={std_metrics['max_y_p75']:.2f}m  max={std_metrics['max_y_max']:.2f}m")
        print(f"    terminated: {std_metrics['terminated_frac']*100:.1f}%  truncated: {std_metrics['truncated_frac']*100:.1f}%  alive: {std_metrics['alive_frac']*100:.1f}%")

        # --- Relaxed termination ---
        set_relaxed_mode()
        print(f"\n  [RELAXED TERMINATION]")
        rel_metrics = evaluate_checkpoint(trial["ckpt"], args.num_envs, args.max_steps)
        print(f"    Episodes: {rel_metrics['n_episodes']} ({rel_metrics['episodes_per_env']:.1f}/env)")
        print(f"    wp_idx: mean={rel_metrics['wp_idx_mean']:.3f} median={rel_metrics['wp_idx_median']:.1f} p75={rel_metrics['wp_idx_p75']:.1f} p90={rel_metrics['wp_idx_p90']:.1f} max={rel_metrics['wp_idx_max']:.0f}")
        print(f"    wp>=3: {rel_metrics['wp_ge3']*100:.1f}%  wp>=4: {rel_metrics['wp_ge4']*100:.1f}%  wp>=6: {rel_metrics['wp_ge6']*100:.1f}%  wp=7: {rel_metrics['wp_eq7']*100:.1f}%")
        print(f"    max_y: mean={rel_metrics['max_y_mean']:.2f}m  p75={rel_metrics['max_y_p75']:.2f}m  max={rel_metrics['max_y_max']:.2f}m")
        print(f"    terminated: {rel_metrics['terminated_frac']*100:.1f}%  truncated: {rel_metrics['truncated_frac']*100:.1f}%  alive: {rel_metrics['alive_frac']*100:.1f}%")

        # --- Delta ---
        delta_wp = rel_metrics["wp_idx_mean"] - std_metrics["wp_idx_mean"]
        delta_y = rel_metrics["max_y_mean"] - std_metrics["max_y_mean"]
        print(f"\n  [DELTA] wp_idx: {delta_wp:+.3f}  max_y: {delta_y:+.2f}m")

        all_results[t_idx] = {"standard": std_metrics, "relaxed": rel_metrics}

    # --- Summary table ---
    if all_results:
        print(f"\n\n{'='*120}")
        print("SUMMARY: Standard vs Relaxed Termination (Stage 2 AutoML)")
        print(f"{'='*120}")
        print(f"{'Trial':>6} | {'--- STANDARD ---':^42} | {'--- RELAXED ---':^42} | {'Delta wp':>9}")
        print(f"{'':>6} | {'wp_mean':>8} {'wp_max':>7} {'wp>=3%':>7} {'max_y':>7} {'term%':>7} {'ep/env':>7} | {'wp_mean':>8} {'wp_max':>7} {'wp>=3%':>7} {'max_y':>7} {'term%':>7} {'ep/env':>7} | {'':>9}")
        print("-" * 120)
        for t_idx in sorted(all_results.keys()):
            s = all_results[t_idx]["standard"]
            r = all_results[t_idx]["relaxed"]
            delta = r["wp_idx_mean"] - s["wp_idx_mean"]
            print(f"  T{t_idx:>3} | {s['wp_idx_mean']:>8.3f} {s['wp_idx_max']:>7.0f} {s['wp_ge3']*100:>6.1f}% {s['max_y_mean']:>7.2f} {s['terminated_frac']*100:>6.1f}% {s['episodes_per_env']:>7.1f} | {r['wp_idx_mean']:>8.3f} {r['wp_idx_max']:>7.0f} {r['wp_ge3']*100:>6.1f}% {r['max_y_mean']:>7.2f} {r['terminated_frac']*100:>6.1f}% {r['episodes_per_env']:>7.1f} | {delta:>+9.3f}")

    out_path = WORKSPACE / "_eval_ab_results.json"
    with open(out_path, "w") as f:
        json.dump({f"T{k}": v for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
