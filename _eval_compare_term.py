"""
Batch headless evaluation of Stage 2 T0-T7 checkpoints: STANDARD vs RELAXED termination.
Runs the same eval methodology for both modes so the comparison is fair.

Does NOT modify any existing files or AutoML state.

Usage:
    uv run python _eval_compare_term.py --trials 0-7 --num-envs 2048 --max-steps 3000
"""
import sys
import json
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent

# --- Register environment ---
sys.path.insert(0, str(WORKSPACE / "starter_kit" / "navigation2"))
import vbot as navigation2_vbot  # noqa: F401, E402

from motrix_envs import registry as env_registry

# --- Save original methods ---
_meta = env_registry._envs.get("vbot_navigation_section011")
env_cls = _meta.env_cls_dict.get("np")
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
    NpEnv._update_truncate(self)


def set_termination_mode(mode: str):
    """Switch termination mode: 'standard' or 'relaxed'."""
    if mode == "relaxed":
        env_cls._compute_terminated = _relaxed_compute_terminated
        env_cls._update_truncate = _relaxed_update_truncate
    else:
        env_cls._compute_terminated = _original_compute_terminated
        env_cls._update_truncate = _original_update_truncate


# --- Trial configs ---
def load_trials(max_trial: int = 7):
    import yaml
    automl_dir = WORKSPACE / "starter_kit_log" / "automl_20260222_124457"
    experiments_dir = automl_dir / "experiments"
    trials = {}
    for exp_dir in sorted(experiments_dir.iterdir()):
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
            if trial_idx > max_trial:
                continue
            with open(summary_path) as f:
                summary = yaml.safe_load(f)
            run_dir = summary.get("results", {}).get("run_dir", "")
            if not run_dir:
                run_dir = summary.get("run_dir", "")
            ckpt = WORKSPACE / run_dir / "checkpoints" / "best_agent.pt"
            if ckpt.exists():
                trials[trial_idx] = {"run_dir": run_dir, "ckpt": str(ckpt)}
    return trials


def evaluate_checkpoint(ckpt_path: str, num_envs: int = 2048, max_steps: int = 3000):
    """Run one headless evaluation and return per-episode metrics."""
    import torch
    from motrix_envs import registry
    from motrix_rl.skrl.torch.train.ppo import Trainer as PpoTrainer, _get_cfg
    from motrix_rl.skrl.torch import wrap_env

    trainer = PpoTrainer("vbot_navigation_section011", cfg_override={"play_num_envs": num_envs})
    env = registry.make("vbot_navigation_section011", num_envs=num_envs)
    skrl_env = wrap_env(env, enable_render=False)
    rlcfg = trainer._rlcfg
    models = trainer._make_model(skrl_env, rlcfg)
    ppo_cfg = _get_cfg(rlcfg, skrl_env)
    agent = trainer._make_agent(models, skrl_env, ppo_cfg)
    agent.load(ckpt_path)
    agent.set_running_mode("eval")

    obs, _ = skrl_env.reset()

    # Per-env episode trackers
    ep_max_wp = np.zeros(num_envs, dtype=np.float32)      # max wp in current episode
    ep_max_y = np.full(num_envs, -999.0, dtype=np.float32) # max y in current episode
    ep_max_phase = np.zeros(num_envs, dtype=np.float32)

    # Accumulators across episodes
    all_ep_max_wp = []    # list of per-episode max wp values
    all_ep_max_y = []
    all_ep_lens = []
    term_count = 0
    trunc_count = 0
    ep_steps = np.zeros(num_envs, dtype=np.int32)

    with torch.inference_mode():
        for step in range(max_steps):
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated_t, truncated_t, infos = skrl_env.step(actions)
            ep_steps += 1

            # Get raw metrics from underlying env
            raw_info = env._state.info if hasattr(env, '_state') else {}
            raw_metrics = raw_info.get("metrics", {})

            wp_idx = raw_metrics.get("wp_idx_mean", np.zeros(num_envs))
            y_prog = raw_metrics.get("max_y_progress", np.full(num_envs, -999.0))
            nav_phase = raw_metrics.get("nav_phase_mean", np.zeros(num_envs))

            ep_max_wp = np.maximum(ep_max_wp, wp_idx)
            ep_max_y = np.maximum(ep_max_y, y_prog)
            ep_max_phase = np.maximum(ep_max_phase, nav_phase)

            # Detect episode end
            terminated_np = terminated_t.cpu().numpy().flatten() if isinstance(terminated_t, torch.Tensor) else np.asarray(terminated_t).flatten()
            truncated_np = truncated_t.cpu().numpy().flatten() if isinstance(truncated_t, torch.Tensor) else np.asarray(truncated_t).flatten()
            done = terminated_np | truncated_np

            # Collect completed episode stats
            done_idxs = np.where(done)[0]
            for i in done_idxs:
                all_ep_max_wp.append(float(ep_max_wp[i]))
                all_ep_max_y.append(float(ep_max_y[i]))
                all_ep_lens.append(int(ep_steps[i]))

            term_count += int(terminated_np.sum())
            trunc_count += int((truncated_np & ~terminated_np).sum())

            # Reset per-episode trackers for done envs
            ep_max_wp = np.where(done, 0.0, ep_max_wp)
            ep_max_y = np.where(done, -999.0, ep_max_y)
            ep_max_phase = np.where(done, 0.0, ep_max_phase)
            ep_steps = np.where(done, 0, ep_steps)

    # Add ongoing (unfinished) episodes
    for i in range(num_envs):
        if ep_steps[i] > 0:  # still running
            all_ep_max_wp.append(float(ep_max_wp[i]))
            all_ep_max_y.append(float(ep_max_y[i]))
            all_ep_lens.append(int(ep_steps[i]))

    wp_arr = np.array(all_ep_max_wp) if all_ep_max_wp else np.zeros(1)
    y_arr = np.array(all_ep_max_y) if all_ep_max_y else np.zeros(1)
    len_arr = np.array(all_ep_lens) if all_ep_lens else np.zeros(1)
    y_valid = y_arr[y_arr > -900]

    total_eps = len(all_ep_max_wp)
    completed_eps = term_count + trunc_count

    return {
        "wp_mean": float(wp_arr.mean()),
        "wp_median": float(np.median(wp_arr)),
        "wp_p75": float(np.percentile(wp_arr, 75)),
        "wp_max": float(wp_arr.max()),
        "y_mean": float(y_valid.mean()) if len(y_valid) > 0 else 0.0,
        "y_max": float(y_arr.max()),
        "ep_len_mean": float(len_arr.mean()),
        "total_episodes": total_eps,
        "completed_episodes": completed_eps,
        "term_count": term_count,
        "trunc_count": trunc_count,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare standard vs relaxed termination for Stage 2 T0-T7")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--trials", type=str, default="0-7")
    args = parser.parse_args()

    if "-" in args.trials:
        start, end = args.trials.split("-")
        trial_indices = list(range(int(start), int(end) + 1))
    else:
        trial_indices = [int(x) for x in args.trials.split(",")]

    trials = load_trials()
    print(f"Found {len(trials)} trials with checkpoints")

    results = {}
    for t_idx in trial_indices:
        if t_idx not in trials:
            print(f"T{t_idx}: checkpoint not found, skipping")
            continue
        ckpt = trials[t_idx]["ckpt"]
        print(f"\n{'='*70}")
        print(f"T{t_idx}: {ckpt}")
        print(f"{'='*70}")

        # --- Standard termination ---
        set_termination_mode("standard")
        print(f"  [STANDARD] evaluating...")
        std = evaluate_checkpoint(ckpt, args.num_envs, args.max_steps)
        print(f"    wp={std['wp_mean']:.3f} (med={std['wp_median']:.1f} p75={std['wp_p75']:.1f} max={std['wp_max']:.0f}) "
              f"y={std['y_mean']:.2f}m eps={std['total_episodes']} term={std['term_count']} trunc={std['trunc_count']}")

        # --- Relaxed termination ---
        set_termination_mode("relaxed")
        print(f"  [RELAXED]  evaluating...")
        rlx = evaluate_checkpoint(ckpt, args.num_envs, args.max_steps)
        print(f"    wp={rlx['wp_mean']:.3f} (med={rlx['wp_median']:.1f} p75={rlx['wp_p75']:.1f} max={rlx['wp_max']:.0f}) "
              f"y={rlx['y_mean']:.2f}m eps={rlx['total_episodes']} term={rlx['term_count']} trunc={rlx['trunc_count']}")

        delta_wp = rlx["wp_mean"] - std["wp_mean"]
        delta_y = rlx["y_mean"] - std["y_mean"]
        print(f"  DELTA: wp={delta_wp:+.3f}  y={delta_y:+.2f}m")

        results[t_idx] = {"standard": std, "relaxed": rlx}

    # --- Summary table ---
    if results:
        print(f"\n\n{'='*120}")
        print("FAIR COMPARISON: Standard vs Relaxed Termination (same eval methodology)")
        print(f"  Eval: {args.num_envs} envs x {args.max_steps} steps per trial")
        print(f"{'='*120}")
        print(f"{'':>6} |{'--- STANDARD ---':^36}|{'--- RELAXED ---':^36}| {'Delta':^12}")
        header = f"{'Trial':>6} | {'wp_mean':>7} {'wp_med':>6} {'wp_max':>6} {'y_mean':>7} {'ep_len':>7} {'#eps':>5} | {'wp_mean':>7} {'wp_med':>6} {'wp_max':>6} {'y_mean':>7} {'ep_len':>7} {'#eps':>5} | {'d_wp':>6} {'d_y':>6}"
        print(header)
        print("-" * len(header))

        for t_idx in sorted(results.keys()):
            s = results[t_idx]["standard"]
            r = results[t_idx]["relaxed"]
            d_wp = r["wp_mean"] - s["wp_mean"]
            d_y = r["y_mean"] - s["y_mean"]
            print(f"  T{t_idx:>3} | {s['wp_mean']:>7.3f} {s['wp_median']:>6.1f} {s['wp_max']:>6.0f} {s['y_mean']:>7.2f} {s['ep_len_mean']:>7.0f} {s['total_episodes']:>5} | "
                  f"{r['wp_mean']:>7.3f} {r['wp_median']:>6.1f} {r['wp_max']:>6.0f} {r['y_mean']:>7.2f} {r['ep_len_mean']:>7.0f} {r['total_episodes']:>5} | "
                  f"{d_wp:>+6.3f} {d_y:>+6.2f}")

        # Averages
        avg_s_wp = np.mean([results[t]["standard"]["wp_mean"] for t in results])
        avg_r_wp = np.mean([results[t]["relaxed"]["wp_mean"] for t in results])
        avg_s_y = np.mean([results[t]["standard"]["y_mean"] for t in results])
        avg_r_y = np.mean([results[t]["relaxed"]["y_mean"] for t in results])
        print(f"  {'AVG':>4} | {avg_s_wp:>7.3f} {'':>6} {'':>6} {avg_s_y:>7.2f} {'':>7} {'':>5} | "
              f"{avg_r_wp:>7.3f} {'':>6} {'':>6} {avg_r_y:>7.2f} {'':>7} {'':>5} | "
              f"{avg_r_wp - avg_s_wp:>+6.3f} {avg_r_y - avg_s_y:>+6.2f}")

    # Save
    out_path = WORKSPACE / "_eval_compare_results.json"
    serializable = {}
    for t_idx, v in results.items():
        serializable[f"T{t_idx}"] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
