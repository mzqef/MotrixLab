"""Monitor AutoML runs — extract key metrics from TensorBoard."""
import os
import glob
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_metrics(run_dir):
    tb_files = glob.glob(os.path.join(run_dir, "events.out.*"))
    if not tb_files:
        return None
    ea = EventAccumulator(run_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    result = {}
    for tag in ["metrics / wp_idx_mean (max)", "metrics / wp_idx_mean (mean)"]:
        if tag in tags:
            vals = ea.Scalars(tag)
            if vals:
                key = "wp_max" if "max" in tag else "wp_mean"
                result[key] = max(v.value for v in vals)

    tag = "Reward / Total reward (mean)"
    if tag in tags:
        vals = ea.Scalars(tag)
        if vals:
            peak_v = max(vals, key=lambda v: v.value)
            result["rew_peak"] = peak_v.value
            result["peak_iter"] = peak_v.step

    for tag in ["Episode / Total timesteps (mean)", "Episode / Total timesteps (min)"]:
        if tag in tags:
            vals = ea.Scalars(tag)
            if vals:
                key = "ep_mean" if "mean" in tag else "ep_min"
                result[key] = vals[-1].value

    for tag in ["metrics / distance_to_target (mean)", "metrics / distance_to_target (max)"]:
        if tag in tags:
            vals = ea.Scalars(tag)
            if vals:
                key = "dist_mean" if "mean" in tag else "dist_max"
                result[key] = vals[-1].value

    tag = "metrics / stair_cleared_fraction (mean)"
    if tag in tags:
        vals = ea.Scalars(tag)
        if vals:
            result["stair_pct"] = max(v.value for v in vals) * 100.0
            result["stair_exact"] = True
    else:
        # Estimate stair% from wp_idx_mean time series.
        # wp_idx >= 2 means stairs cleared (WP0=right_approach, WP1=stair_top).
        # Read per-step mean/max and build step-aligned estimate.
        mean_tag = "metrics / wp_idx_mean (mean)"
        max_tag = "metrics / wp_idx_mean (max)"
        if mean_tag in tags and max_tag in tags:
            mean_vals = {v.step: v.value for v in ea.Scalars(mean_tag)}
            max_vals = {v.step: v.value for v in ea.Scalars(max_tag)}
            best_stair_pct = 0.0
            for step in mean_vals:
                m = mean_vals[step]
                mx = max_vals.get(step, m)
                if mx < 2:
                    pct = 0.0
                else:
                    # Estimate: clamp((mean - 1) / 1, 0, 1).
                    # Treats wp_idx=1 as "not cleared", >=2 as "cleared".
                    pct = max(0.0, min(1.0, m - 1.0))
                best_stair_pct = max(best_stair_pct, pct)
            result["stair_pct"] = best_stair_pct * 100.0
            result["stair_exact"] = False

    return result


def find_runs_from_summaries(automl_id):
    exp_dir = f"starter_kit_log/{automl_id}/experiments"
    if not os.path.isdir(exp_dir):
        return {}
    runs = {}
    for exp_name in sorted(os.listdir(exp_dir)):
        summary_file = os.path.join(exp_dir, exp_name, "summary.yaml")
        if not os.path.exists(summary_file):
            continue
        with open(summary_file) as f:
            summary = yaml.safe_load(f)
        trial_idx = summary.get("trial_index", -1)
        run_dir = summary.get("results", {}).get("run_dir", "")
        if run_dir and os.path.isdir(run_dir):
            runs[f"i{trial_idx}"] = run_dir
    return runs


def print_run(automl_id, label):
    print("=" * 110)
    print(f"{automl_id} -- {label}")
    print("=" * 110)
    hdr = f"{'trial':>6}  {'wp_max':>6}  {'wp_mean':>7}  {'rew_peak':>9}  {'peak_it':>7}  {'ep_min':>6}  {'ep_mean':>7}  {'dist_mean':>9}  {'dist_max':>8}  {'stair%':>7}"
    print(hdr)
    print("-" * len(hdr))

    runs = find_runs_from_summaries(automl_id)
    if not runs:
        print("  (no runs found)")
        return

    for trial, path in sorted(runs.items(), key=lambda x: int(x[0][1:])):
        m = extract_metrics(path)
        if m:
            if 'stair_pct' in m:
                prefix = "" if m.get("stair_exact") else "~"
                stair_str = f"{prefix}{m['stair_pct']:.0f}%".rjust(7)
            else:
                stair_str = "    n/a"
            print(
                f"{trial:>6}  "
                f"{m.get('wp_max', 0):>6.1f}  "
                f"{m.get('wp_mean', 0):>7.2f}  "
                f"{m.get('rew_peak', 0):>9.1f}  "
                f"{m.get('peak_iter', 0):>7}  "
                f"{m.get('ep_min', 0):>6.0f}  "
                f"{m.get('ep_mean', 0):>7.0f}  "
                f"{m.get('dist_mean', 0):>9.2f}  "
                f"{m.get('dist_max', 0):>8.2f}  "
                f"{stair_str}"
            )

    state_file = f"starter_kit_log/{automl_id}/state.yaml"
    with open(state_file) as f:
        state = yaml.safe_load(f)
    elapsed = state.get("elapsed_hours", 0)
    budget = state.get("budget_hours", 0)
    iteration = state.get("current_iteration", 0)
    print(f"\n  iter={iteration}, elapsed={elapsed:.2f}h / {budget}h budget, remaining={max(0, budget-elapsed):.2f}h")


if __name__ == "__main__":
    print_run("automl_20260227_220608", "footprint-contains (current, running)")
    print()
    print_run("automl_20260227_173458", "radius-based (previous, died at 2.6h)")
