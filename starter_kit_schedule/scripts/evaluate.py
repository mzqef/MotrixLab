"""
Evaluate training results from TensorBoard logs.

Provides utilities to:
- Read scalar values from TensorBoard event files
- Extract key metrics (reward, stability, per-component breakdown)
- Rank experiments by performance
"""

import glob
import json
import os


def _find_event_files(run_dir):
    """Find TensorBoard event files in a run directory."""
    patterns = [
        os.path.join(run_dir, "events.out.tfevents.*"),
        os.path.join(run_dir, "**", "events.out.tfevents.*"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return files


def read_tb_scalars(run_dir, tag, max_entries=0):
    """
    Read scalar values from TensorBoard event files.

    Args:
        run_dir: Path to the run directory containing event files
        tag: Scalar tag name (e.g., 'Reward / Total reward (mean)')
        max_entries: Maximum entries to load (0 = all)

    Returns:
        List of (step, value) tuples
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[Evaluate] WARNING: tensorboard not installed, trying tbparse...")
        return _read_tb_scalars_tbparse(run_dir, tag)

    size_guidance = {"scalars": max_entries} if max_entries > 0 else {"scalars": 0}
    ea = EventAccumulator(run_dir, size_guidance=size_guidance)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if tag not in available_tags:
        return []

    events = ea.Scalars(tag)
    return [(e.step, e.value) for e in events]


def _read_tb_scalars_tbparse(run_dir, tag):
    """Fallback: read TensorBoard scalars using tbparse."""
    try:
        from tbparse import SummaryReader

        reader = SummaryReader(run_dir)
        df = reader.scalars
        filtered = df[df["tag"] == tag]
        return list(zip(filtered["step"].tolist(), filtered["value"].tolist()))
    except ImportError:
        print("[Evaluate] ERROR: Neither tensorboard nor tbparse available")
        return []


def list_tb_tags(run_dir):
    """List all available scalar tags in a TensorBoard run directory."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(run_dir, size_guidance={"scalars": 1})
        ea.Reload()
        return ea.Tags().get("scalars", [])
    except ImportError:
        return []


def evaluate_run(run_dir, primary_metric="Reward / Instantaneous reward (mean)"):
    """
    Evaluate a single training run by reading TensorBoard logs.

    Args:
        run_dir: Path to run directory
        primary_metric: TensorBoard scalar tag for the main performance metric.
                        Falls back to legacy 'Reward / Total reward (mean)' if not found.

    Returns:
        Dict with evaluation results:
        - final_reward: Average reward over last 20% of training
        - max_reward: Maximum reward achieved
        - last_reward: Last recorded reward value
        - total_steps: Total training steps
        - stability: Reward range in last 20% (lower = more stable)
        - status: 'ok' or 'no_data' or 'failed'
    """
    if not os.path.exists(run_dir):
        return {"status": "failed", "final_reward": float("-inf"), "error": "run_dir not found"}

    data = read_tb_scalars(run_dir, primary_metric)
    # Fallback to legacy tag name if primary not found
    if not data and primary_metric == "Reward / Instantaneous reward (mean)":
        data = read_tb_scalars(run_dir, "Reward / Total reward (mean)")
    if not data:
        return {"status": "no_data", "final_reward": float("-inf"), "error": "no reward data in TensorBoard"}

    steps, values = zip(*data)

    # Use last 20% of data for final performance assessment
    n = max(1, len(values) // 5)
    final_values = list(values[-n:])

    result = {
        "status": "ok",
        "final_reward": sum(final_values) / len(final_values),
        "max_reward": max(values),
        "last_reward": values[-1],
        "total_steps": steps[-1],
        "num_entries": len(values),
        "stability": max(final_values) - min(final_values),
    }

    # Also read custom metrics if available
    for metric_tag in [
        "metrics / reached_fraction (mean)",
        "metrics / distance_to_target (mean)",
        "metrics / wp_idx_mean (mean)",
    ]:
        metric_data = read_tb_scalars(run_dir, metric_tag)
        if metric_data:
            _, metric_values = zip(*metric_data)
            n_metric = max(1, len(metric_values) // 5)
            final_metric = list(metric_values[-n_metric:])
            key = metric_tag.split("/")[1].strip().replace(" (mean)", "")
            result[f"final_{key}"] = sum(final_metric) / len(final_metric)

    # Episode length from "Episode / Total timesteps (mean)"
    ep_len_data = read_tb_scalars(run_dir, "Episode / Total timesteps (mean)")
    if ep_len_data:
        _, ep_len_values = zip(*ep_len_data)
        n_ep = max(1, len(ep_len_values) // 5)
        result["final_episode_length"] = sum(list(ep_len_values[-n_ep:])) / n_ep

    # Termination rate proxy from "Reward Instant / termination (mean)".
    # termination reward fires once per terminated episode with a negative value;
    # its mean across envs is negative when episodes terminate.
    # We normalize by the configured termination penalty to get a rate ∈ [0, 1].
    term_data = read_tb_scalars(run_dir, "Reward Instant / termination (mean)")
    if term_data:
        _, term_values = zip(*term_data)
        n_term = max(1, len(term_values) // 5)
        final_term = sum(list(term_values[-n_term:])) / n_term
        # If mean termination reward is negative, episodes are terminating.
        # Crude rate: abs(mean_reward / penalty_per_event). Clamp to [0, 1].
        # With penalty=-100, a mean of -5 → ~5% termination rate.
        result["final_termination_reward"] = final_term

    return result


def rank_experiments(experiment_results, key="final_reward"):
    """
    Rank experiments by a specified metric (higher is better).

    Args:
        experiment_results: List of dicts with evaluation results
        key: Metric key to sort by

    Returns:
        Sorted list (best first)
    """
    valid = [r for r in experiment_results if r.get("status") == "ok"]
    failed = [r for r in experiment_results if r.get("status") != "ok"]
    ranked = sorted(valid, key=lambda r: r.get(key, float("-inf")), reverse=True)
    return ranked + failed


def load_experiment_meta(run_dir):
    """Load experiment metadata JSON from a run directory."""
    meta_path = os.path.join(run_dir, "experiment_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return None


def summarize_phase(results):
    """
    Print a summary table of experiment results.

    Args:
        results: List of dicts with 'run_tag', 'eval', and 'config' keys
    """
    print(f"\n{'='*80}")
    print(f"{'Rank':<5} {'Tag':<25} {'Final Reward':<15} {'Max Reward':<15} {'Stability':<12} {'Steps':<10}")
    print(f"{'-'*80}")

    ranked = rank_experiments([r["eval"] for r in results])

    for i, r in enumerate(ranked):
        tag = "?"
        for res in results:
            if res.get("eval") is r:
                tag = res.get("run_tag", "?")
                break

        status = r.get("status", "?")
        if status == "ok":
            print(
                f"{i+1:<5} {tag:<25} {r['final_reward']:<15.2f} "
                f"{r['max_reward']:<15.2f} {r['stability']:<12.2f} {r['total_steps']:<10}"
            )
        else:
            print(f"{i+1:<5} {tag:<25} {'FAILED':<15} {r.get('error', status)}")

    print(f"{'='*80}\n")
