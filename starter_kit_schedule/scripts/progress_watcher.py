"""
Progress watcher for long-running AutoML campaigns.

Monitors training progress, writes WAKE_UP.md with current state,
and provides a CLI for checking status at any time.

Usage:
    # Check current AutoML status
    uv run starter_kit_schedule/scripts/progress_watcher.py --status

    # Watch loop (runs every N seconds, writes WAKE_UP.md)
    uv run starter_kit_schedule/scripts/progress_watcher.py --watch --interval 120

    # One-shot snapshot
    uv run starter_kit_schedule/scripts/progress_watcher.py --snapshot
"""

import argparse
import glob
import os
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_ROOT = PROJECT_ROOT / "starter_kit_log"
WAKEUP_PATH = PROJECT_ROOT / "WAKE_UP.md"


def find_latest_automl_dir():
    """Find the most recent automl run directory."""
    automl_dirs = sorted(LOG_ROOT.glob("automl_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in automl_dirs:
        if (d / "state.yaml").exists():
            return d
    return automl_dirs[0] if automl_dirs else None


def load_automl_state(automl_dir):
    """Load AutoML state YAML (with fallback for numpy objects)."""
    import yaml
    state_path = automl_dir / "state.yaml"
    if state_path.exists():
        try:
            with open(state_path) as f:
                return yaml.safe_load(f)
        except yaml.YAMLError:
            # Fallback: read as text and extract key fields with regex
            import re
            text = state_path.read_text()
            state = {}
            for key in ["automl_id", "status", "current_phase", "current_stage",
                        "current_iteration", "elapsed_hours", "budget_hours", "mode"]:
                m = re.search(rf"^{key}:\s*(.+)$", text, re.MULTILINE)
                if m:
                    val = m.group(1).strip().strip("'\"")
                    try:
                        val = float(val)
                    except ValueError:
                        try:
                            val = int(val)
                        except ValueError:
                            pass
                    state[key] = val
            return state
    return None


def find_active_runs():
    """Find currently running training experiments by checking recent TensorBoard writes."""
    runs_dir = PROJECT_ROOT / "runs" / "vbot_navigation_section001"
    if not runs_dir.exists():
        return []

    active = []
    now = time.time()
    for run_dir in sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
        event_files = glob.glob(str(run_dir / "events.out.tfevents.*"))
        if event_files:
            latest_event = max(event_files, key=os.path.getmtime)
            age = now - os.path.getmtime(latest_event)
            if age < 600:  # Modified in last 10 minutes
                active.append({
                    "dir": run_dir.name,
                    "age_seconds": age,
                    "event_file": latest_event,
                })
    return active


def get_latest_metrics(run_dir):
    """Read latest TensorBoard metrics from a run directory."""
    try:
        from starter_kit_schedule.scripts.evaluate import read_tb_scalars
        metrics = {}
        for tag, key in [
            ("Reward / Instantaneous reward (mean)", "reward"),
            ("metrics / distance_to_target (mean)", "distance"),
            ("metrics / reached_fraction (mean)", "reached"),
        ]:
            data = read_tb_scalars(str(run_dir), tag)
            if data:
                steps, vals = zip(*data)
                metrics[key] = {
                    "last": vals[-1],
                    "max": max(vals),
                    "steps": steps[-1],
                    "entries": len(vals),
                }
        return metrics
    except Exception as e:
        return {"error": str(e)}


def generate_snapshot():
    """Generate a complete snapshot of current training state."""
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "automl": None,
        "active_runs": [],
        "best_runs": [],
    }

    # AutoML state
    automl_dir = find_latest_automl_dir()
    if automl_dir:
        snapshot["automl"] = load_automl_state(automl_dir)
        snapshot["automl_dir"] = str(automl_dir)

    # Active runs
    snapshot["active_runs"] = find_active_runs()

    # Best runs by reward
    runs_dir = PROJECT_ROOT / "runs" / "vbot_navigation_section001"
    if runs_dir.exists():
        all_runs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
        for run_dir in all_runs:
            metrics = get_latest_metrics(run_dir)
            if "reward" in metrics:
                snapshot["best_runs"].append({
                    "dir": run_dir.name,
                    "reward_last": metrics["reward"]["last"],
                    "reward_max": metrics["reward"]["max"],
                    "distance": metrics.get("distance", {}).get("last", "?"),
                    "reached": metrics.get("reached", {}).get("last", 0),
                    "steps": metrics["reward"]["steps"],
                })

    # Sort best_runs by reward
    snapshot["best_runs"].sort(key=lambda r: r.get("reward_max", 0), reverse=True)

    return snapshot


def write_wakeup_md(snapshot):
    """Write WAKE_UP.md with current state for agent context."""
    lines = [
        "# WAKE_UP — Current Training State",
        f"**Generated**: {snapshot['timestamp']}",
        "",
    ]

    # AutoML section
    automl = snapshot.get("automl")
    if automl:
        lines.extend([
            "## AutoML Status",
            f"- **ID**: {automl.get('automl_id', 'N/A')}",
            f"- **Status**: {automl.get('status', 'N/A')}",
            f"- **Phase**: {automl.get('current_phase', 'N/A')}",
            f"- **Iteration**: {automl.get('current_iteration', 'N/A')}",
            f"- **Elapsed**: {automl.get('elapsed_hours', 0):.1f} / {automl.get('budget_hours', 0):.1f} hours",
            "",
        ])

        best = automl.get("best_results", {})
        if best:
            lines.append("### Best Results")
            for stage, result in best.items():
                m = result.get("metrics", {})
                lines.append(f"- **{stage}**: reward={m.get('episode_reward_mean', 'N/A'):.2f}, "
                           f"success={m.get('success_rate', 0):.1%}")
            lines.append("")

    # Active runs
    active = snapshot.get("active_runs", [])
    if active:
        lines.extend(["## Active Training Runs", ""])
        for run in active:
            lines.append(f"- `{run['dir']}` (modified {run['age_seconds']:.0f}s ago)")
        lines.append("")

    # Best runs
    best_runs = snapshot.get("best_runs", [])
    if best_runs:
        lines.extend([
            "## Top Runs (by max reward)",
            "| Run | Max Reward | Last Reward | Distance | Reached | Steps |",
            "|-----|-----------|-------------|----------|---------|-------|",
        ])
        for r in best_runs[:5]:
            lines.append(
                f"| {r['dir'][:30]} | {r['reward_max']:.2f} | {r['reward_last']:.2f} | "
                f"{r['distance']:.2f} | {r['reached']:.2%} | {r['steps']:,} |"
            )
        lines.append("")

    # Action items for the agent
    lines.extend([
        "## Suggested Next Actions",
        "1. Check TensorBoard: `uv run tensorboard --logdir runs/vbot_navigation_section001`",
        "2. Evaluate best: `uv run scripts/play.py --env vbot_navigation_section001`",
        "3. Resume AutoML: `uv run starter_kit_schedule/scripts/automl.py --resume`",
        "4. Check reward curves and adjust weights if plateauing",
        "",
    ])

    WAKEUP_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ProgressWatcher] WAKE_UP.md written to {WAKEUP_PATH}")


def print_status(snapshot):
    """Print formatted status to console."""
    print(f"\n{'='*60}")
    print(f"  Training Progress — {snapshot['timestamp']}")
    print(f"{'='*60}")

    automl = snapshot.get("automl")
    if automl:
        print(f"\nAutoML: {automl.get('automl_id', 'N/A')}")
        print(f"  Status: {automl.get('status', 'N/A')}")
        print(f"  Phase: {automl.get('current_phase', 'N/A')}")
        print(f"  Iteration: {automl.get('current_iteration', 0)}")
        print(f"  Budget: {automl.get('elapsed_hours', 0):.1f}/{automl.get('budget_hours', 0):.1f}h")

    active = snapshot.get("active_runs", [])
    if active:
        print(f"\nActive runs: {len(active)}")
        for r in active:
            print(f"  - {r['dir']} ({r['age_seconds']:.0f}s ago)")

    best = snapshot.get("best_runs", [])
    if best:
        print(f"\nTop runs:")
        for r in best[:5]:
            print(f"  {r['dir'][:35]:35s} reward={r['reward_max']:.2f} dist={r['distance']:.2f} "
                  f"reached={r['reached']:.1%} steps={r['steps']:,}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Training Progress Watcher")
    parser.add_argument("--status", action="store_true", help="Print current status")
    parser.add_argument("--snapshot", action="store_true", help="Generate WAKE_UP.md snapshot")
    parser.add_argument("--watch", action="store_true", help="Continuous watch mode")
    parser.add_argument("--interval", type=int, default=120, help="Watch interval in seconds")
    args = parser.parse_args()

    if args.status:
        snapshot = generate_snapshot()
        print_status(snapshot)
    elif args.snapshot:
        snapshot = generate_snapshot()
        write_wakeup_md(snapshot)
        print_status(snapshot)
    elif args.watch:
        print(f"[ProgressWatcher] Starting watch mode (interval={args.interval}s)")
        while True:
            try:
                snapshot = generate_snapshot()
                write_wakeup_md(snapshot)
                print_status(snapshot)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n[ProgressWatcher] Stopped")
                break
    else:
        snapshot = generate_snapshot()
        print_status(snapshot)


if __name__ == "__main__":
    main()
