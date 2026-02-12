#!/usr/bin/env python3
"""
Training Monitor & TensorBoard Analyzer
========================================
Unified tool for monitoring live training runs and analyzing TensorBoard logs.

Replaces: _monitor_v7.py, _monitor_section011.py, _monitor_stage2.py,
           _deep_analysis.py, _check_latest.py, _read_tb.py, etc.

Usage:
    # Monitor latest run for an env (auto-detect latest run directory)
    uv run starter_kit_schedule/scripts/monitor_training.py --env vbot_navigation_section011

    # Deep analysis with full reward breakdown
    uv run starter_kit_schedule/scripts/monitor_training.py --env vbot_navigation_section011 --deep

    # Monitor a specific run directory
    uv run starter_kit_schedule/scripts/monitor_training.py --run runs/vbot_navigation_section011/26-02-12_15-01-28_PPO

    # List all TensorBoard tags
    uv run starter_kit_schedule/scripts/monitor_training.py --env vbot_navigation_section011 --tags

    # Compare two runs side-by-side
    uv run starter_kit_schedule/scripts/monitor_training.py --compare runs/.../run_A runs/.../run_B
"""

import argparse
import glob
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]  # d:\MotrixLab


# ============================================================
# TensorBoard helpers
# ============================================================

def _get_event_accumulator(run_dir: str):
    """Load TensorBoard EventAccumulator for a run directory."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("ERROR: tensorboard not installed. Run: uv sync --all-packages --all-extras")
        sys.exit(1)
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    return ea


def _find_latest_run(env_name: str) -> str | None:
    """Find the most recent run directory for an environment."""
    runs_dir = WORKSPACE / "runs" / env_name
    if not runs_dir.exists():
        return None
    dirs = sorted(runs_dir.iterdir(), key=lambda d: d.name, reverse=True)
    for d in dirs:
        if d.is_dir() and any(d.glob("events.out.tfevents.*")):
            return str(d)
    return None


def _get_scalars(ea, tag: str) -> list[tuple[int, float]]:
    """Get (step, value) pairs for a tag."""
    try:
        events = ea.Scalars(tag)
        return [(e.step, e.value) for e in events]
    except Exception:
        return []


def _trend_arrow(early_vals: list[float], recent_vals: list[float], threshold: float = 0.01) -> str:
    """Return ↑/↓/→ trend arrow from early vs recent averages."""
    if not early_vals or not recent_vals:
        return "?"
    early = sum(early_vals) / len(early_vals)
    recent = sum(recent_vals) / len(recent_vals)
    if recent > early + threshold:
        return "↑"
    elif recent < early - threshold:
        return "↓"
    return "→"


# ============================================================
# Monitor mode: compact live-training dashboard
# ============================================================

def monitor(run_dir: str, env_name: str = ""):
    """Display compact training progress dashboard."""
    ea = _get_event_accumulator(run_dir)
    tags = ea.Tags().get("scalars", [])
    run_name = os.path.basename(run_dir)
    print(f"=== Training Monitor: {run_name} ===")
    if env_name:
        print(f"    Environment: {env_name}")

    # --- Progress ---
    reward_tag = "Reward / Instantaneous (mean)"
    if reward_tag not in tags:
        # Try alternate name
        for t in tags:
            if "Reward" in t and "Instantaneous" in t and "mean" in t:
                reward_tag = t
                break

    if reward_tag in tags:
        events = ea.Scalars(reward_tag)
        if events:
            total_steps = events[-1].step
            total_iters = len(events)
            # Try to detect max_env_steps from config
            max_steps = _detect_max_steps(env_name)
            pct = total_steps / max_steps * 100 if max_steps else 0
            print(f"\n--- Progress ---")
            print(f"  Iterations: {total_iters:,}")
            print(f"  Env steps:  {total_steps:,} / {max_steps:,} ({pct:.1f}%)" if max_steps else f"  Env steps: {total_steps:,}")

    # --- Key Metrics ---
    key_metrics = [
        ("Reward / Instantaneous (mean)", "Avg Reward"),
        ("Episode / Total length (mean)", "Avg Ep Length"),
        ("metrics / wp_idx_mean (mean)", "WP Index"),
        ("metrics / distance_to_target (mean)", "Dist to Target"),
        ("metrics / reached_fraction (mean)", "Reached %"),
        ("Reward Total/ alive_bonus (mean)", "Alive Bonus"),
        ("Reward Total/ forward_velocity (mean)", "Fwd Velocity Rwd"),
        ("Reward Total/ wp_approach (mean)", "WP Approach"),
        ("Reward Total/ wp_bonus (mean)", "WP Bonus"),
        ("Reward Total/ smiley_bonus (mean)", "Smiley Bonus"),
        ("Reward Total/ red_packet_bonus (mean)", "Red Packet Bonus"),
        ("Reward Total/ celeb_bonus (mean)", "Celeb Bonus"),
        ("Reward Total/ height_progress (mean)", "Height Progress"),
        ("Reward Total/ traversal_bonus (mean)", "Traversal Bonus"),
        ("Reward Total/ foot_clearance (mean)", "Foot Clearance"),
        ("Reward Total/ termination (mean)", "Termination"),
        ("Reward Total/ score_clear_penalty (mean)", "Score Clear"),
    ]

    print(f"\n{'Metric':<22} {'First':>11} {'Latest':>11} {'Trend':>5} {'Steps':>10}")
    print("-" * 65)
    for tag, label in key_metrics:
        # Fuzzy match: try exact first, then partial
        matched_tag = _fuzzy_tag(tags, tag)
        if matched_tag:
            events = ea.Scalars(matched_tag)
            if events:
                vals = [e.value for e in events]
                first, last = vals[0], vals[-1]
                arrow = _trend_arrow(vals[:3], vals[-3:])
                step = events[-1].step
                print(f"  {label:<20} {first:>11.4f} {last:>11.4f} {arrow:>5} {step:>10d}")

    # --- Checkpoints ---
    ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "agent_*.pt"))
    if not ckpts:
        ckpts = glob.glob(os.path.join(run_dir, "agent_*.pt"))
    best_ckpt = glob.glob(os.path.join(run_dir, "checkpoints", "best_agent.pt"))
    print(f"\n  Checkpoints: {len(ckpts)} saved" + (" (best_agent.pt exists)" if best_ckpt else ""))

    # --- Quick diagnosis ---
    _quick_diagnosis(ea, tags)


# ============================================================
# Deep analysis mode: comprehensive reward & policy breakdown
# ============================================================

def deep_analysis(run_dir: str, env_name: str = ""):
    """Comprehensive TensorBoard analysis with reward breakdown and auto-diagnosis."""
    ea = _get_event_accumulator(run_dir)
    tags = ea.Tags().get("scalars", [])
    run_name = os.path.basename(run_dir)

    print(f"{'='*70}")
    print(f"DEEP ANALYSIS: {run_name}")
    if env_name:
        print(f"Environment: {env_name}")
    print(f"{'='*70}")

    # --- Episode Statistics ---
    print(f"\n{'='*70}")
    print("EPISODE STATISTICS")
    print(f"{'='*70}")
    ep_tags = sorted(t for t in tags if "Episode" in t or "episode" in t)
    for tag in ep_tags:
        events = ea.Scalars(tag)
        if events:
            vals = [e.value for e in events]
            label = tag.split("/")[-1].strip() if "/" in tag else tag
            print(f"  {label:<40} first={vals[0]:.2f}  last={vals[-1]:.2f}  min={min(vals):.2f}  max={max(vals):.2f}")

    # --- Reward Components (with trend) ---
    print(f"\n{'='*70}")
    print("REWARD COMPONENTS (mean values)")
    print(f"{'='*70}")
    reward_tags = sorted(t for t in tags if "Reward Total/" in t and "(mean)" in t)
    for tag in reward_tags:
        events = ea.Scalars(tag)
        if events:
            vals = [e.value for e in events]
            label = tag.replace("Reward Total/ ", "").replace(" (mean)", "")
            step = events[-1].step
            early = sum(vals[:3]) / max(len(vals[:3]), 1)
            recent = sum(vals[-3:]) / max(len(vals[-3:]), 1)
            arrow = _trend_arrow(vals[:3], vals[-3:])
            print(f"  {label:<35} {early:>10.4f} → {recent:>10.4f} {arrow}  (step {step})")

    # --- Reward Instantaneous ---
    print(f"\n{'='*70}")
    print("REWARD INSTANTANEOUS")
    print(f"{'='*70}")
    for suffix in ["(mean)", "(max)", "(min)"]:
        tag = f"Reward / Instantaneous {suffix}"
        matched = _fuzzy_tag(tags, tag)
        if matched:
            events = ea.Scalars(matched)
            if events:
                vals = [e.value for e in events]
                print(f"  {suffix:<15} first={vals[0]:>10.2f}  last={vals[-1]:>10.2f}  best={max(vals):>10.2f}")

    # --- Custom Metrics ---
    print(f"\n{'='*70}")
    print("METRICS")
    print(f"{'='*70}")
    metric_tags = sorted(t for t in tags if "metrics" in t.lower())
    for tag in metric_tags:
        events = ea.Scalars(tag)
        if events:
            vals = [e.value for e in events]
            label = tag.split("/")[-1].strip()
            early = sum(vals[:3]) / max(len(vals[:3]), 1)
            recent = sum(vals[-3:]) / max(len(vals[-3:]), 1)
            arrow = _trend_arrow(vals[:3], vals[-3:], threshold=0.001)
            print(f"  {label:<40} {early:>10.4f} → {recent:>10.4f} {arrow}")

    # --- Policy Stats ---
    print(f"\n{'='*70}")
    print("POLICY STATS")
    print(f"{'='*70}")
    policy_tags = sorted(t for t in tags if any(k in t.lower() for k in ["loss", "entropy", "lr", "learning", "kl", "std"]))
    for tag in policy_tags[:15]:
        events = ea.Scalars(tag)
        if events:
            vals = [e.value for e in events]
            label = tag.split("/")[-1].strip() if "/" in tag else tag
            print(f"  {label:<40} first={vals[0]:.6f}  last={vals[-1]:.6f}")

    # --- Diagnosis ---
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")
    _quick_diagnosis(ea, tags, verbose=True)


# ============================================================
# Compare mode: side-by-side run comparison
# ============================================================

def compare_runs(run_dirs: list[str]):
    """Compare key metrics across multiple runs side-by-side."""
    labels = [os.path.basename(d) for d in run_dirs]
    eas = [_get_event_accumulator(d) for d in run_dirs]
    all_tags = [ea.Tags().get("scalars", []) for ea in eas]

    compare_metrics = [
        ("Reward / Instantaneous (mean)", "Avg Reward"),
        ("Episode / Total length (mean)", "Avg Ep Length"),
        ("metrics / wp_idx_mean (mean)", "WP Index"),
        ("metrics / distance_to_target (mean)", "Dist to Target"),
        ("Reward Total/ termination (mean)", "Termination"),
        ("Reward Total/ alive_bonus (mean)", "Alive Bonus"),
        ("Reward Total/ forward_velocity (mean)", "Fwd Velocity"),
    ]

    # Header
    col_w = max(len(l) for l in labels) + 2
    col_w = max(col_w, 14)
    header = f"  {'Metric':<22}"
    for label in labels:
        header += f" {label[:col_w-1]:>{col_w}}"
    print(f"{'='*70}")
    print("RUN COMPARISON")
    print(f"{'='*70}")
    print(header)
    print("-" * (24 + col_w * len(labels)))

    for tag_pattern, label in compare_metrics:
        row = f"  {label:<22}"
        for i, (ea, tags) in enumerate(zip(eas, all_tags)):
            matched = _fuzzy_tag(tags, tag_pattern)
            if matched:
                events = ea.Scalars(matched)
                if events:
                    row += f" {events[-1].value:>{col_w}.4f}"
                else:
                    row += f" {'N/A':>{col_w}}"
            else:
                row += f" {'N/A':>{col_w}}"
        print(row)


# ============================================================
# List tags mode
# ============================================================

def list_tags(run_dir: str):
    """List all TensorBoard scalar tags with their latest value."""
    ea = _get_event_accumulator(run_dir)
    tags = ea.Tags().get("scalars", [])
    print(f"Tags in {os.path.basename(run_dir)} ({len(tags)} total):\n")
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        if events:
            print(f"  {tag:<60} last={events[-1].value:.6f}  (n={len(events)})")


# ============================================================
# Helpers
# ============================================================

def _fuzzy_tag(tags: list[str], pattern: str) -> str | None:
    """Find a tag matching a pattern (exact → partial)."""
    if pattern in tags:
        return pattern
    # Partial match
    pattern_lower = pattern.lower()
    for tag in tags:
        if pattern_lower in tag.lower():
            return tag
    return None


def _detect_max_steps(env_name: str) -> int | None:
    """Try to detect max_env_steps from RL config or default."""
    try:
        from motrix_rl.registry import default_rl_cfg
        rlcfg = default_rl_cfg(env_name, "ppo", "torch")
        return getattr(rlcfg, 'max_env_steps', None) or getattr(rlcfg, 'timesteps', None)
    except Exception:
        pass
    # Common defaults
    defaults = {
        "vbot_navigation_section001": 80_000_000,
        "vbot_navigation_section011": 80_000_000,
        "vbot_navigation_section012": 80_000_000,
        "vbot_navigation_long_course": 100_000_000,
    }
    return defaults.get(env_name)


def _quick_diagnosis(ea, tags: list[str], verbose: bool = False):
    """Print auto-diagnosis based on metrics."""
    # Termination rate
    term_tag = _fuzzy_tag(tags, "Reward Total/ termination (mean)")
    if term_tag:
        events = ea.Scalars(term_tag)
        if events:
            vals = [e.value for e in events]
            # Estimate termination rate from penalty magnitude
            # (base_termination can be -50 or -100, approximate)
            approx_rate = min(abs(vals[-1]) / 100.0, 1.0)
            print(f"  Termination rate (approx): {approx_rate*100:.1f}%")

    # Navigation progress
    wp_tag = _fuzzy_tag(tags, "wp_idx_mean (mean)")
    if wp_tag:
        events = ea.Scalars(wp_tag)
        if events:
            vals = [e.value for e in events]
            if max(vals) > 0.01:
                print(f"  Navigation: STARTED (wp_idx max={max(vals):.4f})")
            else:
                print(f"  Navigation: NOT STARTED (wp_idx=0 at all {len(vals)} checkpoints)")

    # Episode length
    ep_tag = _fuzzy_tag(tags, "Episode / Total length (mean)")
    if ep_tag:
        events = ea.Scalars(ep_tag)
        if events:
            vals = [e.value for e in events]
            last = vals[-1]
            print(f"  Avg episode length: {vals[0]:.0f} → {last:.0f} steps")
            if last < 50:
                print(f"  ⚠️ VERY SHORT episodes — robot falls almost immediately!")
            elif last < 200:
                print(f"  ⚠️ SHORT episodes — robot falls quickly")
            elif last < 500:
                print(f"  Episodes moderate — robot surviving some time")
            elif last < 2000:
                print(f"  Episodes good — robot surviving well")
            else:
                print(f"  ✅ Episodes long — robot very stable")

    if verbose:
        # Alive bonus
        alive_tag = _fuzzy_tag(tags, "alive_bonus (mean)")
        if alive_tag:
            events = ea.Scalars(alive_tag)
            if events:
                print(f"  Alive bonus (episode total): {events[-1].value:.4f}")

        # Score clear
        sc_tag = _fuzzy_tag(tags, "score_clear_penalty (mean)")
        if sc_tag:
            events = ea.Scalars(sc_tag)
            if events:
                vals = [e.value for e in events]
                print(f"  Score clear penalty: first={vals[0]:.4f}  last={vals[-1]:.4f}")

        # Checkpoints
        reward_tag = _fuzzy_tag(tags, "Reward / Instantaneous (mean)")
        if reward_tag:
            print(f"  Total TB datapoints: {len(ea.Scalars(reward_tag))}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Training Monitor & TensorBoard Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick monitor of latest run
  uv run starter_kit_schedule/scripts/monitor_training.py --env vbot_navigation_section011

  # Deep analysis
  uv run starter_kit_schedule/scripts/monitor_training.py --env vbot_navigation_section011 --deep

  # List all TB tags
  uv run starter_kit_schedule/scripts/monitor_training.py --env vbot_navigation_section011 --tags

  # Compare two runs
  uv run starter_kit_schedule/scripts/monitor_training.py --compare runs/.../run_A runs/.../run_B
        """
    )
    parser.add_argument("--env", type=str, help="Environment name (auto-finds latest run)")
    parser.add_argument("--run", type=str, help="Specific run directory path")
    parser.add_argument("--deep", action="store_true", help="Deep analysis mode with full breakdown")
    parser.add_argument("--tags", action="store_true", help="List all TensorBoard tags")
    parser.add_argument("--compare", nargs="+", metavar="RUN_DIR", help="Compare multiple run directories")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare)
        return

    # Resolve run directory
    run_dir = args.run
    env_name = args.env or ""
    if not run_dir:
        if not env_name:
            parser.error("Either --env or --run is required")
        run_dir = _find_latest_run(env_name)
        if not run_dir:
            print(f"No runs found for env '{env_name}' in {WORKSPACE / 'runs' / env_name}")
            sys.exit(1)

    if not os.path.isdir(run_dir):
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    if args.tags:
        list_tags(run_dir)
    elif args.deep:
        deep_analysis(run_dir, env_name)
    else:
        monitor(run_dir, env_name)


if __name__ == "__main__":
    main()
