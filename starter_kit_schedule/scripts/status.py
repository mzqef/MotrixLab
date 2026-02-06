# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""
Check status of training campaign.

Usage:
    uv run starter_kit_schedule/scripts/status.py
    uv run starter_kit_schedule/scripts/status.py --watch
    uv run starter_kit_schedule/scripts/status.py --verbose
"""

import os
import sys
import yaml
import json
import time
from datetime import datetime
from pathlib import Path

from absl import app, flags

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FLAGS = flags.FLAGS

flags.DEFINE_bool("watch", False, "Continuously watch status")
flags.DEFINE_bool("verbose", False, "Show detailed information")
flags.DEFINE_integer("interval", 30, "Watch interval in seconds")

SCHEDULE_DIR = PROJECT_ROOT / "starter_kit_schedule"
LOG_DIR = PROJECT_ROOT / "starter_kit_log"


def load_yaml_safe(path: Path) -> dict | None:
    """Safely load a YAML file."""
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return None


def format_duration(hours: float) -> str:
    """Format duration in human-readable format."""
    if hours < 1:
        return f"{hours * 60:.1f}m"
    elif hours < 24:
        return f"{hours:.1f}h"
    else:
        return f"{hours / 24:.1f}d"


def print_status():
    """Print current campaign status."""
    
    # Load queue
    queue = load_yaml_safe(SCHEDULE_DIR / "progress" / "queue.yaml")
    if not queue:
        print("No active campaign found.")
        print("Run: uv run starter_kit_schedule/scripts/automl.py --mode stage --env <env_name> --budget-hours 12 --hp-trials 8")
        return
    
    # Load current run
    current_run = load_yaml_safe(SCHEDULE_DIR / "progress" / "current_run.yaml")
    
    # Load plan
    plan = load_yaml_safe(SCHEDULE_DIR / "plans" / "active_plan.yaml")
    
    # Calculate progress
    total = (len(queue.get("configs_pending", [])) + 
             len(queue.get("configs_running", [])) + 
             len(queue.get("configs_completed", [])) + 
             len(queue.get("configs_failed", [])))
    
    completed = len(queue.get("configs_completed", []))
    failed = len(queue.get("configs_failed", []))
    running = len(queue.get("configs_running", []))
    pending = len(queue.get("configs_pending", []))
    
    progress_pct = (completed / total * 100) if total > 0 else 0
    
    # Status indicator
    status_emoji = {
        "queued": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "paused": "⏸️"
    }
    
    status = queue.get("status", "unknown")
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Training Campaign Status                   ║
╠══════════════════════════════════════════════════════════════╣
║  Campaign: {queue.get('campaign_id', 'N/A'):<48} ║
║  Status: {status_emoji.get(status, '❓')} {status:<48} ║
║  Progress: {progress_pct:5.1f}% [{completed}/{total}]                                   ║
╠══════════════════════════════════════════════════════════════╣
║  Completed: {completed:<10} Failed: {failed:<10} Running: {running:<10}   ║
║  Pending: {pending:<49} ║
╚══════════════════════════════════════════════════════════════╝""")
    
    # Current run details
    if current_run and status == "running":
        print(f"""
┌──────────────────────────────────────────────────────────────┐
│  Current Run                                                  │
├──────────────────────────────────────────────────────────────┤
│  Run ID: {current_run.get('run_id', 'N/A'):<50} │
│  Config: {current_run.get('config_id', 'N/A'):<50} │
│  Started: {current_run.get('started_at', 'N/A'):<49} │
└──────────────────────────────────────────────────────────────┘""")
    
    if FLAGS.verbose:
        # Show completed configs with results
        if queue.get("configs_completed"):
            print("\n📊 Completed Configurations:")
            print("-" * 60)
            
            for config_id in queue["configs_completed"][-5:]:  # Last 5
                # Find experiment summary
                exp_dirs = list(LOG_DIR.glob(f"experiments/exp_*_{config_id}"))
                if exp_dirs:
                    summary = load_yaml_safe(exp_dirs[0] / "summary.yaml")
                    if summary:
                        duration = summary.get("execution", {}).get("duration_hours", 0)
                        print(f"  {config_id}: {format_duration(duration)}")
        
        # Show failed configs
        if queue.get("configs_failed"):
            print("\n❌ Failed Configurations:")
            print("-" * 60)
            for config_id in queue["configs_failed"]:
                print(f"  {config_id}")
    
    # Show next steps
    print("\n")
    if status == "completed":
        print("✅ Campaign completed! Run analysis:")
        print("   uv run starter_kit_schedule/scripts/analyze.py")
    elif pending > 0:
        print(f"📋 {pending} configs remaining. To continue:")
        print("   uv run starter_kit_schedule/scripts/automl.py --mode stage")
    elif running > 0:
        print("🔄 Training in progress...")
        print("   TensorBoard: uv run tensorboard --logdir starter_kit_log/experiments/")


def main(argv):
    if FLAGS.watch:
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print_status()
                print(f"\nRefreshing in {FLAGS.interval}s... (Ctrl+C to stop)")
                time.sleep(FLAGS.interval)
        except KeyboardInterrupt:
            print("\nStopped watching.")
    else:
        print_status()


if __name__ == "__main__":
    app.run(main)
