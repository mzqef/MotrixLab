# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""
Run hyperparameter search campaign.

Usage:
    uv run starter_kit_schedule/scripts/run_search.py
    uv run starter_kit_schedule/scripts/run_search.py --resume
    uv run starter_kit_schedule/scripts/run_search.py --plan path/to/plan.yaml
"""

import os
import sys
import json
import yaml
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from absl import app, flags

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FLAGS = flags.FLAGS

flags.DEFINE_string("plan", None, "Path to plan file (default: active_plan.yaml)")
flags.DEFINE_bool("resume", False, "Resume from last checkpoint")
flags.DEFINE_bool("dry_run", False, "Print commands without executing")
flags.DEFINE_integer("max_configs", None, "Maximum configs to run (for testing)")

SCHEDULE_DIR = PROJECT_ROOT / "starter_kit_schedule"
LOG_DIR = PROJECT_ROOT / "starter_kit_log"


def load_queue() -> dict:
    """Load the current queue state."""
    queue_path = SCHEDULE_DIR / "progress" / "queue.yaml"
    if queue_path.exists():
        with open(queue_path) as f:
            return yaml.safe_load(f)
    return None


def save_queue(queue: dict):
    """Save the queue state."""
    queue_path = SCHEDULE_DIR / "progress" / "queue.yaml"
    with open(queue_path, "w") as f:
        yaml.dump(queue, f, default_flow_style=False, allow_unicode=True)


def load_config(config_id: str) -> dict:
    """Load a configuration file."""
    config_path = SCHEDULE_DIR / "configs" / "generated" / f"{config_id}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_current_run(run_info: dict):
    """Save current run status."""
    run_path = SCHEDULE_DIR / "progress" / "current_run.yaml"
    with open(run_path, "w") as f:
        yaml.dump(run_info, f, default_flow_style=False, allow_unicode=True)


def create_experiment_dir(config_id: str, campaign_id: str) -> Path:
    """Create experiment directory for logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp_{timestamp}_{config_id}"
    exp_dir = LOG_DIR / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir, exp_id


def run_training(config: dict, exp_dir: Path, exp_id: str, 
                 resume_checkpoint: str | None = None) -> dict:
    """Run training for a single configuration."""
    
    env = config["environment"]
    hp = config["hyperparameters"]
    
    # Save frozen config to experiment directory
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # Build training command
    cmd = [
        "uv", "run", "scripts/train.py",
        "--env", env,
        "--num-envs", str(hp.get("num_envs", 2048)),
    ]
    
    if hp.get("seed"):
        cmd.extend(["--seed", str(hp["seed"])])
    
    if resume_checkpoint:
        cmd.extend(["--resume-from", resume_checkpoint])
    
    # Log the command
    with open(exp_dir / "events.log", "a") as f:
        f.write(f"{datetime.now().isoformat()}Z [INFO] Starting training\n")
        f.write(f"{datetime.now().isoformat()}Z [INFO] Command: {' '.join(cmd)}\n")
        f.write(f"{datetime.now().isoformat()}Z [INFO] Hyperparameters: {json.dumps(hp)}\n")
    
    if FLAGS.dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return {"status": "dry_run", "command": cmd}
    
    # Execute training
    start_time = time.time()
    
    try:
        # Run training process
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output to log file and console
        metrics_file = open(exp_dir / "metrics.jsonl", "a")
        events_file = open(exp_dir / "events.log", "a")
        
        for line in process.stdout:
            print(line, end="")  # Echo to console
            
            # Parse metrics from output (customize based on your training output format)
            if "reward" in line.lower() or "loss" in line.lower():
                try:
                    # Try to extract metrics - adjust parsing based on actual output
                    metrics_file.write(json.dumps({
                        "timestamp": datetime.now().isoformat() + "Z",
                        "raw_line": line.strip()
                    }) + "\n")
                    metrics_file.flush()
                except:
                    pass
            
            events_file.write(f"{datetime.now().isoformat()}Z [OUTPUT] {line}")
            events_file.flush()
        
        process.wait()
        metrics_file.close()
        events_file.close()
        
        elapsed_time = time.time() - start_time
        
        result = {
            "status": "completed" if process.returncode == 0 else "failed",
            "return_code": process.returncode,
            "elapsed_seconds": elapsed_time,
            "elapsed_hours": elapsed_time / 3600
        }
        
        # Write completion event
        with open(exp_dir / "events.log", "a") as f:
            f.write(f"{datetime.now().isoformat()}Z [INFO] Training {result['status']} "
                   f"(return code: {process.returncode}, elapsed: {elapsed_time:.1f}s)\n")
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        with open(exp_dir / "events.log", "a") as f:
            f.write(f"{datetime.now().isoformat()}Z [ERROR] Training failed with exception: {str(e)}\n")
        
        return {
            "status": "failed",
            "error": str(e),
            "elapsed_seconds": elapsed_time
        }


def write_experiment_summary(exp_dir: Path, exp_id: str, config: dict, result: dict):
    """Write experiment summary after completion."""
    summary = {
        "experiment_id": exp_id,
        "config_id": config["config_id"],
        "campaign_id": config.get("campaign_id"),
        "config": {
            "environment": config["environment"],
            "algorithm": "PPO",
            "hyperparameters": config["hyperparameters"]
        },
        "execution": {
            "started_at": (datetime.now().isoformat() + "Z"),
            "completed_at": (datetime.now().isoformat() + "Z"),
            "duration_hours": result.get("elapsed_hours", 0),
            "status": result["status"],
            "exit_reason": result.get("error", "completed")
        },
        "results": {
            "final_metrics": {},  # To be filled from metrics.jsonl
            "best_metrics": {},
            "convergence": {"converged": False}
        },
        "artifacts": {
            "experiment_dir": str(exp_dir),
            "config_file": str(exp_dir / "config.yaml"),
            "metrics_file": str(exp_dir / "metrics.jsonl"),
            "events_log": str(exp_dir / "events.log")
        }
    }
    
    with open(exp_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)


def update_index(exp_id: str, config: dict, result: dict):
    """Update the master index with experiment results."""
    index_path = LOG_DIR / "index.yaml"
    
    if index_path.exists():
        with open(index_path) as f:
            index = yaml.safe_load(f) or {"campaigns": [], "experiments": []}
    else:
        index = {"campaigns": [], "experiments": []}
    
    index["experiments"].append({
        "experiment_id": exp_id,
        "config_id": config["config_id"],
        "campaign_id": config.get("campaign_id"),
        "environment": config["environment"],
        "status": result["status"],
        "completed_at": datetime.now().isoformat() + "Z"
    })
    
    with open(index_path, "w") as f:
        yaml.dump(index, f, default_flow_style=False, allow_unicode=True)


def main(argv):
    # Load queue
    queue = load_queue()
    if queue is None:
        print("No campaign found. Run init_campaign.py first.")
        return
    
    # Check for pending configs
    if not queue["configs_pending"]:
        print("No pending configurations. Campaign may be complete.")
        print(f"Completed: {len(queue['configs_completed'])}")
        print(f"Failed: {len(queue['configs_failed'])}")
        return
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   Running Training Campaign                   ║
╠══════════════════════════════════════════════════════════════╣
║  Campaign: {queue['campaign_id']:<48} ║
║  Pending: {len(queue['configs_pending']):<49} ║
║  Completed: {len(queue['configs_completed']):<47} ║
║  Failed: {len(queue['configs_failed']):<50} ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Process configs
    max_configs = FLAGS.max_configs or len(queue["configs_pending"])
    configs_to_run = queue["configs_pending"][:max_configs]
    
    for config_id in configs_to_run:
        print(f"\n{'='*60}")
        print(f"Starting: {config_id}")
        print(f"{'='*60}")
        
        # Load config
        config = load_config(config_id)
        
        # Create experiment directory
        exp_dir, exp_id = create_experiment_dir(config_id, queue["campaign_id"])
        
        # Update queue - move to running
        queue["configs_pending"].remove(config_id)
        queue["configs_running"].append(config_id)
        queue["status"] = "running"
        save_queue(queue)
        
        # Save current run info
        save_current_run({
            "run_id": exp_id,
            "plan_id": queue["campaign_id"],
            "config_id": config_id,
            "status": "running",
            "started_at": datetime.now().isoformat() + "Z",
            "experiment_dir": str(exp_dir)
        })
        
        # Run training
        result = run_training(config, exp_dir, exp_id)
        
        # Update queue based on result
        queue["configs_running"].remove(config_id)
        if result["status"] == "completed" or result["status"] == "dry_run":
            queue["configs_completed"].append(config_id)
        else:
            queue["configs_failed"].append(config_id)
        save_queue(queue)
        
        # Write experiment summary
        write_experiment_summary(exp_dir, exp_id, config, result)
        
        # Update master index
        update_index(exp_id, config, result)
        
        print(f"\n{config_id}: {result['status']}")
        if "elapsed_hours" in result:
            print(f"  Duration: {result['elapsed_hours']:.2f} hours")
    
    # Final summary
    queue = load_queue()  # Reload to get final state
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                     Campaign Progress                         ║
╠══════════════════════════════════════════════════════════════╣
║  Completed: {len(queue['configs_completed']):<47} ║
║  Failed: {len(queue['configs_failed']):<50} ║
║  Remaining: {len(queue['configs_pending']):<47} ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    if not queue["configs_pending"]:
        queue["status"] = "completed"
        save_queue(queue)
        print("\n✓ Campaign completed!")
    else:
        print(f"\nRun again to continue with remaining {len(queue['configs_pending'])} configs.")


if __name__ == "__main__":
    app.run(main)
