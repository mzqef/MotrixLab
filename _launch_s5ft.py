"""
Stage 5 Full Training (S5FT) — Launch full 100M-step deployment runs
from all 6 S5 AutoML trial best checkpoints.

Each trial keeps its own reward scales + HP config from the AutoML search,
but with:
  - checkpoint = trial's own best_agent.pt
  - max_env_steps = 100_000_000
  - learning_rate halved (0.5x) for warm-start stability
  - check_point_interval = 500 (frequent saves to catch peak)
  - freeze_preprocessor = true (prevent normalizer drift)
  - env_overrides unchanged (relaxed termination)

Runs sequentially (one at a time) to avoid GPU contention.
"""

import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# S5 AutoML trial configs (i0..i5) and their run directories
S5_AUTOML_DIR = "starter_kit_log/automl_20260226_173838"
S5_TRIALS = [
    {
        "trial": "T0",
        "config": f"{S5_AUTOML_DIR}/configs/automl_20260226_173838_vbot_navigation_section011_i0_1772098718.json",
        "run_dir": "runs/vbot_navigation_section011/26-02-26_17-38-41-791146_PPO",
    },
    {
        "trial": "T1",
        "config": f"{S5_AUTOML_DIR}/configs/automl_20260226_173838_vbot_navigation_section011_i1_1772100005.json",
        "run_dir": "runs/vbot_navigation_section011/26-02-26_18-00-08-446039_PPO",
    },
    {
        "trial": "T2",
        "config": f"{S5_AUTOML_DIR}/configs/automl_20260226_173838_vbot_navigation_section011_i2_1772101274.json",
        "run_dir": "runs/vbot_navigation_section011/26-02-26_18-21-17-770787_PPO",
    },
    {
        "trial": "T3",
        "config": f"{S5_AUTOML_DIR}/configs/automl_20260226_173838_vbot_navigation_section011_i3_1772102557.json",
        "run_dir": "runs/vbot_navigation_section011/26-02-26_18-42-40-540723_PPO",
    },
    {
        "trial": "T4",
        "config": f"{S5_AUTOML_DIR}/configs/automl_20260226_173838_vbot_navigation_section011_i4_1772103834.json",
        "run_dir": "runs/vbot_navigation_section011/26-02-26_19-03-57-147785_PPO",
    },
    {
        "trial": "T5",
        "config": f"{S5_AUTOML_DIR}/configs/automl_20260226_173838_vbot_navigation_section011_i5_1772105107.json",
        "run_dir": "runs/vbot_navigation_section011/26-02-26_19-25-10-275196_PPO",
    },
]

MAX_ENV_STEPS = 100_000_000
LR_MULTIPLIER = 0.5  # Halve LR for warm-start stability
CHECKPOINT_INTERVAL = 500


def main():
    os.chdir(PROJECT_ROOT)

    # Prepare output directory for S5FT configs
    s5ft_config_dir = os.path.join("starter_kit_log", "s5ft_configs")
    os.makedirs(s5ft_config_dir, exist_ok=True)

    results = []

    for trial_info in S5_TRIALS:
        trial_name = trial_info["trial"]
        config_path = trial_info["config"]
        run_dir = trial_info["run_dir"]
        ckpt_path = os.path.join(run_dir, "checkpoints", "best_agent.pt")

        if not os.path.exists(ckpt_path):
            print(f"[S5FT] SKIP {trial_name}: no checkpoint at {ckpt_path}")
            continue

        # Load original AutoML config
        with open(config_path) as f:
            cfg = json.load(f)

        # Modify for full training
        original_lr = cfg["rl_overrides"].get("learning_rate", 0.0004)
        new_lr = original_lr * LR_MULTIPLIER

        cfg["run_tag"] = f"s5ft_{trial_name}"
        cfg["checkpoint"] = ckpt_path
        cfg["freeze_preprocessor"] = True
        cfg["rl_overrides"]["max_env_steps"] = MAX_ENV_STEPS
        cfg["rl_overrides"]["learning_rate"] = new_lr
        cfg["rl_overrides"]["check_point_interval"] = CHECKPOINT_INTERVAL

        # Save modified config
        out_config = os.path.join(s5ft_config_dir, f"s5ft_{trial_name}.json")
        with open(out_config, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"\n{'='*60}")
        print(f"[S5FT] Starting {trial_name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  LR: {original_lr:.6f} → {new_lr:.6f} (0.5×)")
        print(f"  Max steps: {MAX_ENV_STEPS:,}")
        print(f"  Checkpoint interval: {CHECKPOINT_INTERVAL}")
        print(f"{'='*60}")

        start = time.time()
        result = subprocess.run(
            [sys.executable, "starter_kit_schedule/scripts/train_one.py", "--config", out_config],
            cwd=PROJECT_ROOT,
        )
        elapsed = time.time() - start

        status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
        print(f"[S5FT] {trial_name} finished: {status} ({elapsed/3600:.1f}h)")
        results.append({"trial": trial_name, "status": status, "elapsed_h": elapsed / 3600})

    # Summary
    print(f"\n{'='*60}")
    print("[S5FT] SUMMARY")
    for r in results:
        print(f"  {r['trial']}: {r['status']} ({r['elapsed_h']:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
