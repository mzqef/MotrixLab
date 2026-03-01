"""Stage 8: Full 100M Training of top 3 Stage 7 configs.

Hard termination, NO random yaw. Runs sequentially.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CONFIGS = [
    {
        "label": "S8_R1_B1T4",
        "source": "B1_T4 (wp_max=3, wp_mean=0.489)",
        "config_path": "starter_kit_log/automl_20260227_184857/configs/automl_20260227_184857_vbot_navigation_section011_i4_1772194731.json",
        "done": True,  # Completed: wp_max=4, wp_mean=1.678, smiley%=77.1%
    },
    {
        "label": "S8_R2_B2T2",
        "source": "B2_T2 (wp_max=3, wp_mean=0.448)",
        "config_path": "starter_kit_log/automl_20260227_215755/configs/automl_20260227_215755_vbot_navigation_section011_i2_1772205027.json",
    },
    {
        "label": "S8_R3_B1T3",
        "source": "B1_T3 (wp_max=2, wp_mean=0.239)",
        "config_path": "starter_kit_log/automl_20260227_184857/configs/automl_20260227_184857_vbot_navigation_section011_i3_1772193264.json",
        "done": True,  # Deferred — will run after R1 resume
    },
]

MAX_ENV_STEPS = 100_000_000
CHECKPOINT_INTERVAL = 500  # ~500 iters between checkpoints


def main():
    train_script = PROJECT_ROOT / "starter_kit_schedule" / "scripts" / "train_one.py"
    stage8_log = PROJECT_ROOT / "starter_kit_log" / "stage8_runs"
    stage8_log.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(CONFIGS):
        label = entry["label"]
        source_cfg_path = PROJECT_ROOT / entry["config_path"]

        # Skip already-completed runs
        if entry.get("done"):
            print(f"\nSkipping {label} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"Stage 8 Run {i+1}/3: {label}")
        print(f"Source: {entry['source']}")
        print(f"{'='*60}")

        # Load original config
        with open(source_cfg_path) as f:
            cfg = json.load(f)

        # Override for Stage 8: 100M steps, reasonable checkpoint interval
        cfg["run_tag"] = f"stage8_{label}_{int(time.time())}"
        cfg["rl_overrides"]["max_env_steps"] = MAX_ENV_STEPS
        cfg["rl_overrides"]["check_point_interval"] = CHECKPOINT_INTERVAL

        # Write Stage 8 config
        s8_config_path = stage8_log / f"{label}_config.json"
        with open(s8_config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"Config: {s8_config_path}")
        print(f"Steps: {MAX_ENV_STEPS:,}")
        print(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")

        # Launch training
        cmd = ["uv", "run", str(train_script), "--config", str(s8_config_path)]
        start = time.time()

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=14400,  # 4 hour timeout per run
        )

        elapsed = time.time() - start
        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"\n{label}: {status} in {elapsed/60:.1f} min")

    print(f"\n{'='*60}")
    print("Stage 8 complete. Check runs/vbot_navigation_section011/ for results.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
