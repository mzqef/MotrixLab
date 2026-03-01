"""Stage 8 continuation: Resume R1 (B1_T4) + fresh R3 (B1_T3).

R1 reached wp_mean=1.678 at 100M and was still climbing.
Resume from agent_48500.pt for another 100M steps.
Then run R3 fresh for 100M steps.
"""
import json
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "starter_kit_schedule" / "scripts" / "train_one.py"
STAGE8_LOG = PROJECT_ROOT / "starter_kit_log" / "stage8_runs"
CHECKPOINT_INTERVAL = 500

RUNS = [
    {
        "label": "S8_R1b_B1T4_resume",
        "source_config": "starter_kit_log/automl_20260227_184857/configs/automl_20260227_184857_vbot_navigation_section011_i4_1772194731.json",
        "checkpoint": "runs/vbot_navigation_section011/26-02-28_03-21-53-946093_PPO/checkpoints/agent_48500.pt",
        "max_env_steps": 100_000_000,
        "description": "Resume B1_T4 from 100M (wp_mean=1.678, still climbing)",
    },
    {
        "label": "S8_R3_B1T3",
        "source_config": "starter_kit_log/automl_20260227_184857/configs/automl_20260227_184857_vbot_navigation_section011_i3_1772193264.json",
        "checkpoint": None,
        "max_env_steps": 100_000_000,
        "description": "Fresh B1_T3 cold-start 100M",
    },
]


def main():
    STAGE8_LOG.mkdir(parents=True, exist_ok=True)

    for i, run in enumerate(RUNS):
        label = run["label"]
        if run.get("done"):
            print(f"\nSkipping {label} (done)")
            continue

        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(RUNS)}: {label}")
        print(f"{run['description']}")
        print(f"{'='*60}")

        with open(PROJECT_ROOT / run["source_config"]) as f:
            cfg = json.load(f)

        cfg["run_tag"] = f"stage8_{label}_{int(time.time())}"
        cfg["rl_overrides"]["max_env_steps"] = run["max_env_steps"]
        cfg["rl_overrides"]["check_point_interval"] = CHECKPOINT_INTERVAL

        if run["checkpoint"]:
            cfg["checkpoint"] = str(PROJECT_ROOT / run["checkpoint"])

        config_path = STAGE8_LOG / f"{label}_config.json"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"Config: {config_path}")
        if run["checkpoint"]:
            print(f"Warm-start: {run['checkpoint']}")
        print(f"Steps: {run['max_env_steps']:,}")

        cmd = ["uv", "run", str(TRAIN_SCRIPT), "--config", str(config_path)]
        start = time.time()

        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=14400)

        elapsed = time.time() - start
        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"\n{label}: {status} in {elapsed/60:.1f} min")

    print(f"\n{'='*60}")
    print("Stage 8 continuation complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
