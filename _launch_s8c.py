"""Stage 8c: Full training on B2 top trials (T3 and T11).

R1b (B1_T4 resume) peaked at wp_mean=3.838 then collapsed to 1.726.
B2_T2 collapsed at 48%. 
Now trying the next two B2 trials for 100M steps each.

B2_T3: score=0.240, wp_mean=0.222, success=0.212
B2_T11: score=0.240, wp_mean=0.218, success=0.238
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
        "label": "S8_R4_B2T3",
        "source_config": "starter_kit_log/automl_20260227_215755/configs/automl_20260227_215755_vbot_navigation_section011_i3_1772208042.json",
        "checkpoint": None,
        "max_env_steps": 100_000_000,
        "description": "B2_T3 fresh 100M (score=0.240, wp_mean=0.222)",
    },
    {
        "label": "S8_R5_B2T11",
        "source_config": "starter_kit_log/automl_20260227_215755/configs/automl_20260227_215755_vbot_navigation_section011_i11_1772219620.json",
        "checkpoint": None,
        "max_env_steps": 100_000_000,
        "description": "B2_T11 fresh 100M (score=0.240, wp_mean=0.218)",
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
        print(f"Steps: {run['max_env_steps']:,}")

        cmd = ["uv", "run", str(TRAIN_SCRIPT), "--config", str(config_path)]
        start = time.time()

        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=14400)

        elapsed = time.time() - start
        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"\n{label}: {status} in {elapsed/60:.1f} min")

    print(f"\n{'='*60}")
    print("Stage 8c complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
