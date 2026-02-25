"""
Stage 3: Relaxed Termination — Batch Runner

Trains all 4 Stage 2 champions with relaxed termination conditions:
  - hard_tilt: 70° → 85°
  - soft_tilt: 50° → disabled
  - base_contact: disabled
  - stagnation: disabled
  - grace_period: 100 → 500 steps

Warm-starts from Stage 2 peak checkpoints with 0.3× LR.
"""

import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [
    "starter_kit_schedule/configs/stage3_relaxed_A_T4.json",
    "starter_kit_schedule/configs/stage3_relaxed_B_T10.json",
    "starter_kit_schedule/configs/stage3_relaxed_C_T6.json",
    "starter_kit_schedule/configs/stage3_relaxed_A_T13.json",
]


def main():
    root = Path(__file__).resolve().parent
    train_script = root / "starter_kit_schedule" / "scripts" / "train_one.py"

    for i, cfg_path in enumerate(CONFIGS, 1):
        full_path = root / cfg_path
        if not full_path.exists():
            print(f"[{i}/{len(CONFIGS)}] SKIP — config not found: {cfg_path}")
            continue

        tag = Path(cfg_path).stem
        print(f"\n{'='*60}")
        print(f"[{i}/{len(CONFIGS)}] Starting: {tag}")
        print(f"{'='*60}")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(train_script), "--config", str(full_path)],
            cwd=str(root),
        )
        elapsed = time.time() - t0
        minutes = elapsed / 60

        if result.returncode == 0:
            print(f"[{i}/{len(CONFIGS)}] DONE: {tag} ({minutes:.1f} min)")
        else:
            print(f"[{i}/{len(CONFIGS)}] FAILED (rc={result.returncode}): {tag} ({minutes:.1f} min)")

    print(f"\nAll {len(CONFIGS)} Stage 3 runs completed.")


if __name__ == "__main__":
    main()
