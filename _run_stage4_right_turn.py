"""Stage 4: Right-Turn Celebration — warm-start from Stage 3 peaks.

Runs 4 configs sequentially (each ~22 min, total ~90 min).
"""
import subprocess
import sys
import time

CONFIGS = [
    "starter_kit_schedule/configs/stage4_right_turn_A_T4.json",
    "starter_kit_schedule/configs/stage4_right_turn_B_T10.json",
    "starter_kit_schedule/configs/stage4_right_turn_C_T6.json",
    "starter_kit_schedule/configs/stage4_right_turn_A_T13.json",
]

def main():
    t0 = time.time()
    for i, cfg_path in enumerate(CONFIGS):
        label = cfg_path.split("_")[-1].replace(".json", "")
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(CONFIGS)}] Starting {label}  (elapsed: {(time.time()-t0)/60:.1f} min)")
        print(f"  config: {cfg_path}")
        print(f"{'='*60}\n")

        result = subprocess.run(
            [sys.executable, "starter_kit_schedule/scripts/train_one.py", "--config", cfg_path],
            cwd=".",
        )
        if result.returncode != 0:
            print(f"WARNING: {label} exited with code {result.returncode}")
        print(f"[{i+1}/{len(CONFIGS)}] {label} finished in {(time.time()-t0)/60:.1f} min total")

    print(f"\nAll Stage 4 runs completed in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
