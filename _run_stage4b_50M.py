"""Stage 4b: 50M Continuation — extend A_T4 from Stage 4 peak.

Single config, ~55 min at ~15k steps/sec.
"""
import subprocess
import sys
import time

CONFIG = "starter_kit_schedule/configs/stage4b_right_turn_A_T4_50M.json"

def main():
    t0 = time.time()
    print(f"Starting Stage 4b: 50M continuation for A_T4")
    print(f"  config: {CONFIG}")
    result = subprocess.run(
        [sys.executable, "starter_kit_schedule/scripts/train_one.py", "--config", CONFIG],
        cwd=".",
    )
    if result.returncode != 0:
        print(f"WARNING: exited with code {result.returncode}")
    print(f"Stage 4b completed in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
