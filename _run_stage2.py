"""
Stage 2: Full-train T1, T0, AVG sequentially (100M steps each).
Run from project root: uv run python _run_stage2.py
"""
import subprocess
import sys
import time
import os

CONFIGS = [
    ("T1",  "starter_kit_schedule/configs_full_train/s012_stage2_T1.json"),
    ("AVG", "starter_kit_schedule/configs_full_train/s012_stage2_AVG.json"),
    ("T0",  "starter_kit_schedule/configs_full_train/s012_stage2_T0.json"),
]

def main():
    results = []
    for label, cfg in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Stage 2 — Starting {label}")
        print(f"  Config: {cfg}")
        print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        t0 = time.time()
        ret = subprocess.run(
            [sys.executable, "starter_kit_schedule/scripts/train_one.py", "--config", cfg],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        elapsed = time.time() - t0
        status = "OK" if ret.returncode == 0 else f"FAIL(rc={ret.returncode})"
        results.append((label, status, elapsed))
        print(f"\n>>> {label} finished: {status} in {elapsed/3600:.1f}h\n")
    
    print("\n" + "="*60)
    print("  Stage 2 Summary")
    print("="*60)
    for label, status, elapsed in results:
        print(f"  {label:5s}: {status:10s}  ({elapsed/3600:.1f}h)")
    print("="*60)

if __name__ == "__main__":
    main()
