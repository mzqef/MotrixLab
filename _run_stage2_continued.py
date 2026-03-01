"""Stage 2 continued: Full-train AVG and T0 (100M steps each)."""
import subprocess, sys, time, os

CONFIGS = [
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
    print("  Stage 2 Summary (AVG + T0)")
    print("="*60)
    for label, status, elapsed in results:
        print(f"  {label:5s}: {status:10s}  ({elapsed/3600:.1f}h)")
    print("="*60)

if __name__ == "__main__":
    main()
