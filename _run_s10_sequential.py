"""Stage 10: Sequential full training — T2 then T8.

Waits for T2 to finish (polls stdout), then launches T8.
Both warm-start from agent_15000.pt (R1b peak).
"""
import subprocess
import sys
import time
import os

def wait_for_completion(stdout_file, label, poll_interval=30):
    """Poll stdout file until training is done (progress bar reaches 100% or PIPELINE_RUN_DIR appears)."""
    print(f"[{label}] Waiting for completion...")
    while True:
        time.sleep(poll_interval)
        try:
            with open(stdout_file, 'r') as f:
                content = f.read()
            if 'PIPELINE_RUN_DIR=' in content:
                print(f"[{label}] COMPLETED!")
                # Extract run dir
                for line in content.split('\n'):
                    if line.startswith('PIPELINE_RUN_DIR='):
                        print(f"  {line}")
                    if line.startswith('PIPELINE_ELAPSED='):
                        print(f"  {line}")
                return True
            # Check if process is still alive
            lines = content.strip().split('\n')
            last_line = lines[-1] if lines else ''
            if '100%' in last_line:
                print(f"[{label}] Progress bar at 100%, waiting for final flush...")
                continue
            # Print progress
            ts = time.strftime('%H:%M:%S')
            print(f"[{ts}] [{label}] {last_line[:80]}")
        except Exception as e:
            print(f"[{label}] Error reading stdout: {e}")
            continue

# Launch T2
print("=" * 60)
print("Stage 10: Sequential Full Training")
print("  T2 → T8 (warm-start from agent_15000.pt)")
print("=" * 60)

# Check if T2 is already running
import psutil
t2_running = False
for proc in psutil.process_iter(['pid', 'cmdline']):
    try:
        cmdline = ' '.join(proc.info['cmdline'] or [])
        if 'S10_T2.json' in cmdline:
            t2_running = True
            print(f"T2 already running (PID {proc.info['pid']})")
            break
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

if not t2_running:
    print("Launching T2...")
    subprocess.Popen([
        "uv", "run", "starter_kit_schedule/scripts/train_one.py",
        "--config", "starter_kit_schedule/configs/S10_T2.json"
    ], stdout=open("_s10_t2_stdout.txt", "w"), stderr=open("_s10_t2_stderr.txt", "w"))

# Wait for T2
wait_for_completion("_s10_t2_stdout.txt", "T2")

# Launch T8
print()
print("=" * 60)
print("Launching T8...")
print("=" * 60)
subprocess.Popen([
    "uv", "run", "starter_kit_schedule/scripts/train_one.py",
    "--config", "starter_kit_schedule/configs/S10_T8.json"
], stdout=open("_s10_t8_stdout.txt", "w"), stderr=open("_s10_t8_stderr.txt", "w"))

# Wait for T8
wait_for_completion("_s10_t8_stdout.txt", "T8")

print()
print("=" * 60)
print("Stage 10 COMPLETE: Both T2 and T8 finished.")
print("=" * 60)
