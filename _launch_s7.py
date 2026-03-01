"""Launch S7 AutoML: Cold-Start, hard termination, NO random yaw (reproducing A_T4 lineage Step 1)."""
import subprocess
import sys
import json

env_overrides = json.dumps({
    "hard_tilt_deg": 70.0,
    "soft_tilt_deg": 50.0,
    "enable_base_contact_term": True,
    "enable_stagnation_truncate": True,
    "grace_period_steps": 100,
    "reset_yaw_scale": 0.0
})

cmd = [
    sys.executable,
    "starter_kit_schedule/scripts/automl.py",
    "--mode", "stage",
    "--env", "vbot_navigation_section011",
    "--budget-hours", "5",
    "--hp-trials", "12",
    "--seed-configs", "starter_kit_schedule/configs/seed_T12_warmstart.json",
    "--env-overrides", env_overrides
]

print("Launching S7:", " ".join(cmd[:6]), "...")
print(f"env_overrides: {env_overrides}")
proc = subprocess.run(cmd)
sys.exit(proc.returncode)
