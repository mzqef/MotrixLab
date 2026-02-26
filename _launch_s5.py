"""Launch S5 AutoML: top 5 S4 seeds, relaxed termination, 2h budget, full 2π yaw."""
import subprocess
import sys
import json

env_overrides = json.dumps({
    "hard_tilt_deg": 85.0,
    "soft_tilt_deg": 0.0,
    "enable_base_contact_term": False,
    "enable_stagnation_truncate": True,
    "grace_period_steps": 500,
})

cmd = [
    sys.executable,
    "starter_kit_schedule/scripts/automl.py",
    "--mode", "stage",
    "--env", "vbot_navigation_section011",
    "--budget-hours", "2",
    "--hp-trials", "6",
    "--checkpoint", "runs/vbot_navigation_section011/26-02-23_13-49-12-918060_PPO/checkpoints/agent_11500.pt",
    "--freeze-preprocessor",
    "--seed-configs",
    "starter_kit_schedule/configs/seed_S4_T7.json",
    "starter_kit_schedule/configs/seed_S4_T16.json",
    "starter_kit_schedule/configs/seed_S4_T4.json",
    "starter_kit_schedule/configs/seed_S4_T15.json",
    "starter_kit_schedule/configs/seed_S4_T6.json",
    "--env-overrides", env_overrides,
]

print("Launching:", " ".join(cmd[:6]), "...")
print(f"env_overrides: {env_overrides}")
proc = subprocess.run(cmd)
sys.exit(proc.returncode)
