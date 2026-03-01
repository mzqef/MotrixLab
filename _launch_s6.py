"""Launch S6 AutoML: X-axis walk + sit-down celebration, soft termination, 2π yaw, A_T4 warm-start, 6h budget."""
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
    "--budget-hours", "6",
    "--hp-trials", "20",
    "--checkpoint", "runs/vbot_navigation_section011/26-02-23_13-49-12-918060_PPO/checkpoints/agent_11500.pt",
    "--freeze-preprocessor",
    "--env-overrides", env_overrides,
]

print("Launching S6:", " ".join(cmd[:6]), "...")
print(f"env_overrides: {env_overrides}")
proc = subprocess.run(cmd)
sys.exit(proc.returncode)
