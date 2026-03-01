"""Resume S9 AutoML after crash."""
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

CHECKPOINT = r"runs\vbot_navigation_section011\26-02-28_13-15-37-944448_PPO\checkpoints\agent_15000.pt"

cmd = [
    sys.executable,
    "starter_kit_schedule/scripts/automl.py",
    "--resume",
    "--env", "vbot_navigation_section011",
    "--checkpoint", CHECKPOINT,
    "--freeze-preprocessor",
    "--env-overrides", env_overrides,
]

print("Resuming S9 AutoML...")
proc = subprocess.run(cmd, timeout=25200)
sys.exit(proc.returncode)
