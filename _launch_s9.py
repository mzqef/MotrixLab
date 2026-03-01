"""Launch S9 AutoML: Warm-start from R1b peak (B1_T4, wp=3.838 @ iter 15000).

Analogous to historical S2B: warm-start AutoML from the best full-train peak checkpoint.
  - Historical S2B: S2A peak (wp=2.232 @ iter 24500) → 15 trials × 10M → A_T4 champion (wp=3.411)
  - S9: R1b peak (wp=3.838 @ iter 15000) → 15 trials × 10M → find refined reward weights

Config:
  - Checkpoint: R1b agent_15000.pt (peak wp_mean=3.838, wp_max=7, smiley=85.3%)
  - Seed: B1_T4 reward scales (the config that produced R1/R1b)
  - Env: strict termination (hard=70°, soft=50°, base_contact ON, stagnation ON)
  - NO random yaw (reset_yaw_scale=0.0)
  - Budget: 6 hours, 15 trials × 10M steps
  - Freeze preprocessor: Yes (warm-start best practice)
"""
import subprocess
import sys
import json

CHECKPOINT = r"runs\vbot_navigation_section011\26-02-28_13-15-37-944448_PPO\checkpoints\agent_15000.pt"

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
    "--budget-hours", "6",
    "--hp-trials", "15",
    "--seed-configs", "starter_kit_schedule/configs/seed_S9_B1T4_warmstart.json",
    "--checkpoint", CHECKPOINT,
    "--freeze-preprocessor",
    "--env-overrides", env_overrides,
]

print("=" * 70)
print("Stage 9: Warm-Start AutoML from R1b Peak")
print("=" * 70)
print(f"Checkpoint: {CHECKPOINT}")
print(f"Seed: seed_S9_B1T4_warmstart.json")
print(f"Budget: 6 hours, 15 trials × 10M steps")
print(f"Env: strict term, NO yaw")
print(f"env_overrides: {env_overrides}")
print("=" * 70)

proc = subprocess.run(cmd, timeout=25200)  # 7h timeout (6h budget + 1h buffer)
sys.exit(proc.returncode)
