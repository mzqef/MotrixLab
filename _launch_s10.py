import json
import subprocess
import os
import time

STEPS = 50_000_000 # 50M steps

CHECKPOINT = r"runs\vbot_navigation_section011\26-02-28_13-15-37-944448_PPO\checkpoints\agent_15000.pt"

# Read seeds_S10.json
with open('seeds_S10.json', 'r') as f:
    seeds = json.load(f)

t2_seed = seeds[0]
t8_seed = seeds[1]

# Base config structure
def create_config(seed_data, run_tag):
    rl_overrides = seed_data['hp_config'].copy()
    rl_overrides['max_env_steps'] = STEPS
    rl_overrides['check_point_interval'] = 500  # More frequent checkpoints

    return {
        "run_tag": run_tag,
        "env_name": "vbot_navigation_section011",
        "starter_kit_dir": "D:\\MotrixLab\\starter_kit\\navigation2",
        "reward_scales": seed_data['reward_config'],
        "rl_overrides": rl_overrides,
        "checkpoint": CHECKPOINT,
        "freeze_preprocessor": True,
        "env_overrides": {
            "hard_tilt_deg": 70.0,
            "soft_tilt_deg": 50.0,
            "enable_base_contact_term": True,
            "enable_stagnation_truncate": True,
            "grace_period_steps": 100,
            "reset_yaw_scale": 0.0
        }
    }

# Ensure configs directory exists
os.makedirs("starter_kit_schedule/configs", exist_ok=True)

# Write configs
with open("starter_kit_schedule/configs/S10_T2.json", "w") as f:
    json.dump(create_config(t2_seed, "S10_T2"), f, indent=2)

with open("starter_kit_schedule/configs/S10_T8.json", "w") as f:
    json.dump(create_config(t8_seed, "S10_T8"), f, indent=2)

print("Generated config S10_T2.json and S10_T8.json")

print("Launching T2...")
proc_t2 = subprocess.Popen([
    "uv", "run", "starter_kit_schedule/scripts/train_one.py",
    "--config", "starter_kit_schedule/configs/S10_T2.json"
], stdout=open("_s10_t2_stdout.txt", "w"), stderr=open("_s10_t2_stderr.txt", "w"))

import time

print("Launching T8...")
time.sleep(2)
proc_t8 = subprocess.Popen([
    "uv", "run", "starter_kit_schedule/scripts/train_one.py",
    "--config", "starter_kit_schedule/configs/S10_T8.json"
], stdout=open("_s10_t8_stdout.txt", "w"), stderr=open("_s10_t8_stderr.txt", "w"))

print(f"Started T2 (PID: {proc_t2.pid}) and T8 (PID: {proc_t8.pid}) in background.")
print("Monitoring tools can now be used.")
