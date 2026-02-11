"""Diagnose R8 reward components during decline."""
import sys, os
sys.path.insert(0, os.path.join('starter_kit_schedule', 'scripts'))
from evaluate import read_tb_scalars

RUN = 'd:/MotrixLab/runs/vbot_navigation_section001/26-02-11_00-11-14-119071_PPO'
tags = [
    'Reward Total/ departure_penalty (mean)',
    'Reward Total/ stop_bonus (mean)',
    'Reward Total/ arrival_bonus (mean)',
    'Reward Total/ approach_reward (mean)',
    'Reward Total/ forward_velocity (mean)',
    'Reward Total/ alive_bonus (mean)',
    'Reward Total/ near_target_speed (mean)',
    'Reward Total/ penalties (mean)',
    'Reward Total/ termination (mean)',
    'Reward Total/ heading_tracking (mean)',
    'Reward Total/ fine_position_tracking (mean)',
    'Reward Total/ position_tracking (mean)',
    'Reward Total/ distance_progress (mean)',
]

print(f"{'Component':<30} {'step 3500':>10} {'step 4000':>10} {'step 4500':>10} {'step 5000':>10} {'trend':>10}")
print("=" * 80)

for tag in tags:
    data = read_tb_scalars(RUN, tag)
    if not data:
        continue
    steps, vals = zip(*data)
    row_vals = {}
    for s, v in zip(steps, vals):
        if s in [3500, 4000, 4500, 5000]:
            row_vals[s] = v
    if row_vals:
        name = tag.split("/")[1].strip().replace(" (mean)", "")
        v3500 = row_vals.get(3500, 0)
        v4000 = row_vals.get(4000, 0)
        v4500 = row_vals.get(4500, 0)
        v5000 = row_vals.get(5000, 0)
        trend = v5000 - v3500
        arrow = "UP" if trend > 10 else ("DOWN" if trend < -10 else "~flat")
        print(f"{name:<30} {v3500:>10.1f} {v4000:>10.1f} {v4500:>10.1f} {v5000:>10.1f} {arrow:>10}")
