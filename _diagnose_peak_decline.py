"""Diagnose peak-then-decline: compare reward components at peak vs decline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter_kit_schedule", "scripts"))
from evaluate import read_tb_scalars

RUN = "d:/MotrixLab/runs/vbot_navigation_section001/26-02-11_00-48-16-248222_PPO"
tags = [
    "metrics / reached_fraction (mean)",
    "metrics / distance_to_target (mean)",
    "Reward / Instantaneous reward (mean)",
    "Reward Total/ fine_position_tracking (mean)",
    "Reward Total/ position_tracking (mean)",
    "Reward Total/ heading_tracking (mean)",
    "Reward Total/ forward_velocity (mean)",
    "Reward Total/ approach_reward (mean)",
    "Reward Total/ stop_bonus (mean)",
    "Reward Total/ arrival_bonus (mean)",
    "Reward Total/ alive_bonus (mean)",
    "Reward Total/ distance_progress (mean)",
    "Reward Total/ termination (mean)",
    "Reward Total/ departure_penalty (mean)",
    "Reward Total/ near_target_speed (mean)",
    "Reward Total/ penalties (mean)",
    "Episode / Total timesteps (mean)",
]

# Peak zone: steps 6000-6500, Decline zone: steps 8000-8500
print(f"{'Component':<52} {'Peak(6k-6.5k)':>14} {'Curr(8k-8.5k)':>14} {'Delta':>10}")
print("-" * 92)
lines = []
for tag in tags:
    data = read_tb_scalars(RUN, tag)
    if not data:
        continue
    d = dict(data)
    peak_vals = [d[s] for s in range(6000, 6600, 100) if s in d]
    curr_vals = [d[s] for s in range(8000, 8600, 100) if s in d]
    if peak_vals and curr_vals:
        p = sum(peak_vals) / len(peak_vals)
        c = sum(curr_vals) / len(curr_vals)
        line = f"{tag:<52} {p:>14.4f} {c:>14.4f} {c-p:>+10.4f}"
        lines.append(line)
        print(line)

with open("_peak_decline_report.txt", "w") as f:
    f.write("\n".join(lines))
