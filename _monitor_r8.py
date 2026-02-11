"""Quick R8b training monitoring script - reads TensorBoard events."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter_kit_schedule", "scripts"))
from evaluate import read_tb_scalars, list_tb_tags

RUN_DIR = "d:/MotrixLab/runs/vbot_navigation_section001/26-02-11_11-57-08-242154_PPO"
R11_DIR = "d:/MotrixLab/runs/vbot_navigation_section001/26-02-11_03-12-45-451724_PPO"
R10_DIR = "d:/MotrixLab/runs/vbot_navigation_section001/26-02-11_02-22-52-156620_PPO"
R7_DIR = "d:/MotrixLab/runs/vbot_navigation_section001/26-02-10_12-52-53-648105_PPO"
R8B_DIR = "d:/MotrixLab/runs/vbot_navigation_section001/26-02-11_00-48-16-248222_PPO"

# Key metrics to monitor
METRICS = [
    "Reward / Instantaneous reward (mean)",
    "Reward / Total reward (mean)",
    "metrics / reached_fraction (mean)",
    "metrics / distance_to_target (mean)",
    "Reward Instant / departure_penalty (mean)",
    "Reward Instant / approach_reward (mean)",
    "Reward Instant / stop_bonus (mean)",
    "Reward Instant / forward_velocity (mean)",
    "Reward Instant / arrival_bonus (mean)",
    "Reward Instant / fine_position_tracking (mean)",
    "Reward Instant / fine_tracking_gated (mean)",
    "Episode / Total timesteps (mean)",
    "Policy / Standard deviation",
]

print(f"{'='*80}")
print("Stage 2 Training Monitor (warm-start from R16)")
print(f"Run: {RUN_DIR.split('/')[-1]}")
print(f"{'='*80}")

# Load R7 comparison data
r7_reached = read_tb_scalars(R7_DIR, "metrics / reached_fraction (mean)")
r7_milestones = {}
if r7_reached:
    for s, v in r7_reached:
        r7_milestones[s] = v

# List all available tags first
tags = list_tb_tags(RUN_DIR)
if not tags:
    print("No TensorBoard data yet. Training may still be initializing.")
    sys.exit(0)

# Read and display each metric
for tag in METRICS:
    data = read_tb_scalars(RUN_DIR, tag)
    if not data:
        continue
    steps, values = zip(*data)
    last_5 = list(zip(steps[-5:], values[-5:]))
    peak_val = max(values)
    peak_step = steps[values.index(peak_val)]
    print(f"\n--- {tag} ---")
    print(f"  Peak: {peak_val:.4f} (step {peak_step})")
    print(f"  Last 5:")
    for s, v in last_5:
        print(f"    step {s:>6d}: {v:.4f}")

# Summary line
reached = read_tb_scalars(RUN_DIR, "metrics / reached_fraction (mean)")
reward = read_tb_scalars(RUN_DIR, "Reward / Instantaneous reward (mean)") or read_tb_scalars(RUN_DIR, "Reward / Total reward (mean)")
if reached and reward:
    _, r_vals = zip(*reached)
    r_steps, _ = zip(*reached)
    _, rew_vals = zip(*reward)
    print(f"\n{'='*80}")
    print(f"SUMMARY: Reached%={r_vals[-1]*100:.2f}% (peak {max(r_vals)*100:.2f}%), Reward={rew_vals[-1]:.3f} (peak {max(rew_vals):.3f})")
    print(f"Total entries: {len(r_vals)}, Latest step: {reached[-1][0]}")
    
    # R7 comparison at same step
    cur_step = reached[-1][0]
    closest_r7_step = min(r7_milestones.keys(), key=lambda s: abs(s - cur_step)) if r7_milestones else None
    if closest_r7_step:
        r7_val = r7_milestones[closest_r7_step]
        print(f"R7 comparison: {r7_val*100:.2f}% at step {closest_r7_step} (Stage2 delta: {(r_vals[-1] - r7_val)*100:+.2f}%)")
    # R11 comparison at same step
    r11_reached = read_tb_scalars(R11_DIR, "metrics / reached_fraction (mean)")
    r11_dict = dict(r11_reached) if r11_reached else {}
    closest_r11 = min(r11_dict.keys(), key=lambda s: abs(s - cur_step)) if r11_dict else None
    if closest_r11:
        print(f"R11 comparison: {r11_dict[closest_r11]*100:.2f}% at step {closest_r11} (peak {max(r11_dict.values())*100:.2f}%)")
    # R10 comparison
    r10_reached = read_tb_scalars(R10_DIR, "metrics / reached_fraction (mean)")
    r10_dict = dict(r10_reached) if r10_reached else {}
    closest_r10 = min(r10_dict.keys(), key=lambda s: abs(s - cur_step)) if r10_dict else None
    if closest_r10:
        print(f"R10 comparison: {r10_dict[closest_r10]*100:.2f}% at step {closest_r10}")
    print(f"R7 peak: {max(r7_milestones.values())*100:.2f}% at step {max(r7_milestones, key=r7_milestones.get)}" if r7_milestones else "R7: no data")
    print(f"{'='*80}")
