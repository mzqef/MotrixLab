"""Read TensorBoard data directly for all S4 AutoML trials."""
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_base = "runs/vbot_navigation_section011/"
# Find runs from this automl (started 2026-02-26 03:34+)
all_runs = sorted(glob.glob(run_base + "26-02-26_*_PPO"))
our_runs = [r for r in all_runs if os.path.basename(r) >= "26-02-26_03-34"]

print(f"Found {len(our_runs)} trial runs\n")
print(f"{'T':>3} {'peak_wp':>8} {'final_wp':>9} {'celeb':>6} {'f_celeb':>8} {'reward':>8} {'ep_len':>7} {'steps':>8}")
print("-" * 72)

for i, rdir in enumerate(our_runs):
    ea = EventAccumulator(rdir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    # wp_idx
    peak_wp = final_wp = "-"
    for tag in tags:
        if "wp_idx" in tag and "mean" in tag:
            events = ea.Scalars(tag)
            if events:
                vals = [e.value for e in events]
                peak_wp = f"{max(vals):.1f}"
                final_wp = f"{vals[-1]:.2f}"
            break

    # celeb_state
    celeb = f_celeb = "-"
    for tag in tags:
        if "celeb_state" in tag:
            events = ea.Scalars(tag)
            if events:
                vals = [e.value for e in events]
                celeb = f"{max(vals):.1f}"
                f_celeb = f"{vals[-1]:.1f}"
            break

    # reward
    reward = "-"
    for tag in tags:
        if tag == "Reward / Total reward (mean)":
            events = ea.Scalars(tag)
            if events:
                vals = [e.value for e in events]
                reward = f"{max(vals):.2f}"
            break

    # episode length
    ep_len = "-"
    for tag in tags:
        if "Episode" in tag and "length" in tag.lower():
            events = ea.Scalars(tag)
            if events:
                vals = [e.value for e in events]
                ep_len = f"{max(vals):.0f}"
            break

    # total steps
    steps = "-"
    for tag in tags:
        if "wp_idx" in tag:
            events = ea.Scalars(tag)
            if events:
                steps = f"{events[-1].step}"
            break

    name = os.path.basename(rdir)[:30]
    print(f"T{i:<2} {peak_wp:>8} {final_wp:>9} {celeb:>6} {f_celeb:>8} {reward:>8} {ep_len:>7} {steps:>8}  {name}")
