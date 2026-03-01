"""Deep comparison of S4 AutoML trials — focus on celebration completion and reward quality."""
import os
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_base = "runs/vbot_navigation_section011/"
all_runs = sorted(glob.glob(run_base + "26-02-26_*_PPO"))
our_runs = [r for r in all_runs if os.path.basename(r) >= "26-02-26_03-34"]

print(f"=== S4 AutoML (automl_20260226_033450) — {len(our_runs)} trials ===\n")

# Collect detailed metrics
results = []
for i, rdir in enumerate(our_runs):
    ea = EventAccumulator(rdir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    
    info = {"trial": i, "run_dir": os.path.basename(rdir)}
    
    # wp_idx
    for tag in tags:
        if "wp_idx" in tag and "mean" in tag:
            events = ea.Scalars(tag)
            vals = [e.value for e in events]
            info["peak_wp"] = max(vals)
            info["final_wp"] = vals[-1]
            info["wp_steps"] = events[-1].step
            # Find when wp first reached 7.0
            for e in events:
                if e.value >= 6.9:
                    info["wp7_step"] = e.step
                    break
            break
    
    # celeb_state
    for tag in tags:
        if "celeb_state" in tag:
            events = ea.Scalars(tag)
            vals = [e.value for e in events]
            info["peak_celeb"] = max(vals)
            info["final_celeb"] = vals[-1]
            # When celeb first reached 3.0
            for e in events:
                if e.value >= 2.9:
                    info["celeb_done_step"] = e.step
                    break
            break
    
    # reward
    for tag in tags:
        if tag == "Reward / Total reward (mean)":
            events = ea.Scalars(tag)
            vals = [e.value for e in events]
            info["peak_reward"] = max(vals)
            info["final_reward"] = vals[-1]
            # Last 20% average
            n20 = max(1, len(vals) // 5)
            info["avg_reward_last20"] = np.mean(vals[-n20:])
            break
    
    # reached fraction
    for tag in tags:
        if "reached" in tag and "fraction" in tag:
            events = ea.Scalars(tag)
            vals = [e.value for e in events]
            info["peak_reached"] = max(vals)
            info["final_reached"] = vals[-1]
            n20 = max(1, len(vals) // 5)
            info["avg_reached_last20"] = np.mean(vals[-n20:])
            break
    
    # episode length
    for tag in tags:
        if "Episode" in tag and "length" in tag.lower():
            events = ea.Scalars(tag)
            vals = [e.value for e in events]
            info["peak_ep_len"] = max(vals)
            info["final_ep_len"] = vals[-1]
            break
    
    # termination rate
    for tag in tags:
        if "termination" in tag.lower() and "rate" in tag.lower():
            events = ea.Scalars(tag)
            vals = [e.value for e in events]
            info["avg_term_rate"] = np.mean(vals[-max(1, len(vals)//5):])
            break

    # turn_count
    for tag in tags:
        if "turn_count" in tag:
            events = ea.Scalars(tag)
            vals = [e.value for e in events]
            info["peak_turns"] = max(vals)
            info["final_turns"] = vals[-1]
            break

    results.append(info)

# Print summary
print(f"{'T':>3} {'pk_wp':>6} {'celeb':>6} {'f_cel':>6} {'cel@step':>9} {'pk_rew':>8} {'f_rew':>8} {'reached%':>9} {'ep_len':>7} {'turns':>6}")
print("-" * 85)
for r in results:
    cel_step = r.get("celeb_done_step", "-")
    cel_step_s = f"{cel_step}" if isinstance(cel_step, int) else "-"
    turns = r.get("peak_turns", "-")
    turns_s = f"{turns:.1f}" if isinstance(turns, (int, float)) else "-"
    print(f"T{r['trial']:<2} {r.get('peak_wp', 0):>6.1f} {r.get('peak_celeb', 0):>6.1f} {r.get('final_celeb', 0):>6.1f} {cel_step_s:>9} {r.get('peak_reward', 0):>8.0f} {r.get('final_reward', 0):>8.0f} {r.get('peak_reached', 0)*100:>8.1f}% {r.get('peak_ep_len', 0):>7.0f} {turns_s:>6}")

# Rank by celebration completion + reward
print("\n=== RANKING (celeb_done first, then reward) ===")
celeb_done = [r for r in results if r.get("peak_celeb", 0) >= 2.9]
celeb_settling = [r for r in results if r.get("peak_celeb", 0) < 2.9]

print(f"\nCELEB_DONE trials ({len(celeb_done)}):")
celeb_done.sort(key=lambda r: r.get("peak_reward", 0), reverse=True)
for r in celeb_done:
    cel_step = r.get("celeb_done_step", "?")
    print(f"  T{r['trial']}: peak_reward={r.get('peak_reward',0):.0f}, final_reward={r.get('final_reward',0):.0f}, celeb@step={cel_step}, reached={r.get('peak_reached',0)*100:.1f}%, turns={r.get('peak_turns','-')}")

print(f"\nCELEB_SETTLING trials ({len(celeb_settling)}):")
celeb_settling.sort(key=lambda r: r.get("peak_reward", 0), reverse=True)
for r in celeb_settling[:5]:
    print(f"  T{r['trial']}: peak_reward={r.get('peak_reward',0):.0f}, reached={r.get('peak_reached',0)*100:.1f}%")
