"""Quick training progress checker."""
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/vbot_navigation_section001/26-02-07_05-47-42-371699_PPO"

ea = EventAccumulator(run_dir)
ea.Reload()

tags = {
    "Reward / Instantaneous reward (mean)": "rwd",
    "metrics / distance_to_target (mean)": "dist",
    "metrics / reached_fraction (mean)": "reached",
    "Episode / Total timesteps (mean)": "ep_len",
    "Reward Total/ arrival_bonus (mean)": "arrival",
    "Reward / Instantaneous reward (max)": "rwd_max",
    "Reward Total/ termination (mean)": "term_total",
}

print(f"{'Metric':<12} {'Value':>10} {'Step':>8}")
print("-" * 35)
for tag, name in tags.items():
    try:
        events = ea.Scalars(tag)
        last = events[-1]
        print(f"{name:<12} {last.value:>10.4f} {last.step:>8}")
    except KeyError:
        pass

# Learning curve (reward)
rwd = ea.Scalars("Reward / Instantaneous reward (mean)")
n = len(rwd)
indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
print(f"\nReward curve ({n} points):")
for i in indices:
    print(f"  step {rwd[i].step:>6}: {rwd[i].value:.4f}")

# Checkpoints
ckpt_dir = Path(run_dir) / "checkpoints"
ckpts = sorted(ckpt_dir.glob("agent_*.pt"))
print(f"\nCheckpoints: {len(ckpts)}")
if ckpts:
    print(f"  Latest: {ckpts[-1].name}")
    print(f"  Best: {'best_agent.pt exists' if (ckpt_dir / 'best_agent.pt').exists() else 'No'}")
