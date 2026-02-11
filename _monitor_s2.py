"""Compact TensorBoard monitor for Stage 2 training."""
import sys
import os

def monitor(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir)
    ea.Reload()
    
    # Use MEAN tags
    reached_tag = "metrics / reached_fraction (mean)"
    distance_tag = "metrics / distance_to_target (mean)"
    eplen_tag = "Episode / Total timesteps (mean)"
    
    tags = ea.Tags().get("scalars", [])
    if reached_tag not in tags:
        print(f"Tag '{reached_tag}' not found. Available:")
        for t in tags[:30]:
            print(f"  {t}")
        return
    
    reached_events = ea.Scalars(reached_tag)
    dist_events = {e.step: e.value for e in ea.Scalars(distance_tag)} if distance_tag in tags else {}
    ep_events = {e.step: e.value for e in ea.Scalars(eplen_tag)} if eplen_tag in tags else {}
    
    # Sample at intervals
    n = len(reached_events)
    if n <= 40:
        indices = range(n)
    else:
        step_size = max(1, n // 35)
        indices = list(range(0, n, step_size))
        if n - 1 not in indices:
            indices.append(n - 1)
    
    print(f"\n=== Stage 2 Training Monitor ({os.path.basename(run_dir)}) ===")
    print(f"  Spawn: 5-8m | Warm-start: R16 agent_9600.pt | LR: 2.17e-4 | 30M steps")
    print(f"\n{'Step':>8} | {'Reached%':>10} | {'Distance':>10} | {'EpLen':>8}")
    print("-" * 50)
    
    best_reached = 0
    best_step = 0
    
    for i in indices:
        e = reached_events[i]
        step = e.step
        reached = e.value
        
        if reached > best_reached:
            best_reached = reached
            best_step = step
        
        dist = dist_events.get(step, float('nan'))
        eplen = ep_events.get(step, float('nan'))
        
        marker = " ***" if reached >= best_reached and reached > 0.05 else ""
        print(f"{step:>8} | {reached*100:>9.2f}% | {dist:>10.2f} | {eplen:>8.0f}{marker}")
    
    print(f"\n*** BEST: {best_reached*100:.2f}% at step {best_step} ***")
    print(f"*** Latest: step {reached_events[-1].step}, {reached_events[-1].value*100:.2f}% ***")

if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\MotrixLab\runs\vbot_navigation_section001\26-02-11_06-43-09-491787_PPO"
    monitor(run_dir)
