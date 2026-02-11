"""Quick TensorBoard metrics reader for monitoring training progress."""
import sys
import os
import glob

def read_tb_metrics(run_dir, max_events=None):
    """Read key metrics from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("ERROR: tensorboard not installed")
        return
    
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {run_dir}")
        return
    
    ea = EventAccumulator(run_dir)
    ea.Reload()
    
    tags = ea.Tags().get("scalars", [])
    
    # Key metrics to monitor
    key_tags = {
        "reached": None,
        "distance": None,
        "ep_len": None,
        "reward_total": None,
        "approach": None,
        "arrival": None,
        "stop": None,
        "fine_track": None,
        "alive": None,
        "departure": None,
    }
    
    for tag in tags:
        tl = tag.lower()
        if "reached_fraction" in tl or "reached" in tl:
            key_tags["reached"] = tag
        elif "closest_distance" in tl or ("distance" in tl and "progress" not in tl):
            key_tags["distance"] = tag
        elif "total timesteps" in tl or "episode_length" in tl:
            key_tags["ep_len"] = tag
        elif "reward / total" in tl or ("reward" in tl and "total" in tl and "instant" not in tl and "/" not in tl):
            key_tags["reward_total"] = tag
        elif "approach" in tl and "instant" in tl:
            key_tags["approach"] = tag
        elif "arrival" in tl and "instant" in tl:
            key_tags["arrival"] = tag
        elif "stop_bonus" in tl and "instant" in tl:
            key_tags["stop"] = tag
        elif "fine" in tl and "tracking" in tl and "instant" in tl:
            key_tags["fine_track"] = tag
        elif "alive" in tl and "instant" in tl:
            key_tags["alive"] = tag
        elif "departure" in tl and "instant" in tl:
            key_tags["departure"] = tag
    
    # Print available tags for debugging
    print(f"\n=== TensorBoard Metrics for {os.path.basename(run_dir)} ===")
    print(f"Total scalar tags: {len(tags)}")
    
    # Find reached_fraction tag more flexibly
    reached_tags = [t for t in tags if "reached" in t.lower()]
    distance_tags = [t for t in tags if "distance" in t.lower() and "progress" not in t.lower()]
    ep_tags = [t for t in tags if "timestep" in t.lower() or "episode" in t.lower()]
    reward_tags = [t for t in tags if "reward" in t.lower() and "total" in t.lower()]
    
    if reached_tags:
        key_tags["reached"] = reached_tags[0]
    if distance_tags:
        key_tags["distance"] = distance_tags[0]
    if ep_tags:
        key_tags["ep_len"] = ep_tags[0]
    if reward_tags:
        key_tags["reward_total"] = reward_tags[0]
    
    print(f"\nMatched tags:")
    for name, tag in key_tags.items():
        if tag:
            print(f"  {name}: {tag}")
    
    # Print metrics at regular intervals
    if key_tags["reached"]:
        events = ea.Scalars(key_tags["reached"])
        print(f"\n--- Reached Fraction ({len(events)} data points) ---")
        print(f"{'Step':>8} | {'Reached%':>10} | {'Distance':>10} | {'EpLen':>8}")
        print("-" * 50)
        
        # Sample at regular intervals
        n = len(events)
        if n <= 30:
            indices = range(n)
        else:
            step_size = max(1, n // 25)
            indices = list(range(0, n, step_size))
            if n - 1 not in indices:
                indices.append(n - 1)
        
        best_reached = 0
        best_step = 0
        
        for i in indices:
            e = events[i]
            step = e.step
            reached = e.value
            
            if reached > best_reached:
                best_reached = reached
                best_step = step
            
            # Get corresponding distance and ep_len
            dist_str = ""
            eplen_str = ""
            
            if key_tags["distance"]:
                dist_events = ea.Scalars(key_tags["distance"])
                # Find closest step
                for de in dist_events:
                    if de.step == step:
                        dist_str = f"{de.value:.2f}"
                        break
            
            if key_tags["ep_len"]:
                ep_events = ea.Scalars(key_tags["ep_len"])
                for ee in ep_events:
                    if ee.step == step:
                        eplen_str = f"{ee.value:.0f}"
                        break
            
            print(f"{step:>8} | {reached*100:>9.2f}% | {dist_str:>10} | {eplen_str:>8}")
        
        print(f"\n*** BEST: {best_reached*100:.2f}% at step {best_step} ***")
    else:
        print("\nWARNING: No 'reached_fraction' tag found!")
        print("Available tags sample:")
        for t in tags[:20]:
            print(f"  {t}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python _monitor_stage2.py <run_dir>")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    read_tb_metrics(run_dir)
