"""Find the best checkpoint in a run by reached_fraction metric."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter_kit_schedule", "scripts"))
from evaluate import read_tb_scalars

def find_best_checkpoint(run_dir, metric="metrics / reached_fraction (mean)", top_n=10):
    """Find checkpoints corresponding to peak metric values."""
    data = read_tb_scalars(run_dir, metric)
    if not data:
        print(f"No data for metric: {metric}")
        return []
    
    # Sort by value descending
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    
    # Map steps to checkpoint files
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    results = []
    for step, value in sorted_data[:top_n]:
        ckpt_path = os.path.join(ckpt_dir, f"agent_{step}.pt")
        exists = os.path.exists(ckpt_path)
        results.append((step, value, ckpt_path, exists))
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"Best Checkpoints: {os.path.basename(args.run_dir)}")
    print(f"{'='*80}")
    
    results = find_best_checkpoint(args.run_dir, top_n=args.top)
    
    # Also get reward data for context
    reward_data = read_tb_scalars(args.run_dir, "Reward / Instantaneous reward (mean)")
    reward_dict = dict(reward_data) if reward_data else {}
    
    # Episode length
    ep_data = read_tb_scalars(args.run_dir, "Episode / Total timesteps (mean)")
    ep_dict = dict(ep_data) if ep_data else {}
    
    # Distance
    dist_data = read_tb_scalars(args.run_dir, "metrics / distance_to_target (mean)")
    dist_dict = dict(dist_data) if dist_data else {}
    
    print(f"\n{'Rank':<6} {'Step':>6} {'Reached%':>10} {'Reward':>8} {'Dist':>6} {'EpLen':>7} {'Checkpoint':>12}")
    print("-" * 65)
    for i, (step, value, path, exists) in enumerate(results, 1):
        reward = reward_dict.get(step, 0)
        ep_len = ep_dict.get(step, 0)
        dist = dist_dict.get(step, 0)
        status = "EXISTS" if exists else "MISSING"
        print(f"{i:<6} {step:>6} {value*100:>9.2f}% {reward:>8.3f} {dist:>6.2f} {ep_len:>7.0f} {status:>12}")
    
    if results and results[0][3]:
        print(f"\nBest checkpoint: {results[0][2]}")
        print(f"  Step {results[0][0]}, Reached% = {results[0][1]*100:.2f}%")

if __name__ == "__main__":
    main()
