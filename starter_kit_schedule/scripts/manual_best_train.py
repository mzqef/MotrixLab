"""
Manual training with optimal configuration (post-AutoML).

Uses the known-good default RL config (lr=3e-4, epochs=8, entropy=0.005)
with heading fix and anti-laziness code for a full 100M step training run.

Expected duration: ~3.5 hours
Expected outcome: reward >> 6.75 (Round 1 best at 10M), high success rate

Usage:
    uv run python starter_kit_schedule/scripts/manual_best_train.py
    uv run python starter_kit_schedule/scripts/manual_best_train.py --max-env-steps 50000000
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(description="Manual training with best config")
    parser.add_argument("--max-env-steps", type=int, default=100_000_000, help="Max env steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--rollouts", type=int, default=24, help="Rollout steps")
    parser.add_argument("--epochs", type=int, default=8, help="Learning epochs")
    parser.add_argument("--entropy", type=float, default=0.005, help="Entropy loss scale")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Checkpoint interval")
    args = parser.parse_args()

    # Use default cfg.py reward scales (already well-tuned)
    # The env code has the heading fix and anti-laziness mechanisms

    # Setup path
    starter_kit_dir = os.path.join(PROJECT_ROOT, "starter_kit", "navigation1")
    if starter_kit_dir not in sys.path:
        sys.path.insert(0, starter_kit_dir)
    import vbot  # noqa: F401

    from motrix_rl.skrl.torch.train import ppo

    env_name = "vbot_navigation_section001"
    
    cfg_override = {
        "learning_rate": args.lr,
        "rollouts": args.rollouts,
        "learning_epochs": args.epochs,
        "entropy_loss_scale": args.entropy,
        "max_env_steps": args.max_env_steps,
        "check_point_interval": args.checkpoint_interval,
    }
    
    print(f"[ManualTrain] Starting {env_name} with {args.max_env_steps:,} steps")
    print(f"[ManualTrain] lr={args.lr}, rollouts={args.rollouts}, epochs={args.epochs}")
    print(f"[ManualTrain] entropy={args.entropy}")

    start = time.time()
    trainer = ppo.Trainer(env_name, cfg_override=cfg_override)
    trainer.train()
    elapsed = time.time() - start

    print(f"[ManualTrain] Completed in {elapsed:.0f}s ({elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
