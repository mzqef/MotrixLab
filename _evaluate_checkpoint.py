"""Headless evaluation of a checkpoint — compute per-episode success rate and competition metrics."""
import sys, os, argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "starter_kit", "navigation1"))
import vbot  # noqa: F401

from motrix_envs import registry

def evaluate_checkpoint(checkpoint_path, env_name="vbot_navigation_section001",
                       num_envs=2048, num_episodes=4096, seed=42):
    """Run policy for num_episodes and compute success metrics."""
    from motrix_rl.skrl.torch.train import ppo
    import torch
    
    np.random.seed(seed)
    
    # Create environment
    env = registry.make(env_name, num_envs=num_envs)
    
    # Load from the RL training config
    from motrix_rl.registry import default_rl_cfg
    rl_cfg = default_rl_cfg(env_name, "ppo", "torch")
    
    # Build agent
    trainer = ppo.Trainer(env_name, cfg_override={"play_num_envs": num_envs, "seed": seed})
    
    # We need to run the trainer's play method in headless mode
    # Actually, let's use a simpler approach — run episodes directly
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Env: {env_name}, Num envs: {num_envs}, Target episodes: {num_episodes}")
    print(f"Running headless evaluation...")
    
    # Use the trainer's internal play mechanism
    trainer.play(checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint headlessly")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--env", default="vbot_navigation_section001")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--episodes", type=int, default=4096)
    args = parser.parse_args()
    
    evaluate_checkpoint(args.checkpoint, args.env, args.num_envs, args.episodes)


if __name__ == "__main__":
    main()
