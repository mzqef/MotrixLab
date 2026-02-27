import sys
from pathlib import Path
import torch
import numpy as np
import time

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE / "starter_kit" / "navigation2"))
import vbot as navigation2_vbot

from motrix_envs import registry as env_registry
from motrix_rl.registry import default_rl_cfg
from motrix_rl.skrl.torch.train.ppo import Trainer
from motrix_rl.skrl.torch import wrap_env

# Import the RL config to register it
import motrix_rl.cfgs
import vbot.cfg
import vbot.vbot_section011_np
import vbot.rl_cfgs

def evaluate_checkpoint(ckpt_path, num_envs=100, max_steps=2000):
    print("Initializing environment...")
    env_name = "vbot_navigation_section011"
    env = env_registry.make(env_name, num_envs=num_envs)

    print("Building agent...")
    # Use Trainer to build the agent
    trainer = Trainer(env_name, "np", enable_render=False)
    rlcfg = trainer._rlcfg
    skrl_env = wrap_env(env, False)
    models = trainer._make_model(skrl_env, rlcfg)

    from motrix_rl.skrl.torch.train.ppo import _get_cfg
    ppo_cfg = _get_cfg(rlcfg, skrl_env)
    agent = trainer._make_agent(models, skrl_env, ppo_cfg)

    print("Loading checkpoint...")
    agent.load(ckpt_path)
    # agent.set_running_mode("eval") # Try without this

    print("Resetting environment...")
    obs, info = skrl_env.reset()

    # Track the final metrics for each environment
    final_wp_idx = np.zeros(num_envs, dtype=np.float32)
    final_celeb = np.zeros(num_envs, dtype=np.float32)
    has_finished = np.zeros(num_envs, dtype=bool)
    
    term_count = 0
    trunc_count = 0

    print("Starting evaluation loop...")
    start_time = time.time()
    for step in range(max_steps):
        if step % 100 == 0:
            print(f"Step {step}/{max_steps}...")
            
        with torch.no_grad():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])

        obs, reward, terminated, truncated, info = skrl_env.step(actions)

        dones = terminated | truncated
        if torch.any(dones):
            done_indices = torch.where(dones)[0]
            for idx in done_indices:
                idx_val = idx.item()
                if not has_finished[idx_val]:
                    final_wp_idx[idx_val] = info["wp_idx"][idx_val].item()
                    final_celeb[idx_val] = info["celeb_state"][idx_val].item()
                    has_finished[idx_val] = True
                    
                if terminated[idx_val]: term_count += 1
                if truncated[idx_val]: trunc_count += 1

        if np.all(has_finished):
            print(f"All environments finished at step {step}")
            break

    print(f"Evaluation finished in {time.time() - start_time:.2f} seconds.")

    # For any envs that didn't finish, use their current state
    unfinished = ~has_finished
    if np.any(unfinished):
        final_wp_idx[unfinished] = info["wp_idx"][unfinished]
        final_celeb[unfinished] = info["celeb_state"][unfinished]

    print(f"Evaluation Results:")
    print(f"  wp_idx_mean: {np.mean(final_wp_idx):.3f}")
    print(f"  celeb_state_mean: {np.mean(final_celeb):.3f}")
    print(f"  wp_idx_max: {np.max(final_wp_idx)}")
    print(f"  celeb_state_max: {np.max(final_celeb)}")
    print(f"  Terminated (falls): {term_count}")
    print(f"  Truncated (timeouts): {trunc_count}")

if __name__ == "__main__":
    ckpt = "runs/vbot_navigation_section011/26-02-27_00-50-58-430826_PPO/checkpoints/best_agent.pt"
    evaluate_checkpoint(ckpt)
