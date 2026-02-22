"""Minimal diagnostic: does the T4 checkpoint produce meaningful locomotion?"""
import sys
from pathlib import Path
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE / "starter_kit" / "navigation2"))
import vbot as _  # noqa: F401

from motrix_envs import registry as env_registry
from motrix_rl.skrl.torch.train.ppo import Trainer as PpoTrainer, _get_cfg
from motrix_rl.skrl.torch import wrap_env

CKPT = str(WORKSPACE / "runs/vbot_navigation_section011/26-02-22_14-22-21-888968_PPO/checkpoints/best_agent.pt")
NUM_ENVS = 16

trainer = PpoTrainer("vbot_navigation_section011", cfg_override={"play_num_envs": NUM_ENVS})
env = env_registry.make("vbot_navigation_section011", num_envs=NUM_ENVS)
skrl_env = wrap_env(env, enable_render=False)
rlcfg = trainer._rlcfg
models = trainer._make_model(skrl_env, rlcfg)
ppo_cfg = _get_cfg(rlcfg, skrl_env)
agent = trainer._make_agent(models, skrl_env, ppo_cfg)
agent.load(CKPT)
agent.set_running_mode("eval")

obs, _ = skrl_env.reset()
print(f"Obs shape: {obs.shape}, device: {obs.device}")
print(f"Obs[0] stats: min={obs[0].min():.3f}, max={obs[0].max():.3f}, mean={obs[0].mean():.3f}")

with torch.inference_mode():
    for step in range(500):
        outputs = agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
        obs, rew, term, trunc, infos = skrl_env.step(actions)

        if step % 25 == 0:
            metrics = infos.get("metrics", {})
            wp_raw = env._state.info.get("wp_idx", np.zeros(NUM_ENVS))
            wp_metric = metrics.get("wp_idx_mean", np.zeros(NUM_ENVS))
            max_y = metrics.get("max_y_progress", np.zeros(NUM_ENVS))
            robot_y = max_y  # use max_y_progress as proxy for forward motion

            print(f"\nStep {step:>4d}:")
            print(f"  actions[0]: min={actions[0].min():.3f} max={actions[0].max():.3f} mean={actions[0].mean():.3f}")
            print(f"  reward: mean={rew.mean():.4f} min={rew.min():.4f} max={rew.max():.4f}")
            print(f"  terminated: {term.sum().item()}/{NUM_ENVS}  truncated: {trunc.sum().item()}/{NUM_ENVS}")
            print(f"  wp_idx(info):   {wp_raw[:8]}")
            print(f"  wp_idx(metric): {wp_metric[:8]}")
            print(f"  max_y_progress: {max_y[:8]}")
            print(f"  robot_y(raw):   {np.round(robot_y[:8], 3)}")
            print(f"  obs[0,:6]: {obs[0,:6].cpu().numpy().round(3)}")

print("\n=== DONE ===")
