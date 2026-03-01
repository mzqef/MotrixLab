import sys
import numpy as np
import torch
from pathlib import Path
import collections

# Register environment
sys.path.insert(0, str(Path(__file__).resolve().parent / "starter_kit" / "navigation1"))
import vbot

from motrix_envs import registry as env_registry
from motrix_rl import registry as rl_registry
from motrix_rl.skrl.torch.train.ppo import Trainer, _get_cfg
from motrix_rl.skrl.torch import wrap_env
from skrl.utils import set_seed
from skrl import config

config.torch.backend = "torch"

ENV_NAME = "vbot_navigation_section001"
NUM_ENVS = 2048
MAX_STEPS = 1000
CKPT = r"D:\MotrixLab\starter_kit_schedule\checkpoints\vbot_navigation_section001\best_agent.pt"

trainer = Trainer(ENV_NAME, sim_backend=None, enable_render=False, cfg_override={"play_num_envs": NUM_ENVS, "seed": 42})
env = env_registry.make(ENV_NAME, sim_backend=None, num_envs=NUM_ENVS)
set_seed(42)
skrl_env = wrap_env(env, False)
models = trainer._make_model(skrl_env, trainer._rlcfg)
agent = trainer._make_agent(models, skrl_env, _get_cfg(trainer._rlcfg, skrl_env))
agent.load(CKPT)

obs, _ = skrl_env.reset()
falls = 0
completed = np.zeros(NUM_ENVS, dtype=bool)

# Track reasons
fall_reasons = []

for step in range(MAX_STEPS):
    with torch.inference_mode():
        outputs = agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
    
    obs, reward, terminated, truncated, infos = skrl_env.step(actions)
    
    term_np = terminated.cpu().numpy().flatten().astype(bool)
    newly_done = term_np & ~completed
    
    if newly_done.any():
        # calculate base contact ourselves based on obs to be sure, 
        # actually env doesn't export base_contact in obs, it's just in data
        
        # side flip:
        pg = obs[:, 6:9].cpu().numpy()  # projected_gravity is typically at idx 6 to 9
        gxy = np.linalg.norm(pg[:, :2], axis=1)
        gz = pg[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip = tilt_angle > np.deg2rad(75)

        # joint vel extreme
        # joint vel starts around idx 21 (lin_vel 3, gyro 3, pg 3, joint_pos 12, joint_vel 12)
        joint_vel = obs[:, 21:33].cpu().numpy()
        vel_max = np.abs(joint_vel).max(axis=1)
        vel_extreme = vel_max > env._cfg.max_dof_vel
            
        robot_xy = env._state.info.get("robot_xy", np.zeros((NUM_ENVS,2)))
        robot_dist_from_origin = np.linalg.norm(robot_xy, axis=1)
            
        for i in range(NUM_ENVS):
            if newly_done[i]:
                falls += 1
                reason = ""
                if side_flip[i]:
                    reason = "SideFlip (>75deg)"
                elif vel_extreme[i]:
                    reason = "JointVel (Extreme)"
                else:
                    reason = "BaseContact (Fall)"
                
                # Check if off platform
                if robot_dist_from_origin[i] > 10.9:
                    reason += " [OFF_PLATFORM_EDGE]"
                
                fall_reasons.append(reason)
                
    completed = completed | term_np | truncated.cpu().numpy().flatten().astype(bool)

print("="*50)
print("Terminated reasons breakdown:")
for k, v in collections.Counter(fall_reasons).items():
    print(f"  {k}: {v}")
print("="*50)
