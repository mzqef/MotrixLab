import sys
import numpy as np
import torch
from pathlib import Path

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
timeouts = 0
reached_count = 0
completed = np.zeros(NUM_ENVS, dtype=bool)

# track failure info
fail_dists = []
fall_steps = []
init_dists = np.full(NUM_ENVS, -1.0)
init_dist_recorded = np.zeros(NUM_ENVS, dtype=bool)

for step in range(MAX_STEPS):
    with torch.inference_mode():
        outputs = agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
    
    obs, reward, terminated, truncated, infos = skrl_env.step(actions)
    
    raw_info = env._state.info
    if "metrics" in raw_info:
        metrics = raw_info["metrics"]
        dist = metrics.get("distance_to_target", np.full(NUM_ENVS, 999.0))
        reached = metrics.get("reached_fraction", np.zeros(NUM_ENVS)) > 0.5
    else:
        dist = np.full(NUM_ENVS, 999.0)
        reached = np.zeros(NUM_ENVS, dtype=bool)

    # Note initial distance
    for i in range(NUM_ENVS):
        if not init_dist_recorded[i]:
            init_dists[i] = dist[i]
            init_dist_recorded[i] = True
        
    term_np = terminated.cpu().numpy().flatten().astype(bool)
    trunc_np = truncated.cpu().numpy().flatten().astype(bool)
    
    newly_done = (term_np | trunc_np) & ~completed
    
    for i in range(NUM_ENVS):
        if newly_done[i]:
            if reached[i]:
                reached_count += 1
            elif term_np[i]:
                falls += 1
                fall_steps.append(step)
                fail_dists.append(dist[i])
            elif trunc_np[i]:
                timeouts += 1
                fail_dists.append(dist[i])
            completed[i] = True

uncompleted = (~completed).sum()
# the ones that haven't naturally completed by MAX_STEPS are timeouts
if uncompleted > 0:
    for i in range(NUM_ENVS):
        if not completed[i]:
            if reached[i]:
                reached_count += 1
            else:
                timeouts += 1
                fail_dists.append(dist[i])

print("="*50)
print(f"Total Envs: {NUM_ENVS}")
print(f"Reached Target: {reached_count} ({(reached_count/NUM_ENVS)*100:.2f}%)")
print(f"Terminated (Falls/Crashes/Out of bounds): {falls} ({(falls/NUM_ENVS)*100:.2f}%)")
print(f"Truncated (Timeouts/Stuck): {timeouts} ({(timeouts/NUM_ENVS)*100:.2f}%)")
if falls > 0:
    print(f"Average fall step: {np.mean(fall_steps):.1f}")
if len(fail_dists) > 0:
    print(f"Average distance to target on failure: {np.mean(fail_dists):.3f}m")
print("="*50)
