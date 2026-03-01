import torch
import numpy as np
import torch.nn as nn
import motrix_envs.registry as reg
from starter_kit.navigation2 import vbot

class Policy(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, num_actions)
        )
    def forward(self, obs):
        return self.net(obs)

env = reg.make("vbot_navigation_section011", num_envs=10)
device = "cuda" if torch.cuda.is_available() else "cpu"

policy = Policy(69, 12).to(device)
state_dict = torch.load("runs/vbot_navigation_section011/26-02-27_15-33-05-350414_PPO/checkpoints/best_agent.pt", map_location=device)
new_state_dict = {}
for k, v in state_dict["policy"].items():
    if k.startswith("_orig_mod.policy_net."):
        new_k = k.replace("_orig_mod.policy_net.", "net.")
        new_state_dict[new_k] = v
    elif k.startswith("_orig_mod.mean_layer."):
        new_k = k.replace("_orig_mod.mean_layer.", "net.6.")
        new_state_dict[new_k] = v
policy.load_state_dict(new_state_dict)
policy.eval()

state = env.init_state()
obs = state.obs
max_x = np.zeros(10)

for _ in range(500):
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        actions = policy(obs_tensor).cpu().numpy()
    
    state = env.step(actions)
    obs = state.obs
    
    root_pos, _, _ = env._extract_root_state(state.data)
    x_pos = root_pos[:, 0]
    max_x = np.maximum(max_x, x_pos)
    
    if np.all(state.terminated | state.truncated):
        break

print(f"Max X position reached: {max_x.max():.2f} meters")
print(f"Mean Max X position: {max_x.mean():.2f} meters")
