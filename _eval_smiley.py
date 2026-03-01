"""Evaluate smiley_reached fraction for all Stage 7 trials.

Loads best_agent.pt directly into a simple PyTorch MLP and runs
the raw NpEnv to count smiley collection stats per trial.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "starter_kit" / "navigation2"))
import vbot as _  # noqa: F401

import numpy as np
import torch
import torch.nn as nn

from motrix_envs import registry

NUM_ENVS = 512
MAX_STEPS = 2000
ENV_NAME = "vbot_navigation_section011"

ENV_OVERRIDES = {
    "hard_tilt_deg": 70.0,
    "soft_tilt_deg": 50.0,
    "enable_base_contact_term": True,
    "enable_stagnation_truncate": True,
    "grace_period_steps": 100,
    "reset_yaw_scale": 0.0,
}

TRIALS = [
    ("B1_T0", r"runs\vbot_navigation_section011\26-02-27_18-49-01-661380_PPO"),
    ("B1_T1", r"runs\vbot_navigation_section011\26-02-27_19-10-24-422607_PPO"),
    ("B1_T2", r"runs\vbot_navigation_section011\26-02-27_19-31-25-199492_PPO"),
    ("B1_T3", r"runs\vbot_navigation_section011\26-02-27_19-54-28-000867_PPO"),
    ("B1_T4", r"runs\vbot_navigation_section011\26-02-27_20-18-54-748261_PPO"),
    ("B1_T5", r"runs\vbot_navigation_section011\26-02-27_20-42-43-482497_PPO"),
    ("B1_T6", r"runs\vbot_navigation_section011\26-02-27_20-59-36-156821_PPO"),
    ("B1_T7", r"runs\vbot_navigation_section011\26-02-27_21-19-33-211179_PPO"),
    ("B2_T0", r"runs\vbot_navigation_section011\26-02-27_21-57-59-236564_PPO"),
    ("B2_T1", r"runs\vbot_navigation_section011\26-02-27_22-29-45-350073_PPO"),
    ("B2_T2", r"runs\vbot_navigation_section011\26-02-27_23-10-31-258152_PPO"),
    ("B2_T3", r"runs\vbot_navigation_section011\26-02-28_00-00-53-773429_PPO"),
    ("B2_T4", r"runs\vbot_navigation_section011\26-02-28_00-43-14-938756_PPO"),
    ("B2_T5", r"runs\vbot_navigation_section011\26-02-28_01-02-24-207770_PPO"),
    ("B2_T6", r"runs\vbot_navigation_section011\26-02-28_01-17-00-006586_PPO"),
    ("B2_T7", r"runs\vbot_navigation_section011\26-02-28_01-34-02-485212_PPO"),
]


class PolicyNet(nn.Module):
    """Simple MLP matching the SKRL PPO policy network."""
    def __init__(self, obs_dim, act_dim, hidden_sizes=(512, 256, 128)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(prev, act_dim)

    def forward(self, x):
        return self.mean(self.net(x))


def load_policy(ckpt_path: str, obs_dim: int, act_dim: int) -> PolicyNet:
    """Load SKRL checkpoint into our simple PolicyNet."""
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    # SKRL saves with keys like 'policy' containing state_dict
    if "policy" in ckpt:
        state = ckpt["policy"]
    else:
        state = ckpt

    # Map SKRL keys to our simple model
    new_state = {}
    for key, val in state.items():
        # Remove _orig_mod. prefix if present (from torch.compile)
        k = key.replace("_orig_mod.", "")
        if k.startswith("policy_net."):
            # policy_net.0.weight -> net.0.weight
            new_key = k.replace("policy_net.", "net.")
            new_state[new_key] = val
        elif k.startswith("mean_layer."):
            new_key = k.replace("mean_layer.", "mean.")
            new_state[new_key] = val

    model = PolicyNet(obs_dim, act_dim).cuda()
    model.load_state_dict(new_state)
    model.eval()
    return model


def eval_trial(ckpt_path: str) -> dict:
    """Run one trial evaluation."""
    env_cfg = registry.default_env_cfg(ENV_NAME)
    for k, v in ENV_OVERRIDES.items():
        if hasattr(env_cfg, k):
            setattr(env_cfg, k, v)
    env_cfg.num_envs = NUM_ENVS
    env = registry.make(ENV_NAME, num_envs=NUM_ENVS, env_cfg=env_cfg)

    obs_dim = env._state.obs.shape[-1]
    act_dim = 12  # VBot has 12 actuators
    model = load_policy(ckpt_path, obs_dim, act_dim)

    state = env.reset()
    completed = np.zeros(NUM_ENVS, dtype=bool)
    total_episodes = 0
    smiley_episodes = 0
    wp_idx_sum = 0
    wp_idx_max_val = 0

    with torch.inference_mode():
        for step in range(MAX_STEPS):
            obs_t = torch.tensor(state.obs, dtype=torch.float32, device="cuda")
            actions = model(obs_t)
            actions = torch.clamp(actions, -1.0, 1.0)
            state = env.step(actions.cpu().numpy())

            done = state.terminated | state.truncated
            newly_done = done & ~completed

            if np.any(newly_done):
                info = state.info
                smileys = info["smileys_reached"][newly_done]
                wp_idx = info["wp_idx"][newly_done]
                any_smiley = np.any(smileys, axis=1)
                smiley_episodes += int(np.sum(any_smiley))
                total_episodes += int(np.sum(newly_done))
                wp_idx_sum += float(np.sum(wp_idx))
                wp_idx_max_val = max(wp_idx_max_val, int(np.max(wp_idx)))

            completed |= done
            if np.all(completed):
                break

    # Count still running
    still_running = ~completed
    if np.any(still_running):
        info = state.info
        smileys = info["smileys_reached"][still_running]
        wp_idx = info["wp_idx"][still_running]
        any_smiley = np.any(smileys, axis=1)
        smiley_episodes += int(np.sum(any_smiley))
        total_episodes += int(np.sum(still_running))
        wp_idx_sum += float(np.sum(wp_idx))
        if len(wp_idx) > 0:
            wp_idx_max_val = max(wp_idx_max_val, int(np.max(wp_idx)))

    env.close()

    smiley_pct = (smiley_episodes / total_episodes * 100) if total_episodes > 0 else 0.0
    wp_mean = (wp_idx_sum / total_episodes) if total_episodes > 0 else 0.0

    return {
        "total": total_episodes,
        "smiley_eps": smiley_episodes,
        "smiley_pct": smiley_pct,
        "wp_mean": wp_mean,
        "wp_max": wp_idx_max_val,
    }


def main():
    print(f"{'Trial':<8} {'Smiley%':>8} {'SmEps':>8} {'Total':>6} {'WP_mean':>8} {'WP_max':>7}")
    print("-" * 52)

    for label, run_dir in TRIALS:
        ckpt = Path(run_dir) / "checkpoints" / "best_agent.pt"
        if not ckpt.exists():
            print(f"{label:<8} {'SKIP':>8}  (no checkpoint)")
            continue

        try:
            r = eval_trial(str(ckpt))
            print(f"{label:<8} {r['smiley_pct']:>7.1f}% {r['smiley_eps']:>8} {r['total']:>6} {r['wp_mean']:>8.3f} {r['wp_max']:>7}")
        except Exception as e:
            import traceback
            print(f"{label:<8} ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
