"""Temporary play script with relaxed termination for debugging bump traversal.
Monkey-patches _compute_terminated to be much more lenient:
  - Hard tilt: 70° → 85°
  - Soft tilt: 50° → 80°
  - Base contact: disabled
  - Grace period: 100 → 500 steps (5 seconds)
  - Stagnation: disabled
"""
import sys
from pathlib import Path
import numpy as np

# --- Setup imports (same as play.py) ---
sys.path.insert(0, str(Path(__file__).resolve().parent / "starter_kit" / "navigation2"))
import vbot as navigation2_vbot  # noqa: F401, E402

from motrix_envs import registry as env_registry

# --- Monkey-patch the environment class ---
_ENV_NAME = "vbot_navigation_section013"
_meta = env_registry._envs.get(_ENV_NAME)
if _meta is None:
    raise RuntimeError(f"Environment {_ENV_NAME} not registered")
env_cls = _meta.env_cls_dict.get("np")
if env_cls is None:
    raise RuntimeError("No 'np' backend for vbot_navigation_section011")

_original_compute_terminated = env_cls._compute_terminated

def _relaxed_compute_terminated(self, state, projected_gravity=None, joint_vel=None, robot_xy=None, current_z=None):
    """Relaxed termination: only terminate on extreme physics failure."""
    data = state.data
    n = self._num_envs

    hard_terminated = np.zeros(n, dtype=bool)

    # Only terminate on extreme tilt (85°) — basically upside-down
    if projected_gravity is not None:
        gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
        gz = projected_gravity[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        hard_terminated |= tilt_angle > np.deg2rad(85)

    # Keep OOB (can't disable competition bounds)
    bounds = getattr(self._cfg, 'course_bounds', None)
    if bounds is not None and robot_xy is not None and current_z is not None:
        oob_x = (robot_xy[:, 0] < bounds.x_min) | (robot_xy[:, 0] > bounds.x_max)
        oob_y = (robot_xy[:, 1] < bounds.y_min) | (robot_xy[:, 1] > bounds.y_max)
        oob_z = current_z < bounds.z_min
        oob = oob_x | oob_y | oob_z
        hard_terminated |= oob
        state.info["oob_terminated"] = oob

    # Keep NaN/physics explosion checks
    if joint_vel is not None:
        vel_max = np.abs(joint_vel).max(axis=1)
        vel_overflow = vel_max > self._cfg.max_dof_vel
        vel_extreme = np.isnan(joint_vel).any(axis=1) | np.isinf(joint_vel).any(axis=1)
        hard_terminated |= vel_overflow | vel_extreme
        last_dof_vel = state.info.get("last_dof_vel", np.zeros_like(joint_vel))
        dof_acc_max = np.abs(joint_vel - np.clip(last_dof_vel, -100.0, 100.0)).max(axis=1)
        hard_terminated |= dof_acc_max > 80.0
    nan_terminated = state.info.get("nan_terminated", np.zeros(n, dtype=bool))
    hard_terminated |= nan_terminated

    # NO soft termination (no base contact, no 50° tilt)
    terminated = hard_terminated
    return state.replace(terminated=terminated)

env_cls._compute_terminated = _relaxed_compute_terminated

# Also disable stagnation truncation
_original_update_truncate = env_cls._update_truncate

def _relaxed_update_truncate(self):
    # Call grandparent (NpEnv) truncation for max episode length only
    from motrix_envs.np.env import NpEnv
    NpEnv._update_truncate(self)
    # Skip stagnation and success truncation

env_cls._update_truncate = _relaxed_update_truncate

print("[RELAXED PLAY] Termination patched: base_contact=OFF, soft_tilt=OFF, hard_tilt=85°, stagnation=OFF")

# --- Now run play.py logic ---
from absl import app, flags
from skrl import config
from motrix_rl import utils
from motrix_rl.skrl import get_log_dir
import logging

logger = logging.getLogger(__name__)

_POLICY = flags.DEFINE_string("policy", None, "The policy to load")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 1, "Number of envs to play")

def main(argv):
    env_name = _ENV_NAME
    policy_path = _POLICY.value
    if not policy_path:
        print("ERROR: --policy required")
        return

    device_supports = utils.get_device_supports()
    rl_override = {"play_num_envs": _NUM_ENVS.value}

    if policy_path.endswith(".pt"):
        from motrix_rl.skrl.torch.train import ppo
        config.torch.backend = "torch"
        trainer = ppo.Trainer(env_name, None, cfg_override=rl_override, enable_render=True)
        trainer.play(policy_path)
    elif policy_path.endswith(".pickle"):
        from motrix_rl.skrl.jax.train import ppo
        config.jax.backend = "jax"
        trainer = ppo.Trainer(env_name, None, cfg_override=rl_override, enable_render=True)
        trainer.play(policy_path)

if __name__ == "__main__":
    app.run(main)
