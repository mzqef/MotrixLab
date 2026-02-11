# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Play 10 VBot robots in 1 environment, all navigating to 1 shared target
"""

import logging
import sys
from pathlib import Path
import numpy as np
import json
import ast

from absl import app, flags
from skrl import config

# Register VBot navigation environments
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vbot  # noqa: F401, E402

from motrix_rl import utils  # noqa: E402
from motrix_rl.skrl import get_log_dir  # noqa: E402

logger = logging.getLogger(__name__)

_POLICY = flags.DEFINE_string("policy", None, "The policy to load (auto-discover if not specified)")
_SEED = flags.DEFINE_integer("seed", 42, "Random seed for reproducibility")
_TARGET_X = flags.DEFINE_float("target-x", 0.0, "Target X position (center of circular platform)")
_TARGET_Y = flags.DEFINE_float("target-y", 0.0, "Target Y position (center of circular platform)")
_TARGET_YAW = flags.DEFINE_float("target-yaw", 0.0, "Target heading (radians)")
_SPAWN_RADIUS = flags.DEFINE_float("spawn-radius", 8.0, "Radius of circle to spread robots around target")
_SPAWN_INNER_RADIUS = flags.DEFINE_float(
    "spawn-inner-radius",
    0.0,
    "Inner radius for annulus spawn (use with --spawn-outer-radius)",
)
_SPAWN_OUTER_RADIUS = flags.DEFINE_float(
    "spawn-outer-radius",
    0.0,
    "Outer radius for annulus spawn (0 to disable; uses --spawn-radius circle)",
)
_MAX_EPISODE_STEPS = flags.DEFINE_integer("max-episode-steps", 1000, "Max steps per episode")


def find_best_policy(env_name: str) -> str:
    """Find the most recent best policy for the given environment"""
    env_dir = Path(get_log_dir(env_name))
    
    if not env_dir.exists():
        raise FileNotFoundError(f"No training results found for environment '{env_name}' in {env_dir}")
    
    training_runs = [d for d in env_dir.iterdir() if d.is_dir()]
    
    if not training_runs:
        raise FileNotFoundError(f"No training runs found for environment '{env_name}'")
    
    # Get the most recent run
    latest_run = max(training_runs, key=lambda x: x.stat().st_mtime)
    checkpoints_dir = latest_run / "checkpoints"
    
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in {latest_run}")
    
    # Try best_agent files first
    best_files = list(checkpoints_dir.glob("best_agent.*"))
    
    if best_files:
        return str(best_files[0])
    
    # Otherwise find the latest checkpoint
    checkpoint_files = list(checkpoints_dir.glob("agent_*.pt")) + list(checkpoints_dir.glob("agent_*.pickle"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No policy files found in {checkpoints_dir}")
    
    def extract_timestep(filename):
        stem = Path(filename).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return 0
        return 0
    
    latest_checkpoint = max(checkpoint_files, key=extract_timestep)
    return str(latest_checkpoint)


def get_inference_backend(policy_path: str):
    if policy_path.endswith(".pt"):
        return "torch"
    elif policy_path.endswith(".pickle"):
        return "jax"
    else:
        raise Exception(f"Unknown policy format: {policy_path}")


def play_with_single_target(env_name: str, policy_path: str, backend: str, 
                            num_envs: int, seed: int, target_x: float, 
                            target_y: float, target_yaw: float,
                            spawn_radius: float, spawn_inner_radius: float,
                            spawn_outer_radius: float, enable_render: bool):
    """Custom play function that creates environment with single target override.
    
    All robots are placed in a circle around the shared target and navigate
    toward it from different directions. render_spacing is set to 0 so all
    envs share the same visual world space.
    """
    import time
    import torch
    from skrl.utils import set_seed
    from motrix_envs import registry as env_registry
    from motrix_rl import registry as rl_registry
    from motrix_rl.skrl.torch import wrap_env
    
    # Get RL config
    rlcfg = rl_registry.default_rl_cfg(env_name, "skrl", backend="torch")
    rlcfg = rlcfg.replace(play_num_envs=num_envs, seed=seed)

    # Load model sizes from the training run metadata when available.
    try:
        policy_path_obj = Path(policy_path)
        run_dir = policy_path_obj.parent.parent
        meta_path = run_dir / "experiment_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            overrides = meta.get("rl_overrides", {})
            policy_sizes = overrides.get("policy_hidden_layer_sizes")
            value_sizes = overrides.get("value_hidden_layer_sizes")
            if policy_sizes:
                rlcfg = rlcfg.replace(policy_hidden_layer_sizes=ast.literal_eval(policy_sizes))
            if value_sizes:
                rlcfg = rlcfg.replace(value_hidden_layer_sizes=ast.literal_eval(value_sizes))
    except Exception as exc:
        logger.warning("Failed to read model sizes from metadata: %s", exc)
    
    # Create environment
    env = env_registry.make(env_name, sim_backend=None, num_envs=num_envs)
    env._cfg.max_episode_steps = _MAX_EPISODE_STEPS.value
    set_seed(seed)
    
    # ===== FIX 1: Collapse all envs into same visual space =====
    env._render_spacing = 0.0
    
    # Pre-compute spawn positions around the target
    shared_target = np.array([target_x, target_y, target_yaw], dtype=np.float32)
    
    def _make_spawn_positions(n, radius, inner_radius, outer_radius):
        """Generate n random positions on a circle or annulus around the target."""
        angles = np.random.uniform(0.0, 2 * np.pi, size=n)
        if outer_radius > 0.0:
            inner = max(0.0, inner_radius)
            outer = max(outer_radius, inner)
            # Uniform area sampling in annulus.
            r = np.sqrt(np.random.uniform(0.0, 1.0, size=n) * (outer**2 - inner**2) + inner**2)
        else:
            r = np.full(n, radius)
        xs = target_x + r * np.cos(angles)
        ys = target_y + r * np.sin(angles)
        return xs, ys, angles
    
    # Override reset to:
    #   1. Replace random targets with the shared target
    #   2. Spread robots in a circle around the target
    #   3. Face each robot toward the target
    #   4. Recompute observations for the correct target
    original_reset = env.reset
    
    def custom_reset(data, done=None):
        obs, info = original_reset(data, done)
        n = data.shape[0]
        
        # Override target for all envs
        info["pose_commands"] = np.tile(shared_target, (n, 1))
        reset_mask = np.ones(n, dtype=bool) if done is None else np.asarray(done, dtype=bool)
        if not np.any(reset_mask):
            return obs, info

        reset_indices = np.where(reset_mask)[0]
        circle_x, circle_y, circle_angles = _make_spawn_positions(
            len(reset_indices), spawn_radius, spawn_inner_radius, spawn_outer_radius
        )
        spawn_height = env._cfg.init_state.pos[2]  # default Z height
        
        dof_pos = data.dof_pos.copy()
        dof_pos[reset_indices, 3] = circle_x   # base X
        dof_pos[reset_indices, 4] = circle_y   # base Y
        dof_pos[reset_indices, 5] = spawn_height  # base Z
        
        # Face each robot toward the center (target)
        for idx, angle in zip(reset_indices, circle_angles):
            face_yaw = angle + np.pi  # point toward center
            quat = env._euler_to_quat(0, 0, face_yaw)
            quat = quat / (np.linalg.norm(quat) + 1e-8)
            dof_pos[idx, 6:10] = quat
        
        data.set_dof_pos(dof_pos, env._model)
        env._model.forward_kinematic(data)
        
        # ===== FIX 4: Recompute observations for repositioned robots =====
        root_pose = env._body.get_pose(data)
        root_pos = root_pose[:, :3]
        root_quat = root_pose[:, 3:7]
        robot_position = root_pos[:, :2]
        robot_heading = env._get_heading_from_quat(root_quat)
        
        target_position = info["pose_commands"][:, :2]
        target_heading_val = info["pose_commands"][:, 2]
        
        # Position error & distance
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        # Heading error
        heading_diff = target_heading_val - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)
        
        position_threshold = 0.3
        reached_all = distance_to_target < position_threshold
        
        # Velocity commands (P-controller toward target)
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2 * np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2 * np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()
        
        velocity_commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1)
        command_normalized = velocity_commands * env.commands_scale
        
        # Recompute all observation components from the new positions
        base_lin_vel = env._model.get_sensor_value(env._cfg.sensor.base_linvel, data)
        gyro = env._model.get_sensor_value(env._cfg.sensor.base_gyro, data)
        projected_gravity = env._compute_projected_gravity(root_quat)
        joint_pos = env.get_dof_pos(data)
        joint_vel = env.get_dof_vel(data)
        joint_pos_rel = joint_pos - env.default_angles
        
        noisy_linvel = base_lin_vel * env._cfg.normalization.lin_vel
        noisy_gyro = gyro * env._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * env._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * env._cfg.normalization.dof_vel
        last_actions = np.zeros((n, env._num_action), dtype=np.float32)
        
        position_error_normalized = position_error / 12.0  # Must match training normalization
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 12.0, 0, 1)  # Must match training normalization
        reached_flag = reached_all.astype(np.float32)
        stop_ready = np.logical_and(reached_all, np.abs(gyro[:, 2]) < 5e-2)
        stop_ready_flag = stop_ready.astype(np.float32)
        
        # Rebuild full observation vector
        obs[:] = np.concatenate([
            noisy_linvel,                                     # 3
            noisy_gyro,                                       # 3
            projected_gravity,                                # 3
            noisy_joint_angle,                                # 12
            noisy_joint_vel,                                  # 12
            last_actions,                                     # 12
            command_normalized,                               # 3
            position_error_normalized,                        # 2
            heading_error_normalized[:, np.newaxis],          # 1
            distance_normalized[:, np.newaxis],               # 1
            reached_flag[:, np.newaxis],                      # 1
            stop_ready_flag[:, np.newaxis],                   # 1
        ], axis=-1)
        
        # Update distances
        info["min_distance"] = distance_to_target.copy()
        info["initial_distance"] = distance_to_target.copy()
        
        # Update target marker and heading arrows
        env._update_target_marker(data, info["pose_commands"])
        base_lin_vel_xy = base_lin_vel[:, :2]
        env._update_heading_arrows(data, root_pos, position_error, base_lin_vel_xy)
        
        return obs, info
    
    env.reset = custom_reset
    
    # Disable success truncation so robots stay visible after reaching center
    # (during training this truncates to save steps, but during demo it looks like vanishing)
    env._success_truncate = np.zeros(num_envs, dtype=bool)
    from motrix_envs.np.env import NpEnv

    def _play_update_truncate():
        # Only apply max_episode_steps truncation (no success truncation)
        NpEnv._update_truncate(env)
        env._success_truncate = np.zeros(num_envs, dtype=bool)
    env._update_truncate = _play_update_truncate
    
    def _termination_causes(state):
        data = state.data
        info = state.info
        base_contact = np.zeros(env._num_envs, dtype=bool)
        if getattr(env, "_has_base_contact_sensor", False):
            try:
                base_contact_value = env._model.get_sensor_value("base_contact", data)
                if base_contact_value.ndim == 0:
                    base_contact = np.array([base_contact_value > 0.01], dtype=bool)
                elif base_contact_value.shape[0] != env._num_envs:
                    base_contact = np.full(env._num_envs, base_contact_value.flatten()[0] > 0.01, dtype=bool)
                else:
                    base_contact = (base_contact_value > 0.01).flatten()[: env._num_envs]
            except BaseException:
                base_contact = np.zeros(env._num_envs, dtype=bool)

        projected_gravity = env._compute_projected_gravity(env._body.get_pose(data)[:, 3:7])
        gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
        gz = projected_gravity[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip = tilt_angle > np.deg2rad(75)

        joint_vel = env.get_dof_vel(data)
        vel_max = np.abs(joint_vel).max(axis=1)
        joint_overflow = vel_max > env._cfg.max_dof_vel
        joint_extreme = np.isnan(joint_vel).any(axis=1) | np.isinf(joint_vel).any(axis=1)
        joint_overflow = joint_overflow | joint_extreme

        max_steps = info.get("steps", np.zeros(env._num_envs, dtype=np.int32)) >= env._cfg.max_episode_steps
        success_truncate = getattr(env, "_success_truncate", np.zeros(env._num_envs, dtype=bool))

        return base_contact, side_flip, joint_overflow, max_steps, success_truncate

    original_reset_done_envs = env._reset_done_envs
    env._last_done_causes = None

    def _play_reset_done_envs():
        state = env._state
        done = state.done
        if np.any(done):
            base_contact, side_flip, joint_overflow, max_steps, success_truncate = _termination_causes(state)
            env._last_done_causes = {
                "done": done.copy(),
                "base_contact": base_contact,
                "side_flip": side_flip,
                "joint_overflow": joint_overflow,
                "max_steps": max_steps,
                "success_truncate": success_truncate,
            }
        else:
            env._last_done_causes = None
        original_reset_done_envs()

    env._reset_done_envs = _play_reset_done_envs

    # Wrap environment for SKRL
    wrapped_env = wrap_env(env, enable_render)
    
    # Create models and agent (import trainer to reuse model building)
    from motrix_rl.skrl.torch.train.ppo import Trainer
    temp_trainer = Trainer(env_name, None, enable_render=False)
    temp_trainer._rlcfg = rlcfg
    models = temp_trainer._make_model(wrapped_env, rlcfg)
    from motrix_rl.skrl.torch.train.ppo import _get_cfg
    ppo_cfg = _get_cfg(rlcfg, wrapped_env)
    agent = temp_trainer._make_agent(models, wrapped_env, ppo_cfg)
    
    # Load policy
    agent.load(policy_path)
    logger.info(f"Loaded policy from: {policy_path}")
    
    # Play loop
    with torch.inference_mode():
        obs, _ = wrapped_env.reset()
        fps = 60
        logger.info(f"Playing with {num_envs} robots navigating to target ({target_x}, {target_y}, {target_yaw})")
        while True:
            t = time.time()
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = wrapped_env.step(actions)
            if torch.is_tensor(terminated):
                terminated = terminated.detach().cpu().numpy()
            if torch.is_tensor(truncated):
                truncated = truncated.detach().cpu().numpy()
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                causes = getattr(env, "_last_done_causes", None)
                if causes is None:
                    done_indices = np.where(done)[0]
                    for idx in done_indices:
                        logger.info("env %d done: unknown", idx)
                else:
                    done_indices = np.where(causes["done"])[0]
                    for idx in done_indices:
                        reasons = []
                        if causes["base_contact"][idx]:
                            reasons.append("base_contact")
                        if causes["side_flip"][idx]:
                            reasons.append("side_flip")
                        if causes["joint_overflow"][idx]:
                            reasons.append("joint_overflow")
                        if causes["max_steps"][idx]:
                            reasons.append("max_episode_steps")
                        if causes["success_truncate"][idx]:
                            reasons.append("success_truncate")
                        reason_text = ", ".join(reasons) if reasons else "unknown"
                        logger.info("env %d done: %s", idx, reason_text)
            wrapped_env.render()
            delta_time = time.time() - t
            if delta_time < 1.0 / fps:
                time.sleep(1.0 / fps - delta_time)


def main(argv):
    device_supports = utils.get_device_supports()
    logger.info(device_supports)
    
    env_name = "vbot_navigation_section001"
    num_envs = 10  # 10 robots
    enable_render = True
    
    # Determine policy path
    if _POLICY.present:
        policy_path = _POLICY.value
        logger.info(f"Using specified policy: {policy_path}")
    else:
        try:
            policy_path = find_best_policy(env_name)
            logger.info(f"Auto-discovered best policy: {policy_path}")
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            logger.error("Please specify a policy using --policy flag or train a model first")
            return
    
    backend = get_inference_backend(policy_path)
    
    if backend == "jax":
        logger.error("JAX backend not yet implemented for this script. Please train with PyTorch.")
        return
        
    elif backend == "torch":
        assert device_supports.torch, "PyTorch is not available on your device"
        
        play_with_single_target(
            env_name=env_name,
            policy_path=policy_path,
            backend=backend,
            num_envs=num_envs,
            seed=_SEED.value,
            target_x=_TARGET_X.value,
            target_y=_TARGET_Y.value,
            target_yaw=_TARGET_YAW.value,
            spawn_radius=_SPAWN_RADIUS.value,
            spawn_inner_radius=_SPAWN_INNER_RADIUS.value,
            spawn_outer_radius=_SPAWN_OUTER_RADIUS.value,
            enable_render=enable_render
        )


if __name__ == "__main__":
    app.run(main)
