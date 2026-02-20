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
VBot全程导航环境（三段地形：section01 + section02 + section03）
使用多航点系统引导机器人穿越三段地形。
观测空间与单段任务完全一致（54维），确保策略可互换。
"""

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotLongCourseEnvCfg, TerrainScaleHelper


# 航点列表：引导机器人穿越三段地形（左路线：楼梯→桥→楼梯下）
# 每个航点 = (x, y)，机器人到达后自动切换到下一个
WAYPOINTS = [
    (0.0, 6.0),      # WP0: section01 高台顶端出口
    (-3.0, 12.0),    # WP1: 左侧楼梯入口
    (-3.0, 15.0),    # WP2: 桥起点
    (-3.0, 20.5),    # WP3: 桥终点
    (-3.0, 23.0),    # WP4: 左侧楼梯2底部（下楼梯）
    (0.0, 24.5),     # WP5: section02 出口平台
    (0.0, 32.3),     # WP6: section03 最终平台（终点）
]
WAYPOINT_THRESHOLD = 1.5  # 到达判定距离（米）- 稍放宽以适应复杂地形
FINAL_THRESHOLD = 0.8     # 最终目标到达判定（更精确）


@registry.env("vbot_navigation_long_course", "np")
class VBotLongCourseEnv(NpEnv):
    """
    VBot全程导航环境：三段地形合并（section01→02→03）
    section01: hfield + 15°坡道 + 高台(顶面z=1.294)
    section02: 左右楼梯 + 拱桥 + 球形/锥形障碍物，y≈10.33~24.33
    section03: 隔离墙 + 21.8°坡道 + 3金球 + 最终平台(顶面z=1.494)，y≈24.33~34.33
    使用多航点系统管理目标切换，与单段任务共享54维观测空间。
    """
    _cfg: VBotLongCourseEnvCfg

    def __init__(self, cfg: VBotLongCourseEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        # 初始化机器人body
        self._body = self._model.get_body(cfg.asset.body_name)

        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")

        # 获取箭头body（用于可视化）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None

        # 动作和观测空间（与单段任务一致：54维）
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)

        # DOF索引（与section011相同结构）
        self._find_target_marker_dof_indices()
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()

        # 初始化缓存
        self._init_buffer()

        # 初始位置
        self.spawn_center = np.array(cfg.init_state.pos, dtype=np.float32)
        self.spawn_range = 0.1

        # 航点系统
        self.waypoints = np.array(WAYPOINTS, dtype=np.float32)
        self.num_waypoints = len(WAYPOINTS)

        # 导航统计计数器
        self.navigation_stats_step = 0
        
        # 注意：全程模式不使用geom-based终止检测
        # 因为三段地形使用不同前缀(S1_/S2_/S3_)而非统一的C_前缀
        # 改用sensor-based终止: base_contact_s1 / base_contact_s2 / base_contact_s3
        self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
        self.num_termination_check = 0
        self.num_foot_check = 4

    def _init_buffer(self):
        """初始化缓存和参数"""
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)

        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )

        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle

        self._init_dof_pos[-self._num_action:] = self.default_angles
        self.action_filter_alpha = 0.3
        # 多地形动态action_scale (基于TerrainZone表)
        self._terrain_scale = TerrainScaleHelper(cfg.control_config)

    def _update_dynamic_action_scale(self, info: dict, data: mtx.SceneData) -> np.ndarray:
        root_pos, _, _ = self._extract_root_state(data)
        probe_y = root_pos[:, 1]
        return self._terrain_scale.update(info, probe_y, data.shape[0])

    def _find_target_marker_dof_indices(self):
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10

    def _find_arrow_dof_indices(self):
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36

        arrow_init_height = self._cfg.init_state.pos[2] + 0.5
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]

    # ================================================================
    # State extraction helpers
    # ================================================================

    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)

    def _extract_root_state(self, data):
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    # ================================================================
    # Physics helpers
    # ================================================================

    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)

    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _euler_to_quat(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return np.array([qx, qy, qz, qw], dtype=np.float32)

    # ================================================================
    # Action
    # ================================================================

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]

        if "filtered_actions" not in state.info:
            state.info["filtered_actions"] = actions
        else:
            state.info["filtered_actions"] = (
                self.action_filter_alpha * actions
                + (1.0 - self.action_filter_alpha) * state.info["filtered_actions"]
            )

        state.info["current_actions"] = state.info["filtered_actions"]
        current_scale = self._update_dynamic_action_scale(state.info, state.data)
        state.data.actuator_ctrls = self._compute_torques(state.info["filtered_actions"], state.data, current_scale)
        return state

    def _compute_torques(self, actions, data, current_scale=None):
        if current_scale is None:
            current_scale = self._cfg.control_config.action_scale
        action_scaled = actions * current_scale[:, np.newaxis] if np.ndim(current_scale) > 0 else actions * current_scale
        target_pos = self.default_angles + action_scaled

        current_pos = self.get_dof_pos(data)
        current_vel = self.get_dof_vel(data)

        kp = 100.0
        kv = 8.0
        torques = kp * (target_pos - current_pos) - kv * current_vel
        self._raw_torques = torques.copy()

        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)
        torques = np.clip(torques, -torque_limits, torque_limits)
        return torques

    # ================================================================
    # Waypoint management
    # ================================================================

    def _advance_waypoints(self, robot_xy: np.ndarray, waypoint_idx: np.ndarray) -> np.ndarray:
        """
        检查并推进航点：如果机器人到达当前航点，切换到下一个。
        返回更新后的waypoint_idx。
        """
        new_idx = waypoint_idx.copy()
        for i in range(self._num_envs):
            wp_i = int(new_idx[i])
            if wp_i >= self.num_waypoints:
                continue  # 已到达最终目标
            wp_pos = self.waypoints[wp_i]
            dist = np.linalg.norm(robot_xy[i] - wp_pos)

            # 中间航点使用较大阈值，最终航点使用精确阈值
            threshold = FINAL_THRESHOLD if wp_i == self.num_waypoints - 1 else WAYPOINT_THRESHOLD
            if dist < threshold:
                new_idx[i] = min(wp_i + 1, self.num_waypoints - 1)
        return new_idx

    def _get_current_target(self, waypoint_idx: np.ndarray) -> np.ndarray:
        """根据当前航点索引获取目标位置 [num_envs, 2]"""
        idx = np.clip(waypoint_idx.astype(int), 0, self.num_waypoints - 1)
        return self.waypoints[idx]  # [num_envs, 2]

    # ================================================================
    # Visualization helpers
    # ================================================================

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(pose_commands[env_idx, 2])
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_yaw
            ]
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)

    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray,
                                desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return

        num_envs = data.shape[0]
        arrow_offset = 0.5
        all_dof_pos = data.dof_pos.copy()

        for env_idx in range(num_envs):
            arrow_height = robot_pos[env_idx, 2] + arrow_offset

            # 当前运动方向箭头
            cur_v = base_lin_vel_xy[env_idx]
            cur_yaw = np.arctan2(cur_v[1], cur_v[0]) if np.linalg.norm(cur_v) > 1e-3 else 0.0
            robot_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            qn = np.linalg.norm(robot_arrow_quat)
            robot_arrow_quat = robot_arrow_quat / qn if qn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([robot_arrow_pos, robot_arrow_quat])

            # 期望方向箭头
            des_v = desired_vel_xy[env_idx]
            des_yaw = np.arctan2(des_v[1], des_v[0]) if np.linalg.norm(des_v) > 1e-3 else 0.0
            desired_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            qn = np.linalg.norm(desired_arrow_quat)
            desired_arrow_quat = desired_arrow_quat / qn if qn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([desired_arrow_pos, desired_arrow_quat])

        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)

    # ================================================================
    # Termination (sensor-based, not geom-based)
    # ================================================================

    def _compute_terminated(self, state: NpEnvState, projected_gravity: np.ndarray = None,
                            joint_vel: np.ndarray = None) -> NpEnvState:
        """
        终止条件：OR聚合三段地形的base_contact传感器 + 侧翻 + 关节速度异常
        """
        data = state.data
        terminated = np.zeros(self._num_envs, dtype=bool)

        # 聚合三段地形的base_contact传感器
        for sensor_name in ("base_contact_s1", "base_contact_s2", "base_contact_s3"):
            try:
                val = self._model.get_sensor_value(sensor_name, data)
                if val.ndim == 0:
                    contact = np.array([val > 0.01], dtype=bool)
                elif val.shape[0] != self._num_envs:
                    contact = np.full(self._num_envs, val.flatten()[0] > 0.01, dtype=bool)
                else:
                    contact = (val > 0.01).flatten()[:self._num_envs]
                terminated = np.logical_or(terminated, contact)
            except Exception:
                pass  # 传感器可能不存在，静默忽略

        # 侧翻检测
        if projected_gravity is not None:
            gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
            gz = projected_gravity[:, 2]
            tilt_angle = np.arctan2(gxy, np.abs(gz))
            terminated = np.logical_or(terminated, tilt_angle > np.deg2rad(75))

        # 关节速度异常
        if joint_vel is not None:
            vel_max = np.abs(joint_vel).max(axis=1)
            vel_overflow = vel_max > self._cfg.max_dof_vel
            vel_extreme = np.isnan(joint_vel).any(axis=1) | np.isinf(joint_vel).any(axis=1)
            terminated = np.logical_or(terminated, vel_overflow | vel_extreme)

        return state.replace(terminated=terminated)

    # ================================================================
    # Reward
    # ================================================================

    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray,
                         base_lin_vel: np.ndarray, gyro: np.ndarray, projected_gravity: np.ndarray,
                         joint_vel: np.ndarray, distance_to_target: np.ndarray, heading_diff: np.ndarray,
                         position_error: np.ndarray, reached_all: np.ndarray,
                         terminated: np.ndarray, robot_heading: np.ndarray = None) -> np.ndarray:
        """
        全程导航奖励：基础导航奖励 + 航点到达奖励 + 最终到达大奖
        """
        scales = self._cfg.reward_config.scales
        term_scale = scales.get("termination", -100.0)
        termination_penalty = np.where(terminated, term_scale, 0.0)

        # ===== 导航核心奖励 =====
        position_tracking = np.exp(-distance_to_target / 5.0)
        fine_position_tracking = np.where(distance_to_target < 1.5, np.exp(-distance_to_target / 0.3), 0.0)

        target_bearing = np.arctan2(position_error[:, 1], position_error[:, 0])
        facing_error = target_bearing - robot_heading
        facing_error = np.where(facing_error > np.pi, facing_error - 2 * np.pi, facing_error)
        facing_error = np.where(facing_error < -np.pi, facing_error + 2 * np.pi, facing_error)
        heading_tracking = np.where(reached_all, 1.0, np.exp(-np.abs(facing_error) / 0.5))

        direction_to_target = position_error / (np.linalg.norm(position_error, axis=1, keepdims=True) + 1e-8)
        forward_velocity = np.clip(np.sum(base_lin_vel[:, :2] * direction_to_target, axis=1), -1.0, 2.0)

        # 距离进步奖励
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        distance_improvement = info["min_distance"] - distance_to_target
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        approach_reward = np.clip(distance_improvement * scales.get("approach_scale", 8.0), -1.0, 1.0)

        initial_distance = info.get("initial_distance", distance_to_target)
        distance_progress = np.clip(1.0 - distance_to_target / (initial_distance + 1e-8), -0.5, 1.0)

        # 存活奖励
        ever_reached = info.get("ever_reached_final", np.zeros(self._num_envs, dtype=bool))
        alive_bonus = np.where(ever_reached, 0.0, 1.0)

        # 时间衰减
        steps = info.get("steps", np.zeros(self._num_envs, dtype=np.float32)).astype(np.float32)
        max_steps = float(self._cfg.max_episode_steps)
        time_decay = np.clip(1.0 - 0.5 * steps / max_steps, 0.5, 1.0)

        # ===== 航点奖励 =====
        waypoint_bonus = info.get("waypoint_bonus_this_step", np.zeros(self._num_envs, dtype=np.float32))

        # ===== 最终到达奖励 =====
        info["ever_reached_final"] = info.get("ever_reached_final", np.zeros(self._num_envs, dtype=bool))
        first_time_reach = np.logical_and(reached_all, ~info["ever_reached_final"])
        info["ever_reached_final"] = np.logical_or(info["ever_reached_final"], reached_all)
        arrival_bonus = np.where(first_time_reach, scales.get("arrival_bonus", 100.0), 0.0)

        # ===== 停止奖励（到达最终目标后） =====
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        stop_base = scales.get("stop_scale", 2.0) * (
            0.8 * np.exp(-(speed_xy / 0.2) ** 2) + 1.2 * np.exp(-(np.abs(gyro[:, 2]) / 0.1) ** 4)
        )
        zero_ang_mask = np.abs(gyro[:, 2]) < 0.05
        zero_ang_bonus = np.where(
            np.logical_and(reached_all, zero_ang_mask),
            scales.get("zero_ang_bonus", 6.0), 0.0
        )
        stop_bonus = np.where(reached_all, stop_base + zero_ang_bonus, 0.0)

        # ===== 惩罚 =====
        orientation_penalty = np.square(projected_gravity[:, 0]) + np.square(projected_gravity[:, 1])
        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1)
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=1)
        dof_vel_penalty = np.sum(np.square(joint_vel), axis=1)

        last_dof_vel = info.get("last_dof_vel", np.zeros_like(joint_vel))
        dof_acc = joint_vel - last_dof_vel
        dof_acc_penalty = np.sum(np.square(dof_acc), axis=1)

        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        penalties = (
            scales.get("orientation", -0.05) * orientation_penalty
            + scales.get("lin_vel_z", -0.3) * lin_vel_z_penalty
            + scales.get("ang_vel_xy", -0.03) * ang_vel_xy_penalty
            + scales.get("torques", -1e-5) * torque_penalty
            + scales.get("dof_vel", -5e-5) * dof_vel_penalty
            + scales.get("dof_acc", -2.5e-7) * dof_acc_penalty
            + scales.get("action_rate", -0.01) * action_rate_penalty
            + termination_penalty
        )

        # ===== 总奖励 =====
        reward = np.where(
            reached_all,
            stop_bonus + arrival_bonus + waypoint_bonus + penalties,
            time_decay * (
                scales.get("position_tracking", 1.5) * position_tracking
                + scales.get("fine_position_tracking", 5.0) * fine_position_tracking
                + scales.get("heading_tracking", 0.8) * heading_tracking
                + scales.get("forward_velocity", 1.5) * forward_velocity
                + scales.get("distance_progress", 2.0) * distance_progress
                + scales.get("alive_bonus", 0.5) * alive_bonus
                + approach_reward
            ) + waypoint_bonus + penalties
        )

        # TensorBoard日志
        info["Reward"] = {
            "position_tracking": scales.get("position_tracking", 1.5) * position_tracking,
            "fine_position_tracking": scales.get("fine_position_tracking", 5.0) * fine_position_tracking,
            "heading_tracking": scales.get("heading_tracking", 0.8) * heading_tracking,
            "forward_velocity": scales.get("forward_velocity", 1.5) * forward_velocity,
            "approach_reward": approach_reward,
            "distance_progress": scales.get("distance_progress", 2.0) * distance_progress,
            "alive_bonus": scales.get("alive_bonus", 0.5) * alive_bonus,
            "waypoint_bonus": waypoint_bonus,
            "arrival_bonus": arrival_bonus,
            "stop_bonus": stop_bonus,
            "penalties": penalties,
            "termination": termination_penalty,
        }
        return reward

    # ================================================================
    # Update state
    # ================================================================

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        cfg = self._cfg

        # 获取基础状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # ===== 航点管理 =====
        robot_xy = root_pos[:, :2]
        waypoint_idx = state.info["waypoint_idx"]

        # 尝试推进航点
        old_idx = waypoint_idx.copy()
        waypoint_idx = self._advance_waypoints(robot_xy, waypoint_idx)
        state.info["waypoint_idx"] = waypoint_idx

        # 航点到达奖励（仅在刚到达新航点时给予）
        advanced = waypoint_idx > old_idx
        waypoint_bonus_val = self._cfg.reward_config.scales.get("waypoint_bonus", 30.0)
        state.info["waypoint_bonus_this_step"] = np.where(advanced, waypoint_bonus_val, 0.0)

        # 当前目标 = 当前航点位置
        current_target_xy = self._get_current_target(waypoint_idx)

        # 构造与单段任务兼容的 pose_commands [x, y, heading]
        # 朝向设为从当前位置指向目标的方向
        target_heading = np.arctan2(
            current_target_xy[:, 1] - robot_xy[:, 1],
            current_target_xy[:, 0] - robot_xy[:, 0]
        )
        pose_commands = np.column_stack([current_target_xy, target_heading])
        state.info["pose_commands"] = pose_commands

        # ===== 导航计算（与单段任务一致） =====
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]

        position_error = target_position - robot_xy
        distance_to_target = np.linalg.norm(position_error, axis=1)

        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        # 最终目标到达判定
        at_final = waypoint_idx >= self.num_waypoints - 1
        reached_all = np.logical_and(at_final, distance_to_target < FINAL_THRESHOLD)

        # 期望速度
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

        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )

        # ===== 观测构造（54维，与单段任务一致） =====
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]

        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)

        stop_ready = np.logical_and(reached_all, np.abs(gyro[:, 2]) < 5e-2)
        stop_ready_flag = stop_ready.astype(np.float32)

        obs = np.concatenate(
            [
                noisy_linvel,                                  # 3
                noisy_gyro,                                    # 3
                projected_gravity,                             # 3
                noisy_joint_angle,                             # 12
                noisy_joint_vel,                               # 12
                last_actions,                                  # 12
                command_normalized,                            # 3
                position_error_normalized,                     # 2
                heading_error_normalized[:, np.newaxis],       # 1
                distance_normalized[:, np.newaxis],            # 1
                reached_flag[:, np.newaxis],                   # 1
                stop_ready_flag[:, np.newaxis],                # 1
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 54)

        # 更新可视化标记
        self._update_target_marker(data, pose_commands)
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel[:, :2])

        # 终止检测
        terminated_state = self._compute_terminated(state, projected_gravity, joint_vel)
        terminated = terminated_state.terminated

        # 奖励计算
        reward = self._compute_reward(
            data, state.info, velocity_commands,
            base_lin_vel, gyro, projected_gravity,
            joint_vel, distance_to_target, heading_diff,
            position_error, reached_all, terminated, robot_heading
        )

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        # 成功截断：到达最终目标且停稳
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        reach_and_stopped = np.logical_and(reached_all, speed_xy < 0.15)
        reach_stop_count = state.info.get("reach_stop_count", np.zeros(self._num_envs, dtype=np.int32))
        reach_stop_count = np.where(reach_and_stopped, reach_stop_count + 1, 0)
        state.info["reach_stop_count"] = reach_stop_count
        self._success_truncate = reach_stop_count >= 50

        state.info["metrics"] = {
            "distance_to_target": distance_to_target,
            "reached_fraction": reached_all.astype(np.float32),
            "waypoint_idx": waypoint_idx.astype(np.float32),
        }
        return state

    def _update_truncate(self):
        super()._update_truncate()
        if hasattr(self, '_success_truncate'):
            self._state.truncated = np.logical_or(self._state.truncated, self._success_truncate)

    # ================================================================
    # Reset
    # ================================================================

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg = self._cfg
        num_envs = data.shape[0]

        # 在起始位置周围小范围随机生成
        random_xy = np.random.uniform(low=-self.spawn_range, high=self.spawn_range, size=(num_envs, 2))
        robot_init_xy = self.spawn_center[:2] + random_xy
        terrain_heights = np.full(num_envs, self.spawn_center[2], dtype=np.float32)
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])

        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))

        dof_pos[:, 3:6] = robot_init_xyz

        # 归一化base四元数
        for env_idx in range(num_envs):
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            qn = np.linalg.norm(quat)
            dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = (
                quat / qn if qn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
            )
            if self._robot_arrow_body is not None:
                for dof_start, dof_end in [(self._robot_arrow_dof_start, self._robot_arrow_dof_end),
                                           (self._desired_arrow_dof_start, self._desired_arrow_dof_end)]:
                    aq = dof_pos[env_idx, dof_start + 3:dof_end]
                    aqn = np.linalg.norm(aq)
                    dof_pos[env_idx, dof_start + 3:dof_end] = (
                        aq / aqn if aqn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
                    )

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # 初始航点索引 = 0（从第一个航点开始）
        waypoint_idx = np.zeros(num_envs, dtype=np.float32)
        current_target_xy = self._get_current_target(waypoint_idx)

        # 初始目标朝向：从当前位置指向第一个航点
        target_heading = np.arctan2(
            current_target_xy[:, 1] - robot_init_xy[:, 1],
            current_target_xy[:, 0] - robot_init_xy[:, 0]
        )
        pose_commands = np.column_stack([current_target_xy, target_heading])

        # 更新可视化
        self._update_target_marker(data, pose_commands)

        # 获取根节点状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        robot_heading_val = self._get_heading_from_quat(root_quat)
        position_error = current_target_xy - robot_init_xy
        distance_to_target = np.linalg.norm(position_error, axis=1)
        heading_diff = target_heading - robot_heading_val
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        reached_all = np.zeros(num_envs, dtype=bool)

        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading_val
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2 * np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2 * np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)

        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()

        velocity_commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1)

        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel[:, :2])

        # 构造观测
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)

        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
        stop_ready_flag = np.zeros(num_envs, dtype=np.float32)

        obs = np.concatenate(
            [
                noisy_linvel,                                  # 3
                noisy_gyro,                                    # 3
                projected_gravity,                             # 3
                noisy_joint_angle,                             # 12
                noisy_joint_vel,                               # 12
                last_actions,                                  # 12
                command_normalized,                            # 3
                position_error_normalized,                     # 2
                heading_error_normalized[:, np.newaxis],       # 1
                distance_normalized[:, np.newaxis],            # 1
                reached_flag[:, np.newaxis],                   # 1
                stop_ready_flag[:, np.newaxis],                # 1
            ],
            axis=-1,
        )
        print(f"obs.shape:{obs.shape}")
        assert obs.shape == (num_envs, 54)

        info = {
            "pose_commands": pose_commands,
            "waypoint_idx": waypoint_idx,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached_final": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),
            "initial_distance": distance_to_target.copy(),
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=np.bool_),
            "waypoint_bonus_this_step": np.zeros(num_envs, dtype=np.float32),
        }

        return obs, info
