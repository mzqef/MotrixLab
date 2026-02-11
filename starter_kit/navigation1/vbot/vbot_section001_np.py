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
VBot 平地导航环境 (Navigation1 竞赛 Stage 1)
- 注册为 vbot_navigation_flat，使用 scene.xml（平地）
- 完整奖励函数：使用 RewardConfig.scales 配置
"""

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection001EnvCfg


def generate_repeating_array(num_period, num_reset, period_counter):
    """
    生成重复数组，用于在固定位置中循环选择
    num_period: 位置总数
    num_reset: 需要重置的环境数
    period_counter: 当前计数器
    """
    idx = []
    for i in range(num_reset):
        idx.append((period_counter + i) % num_period)
    return np.array(idx)


@registry.env("vbot_navigation_section001", "np")
class VBotSection001Env(NpEnv):
    """
    VBot在Section001地形上的导航任务
    继承自NpEnv，使用VBotSection001EnvCfg配置
    """
    _cfg: VBotSection001EnvCfg

    def __init__(self, cfg: VBotSection001EnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        # 初始化机器人body和接触
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()

        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")

        # 获取箭头body（用于可视化，不影响物理）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None

        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)

        # 查找target_marker的DOF索引
        self._find_target_marker_dof_indices()

        # 查找箭头的DOF索引
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()

        # 初始化缓存
        self._init_buffer()

        # 检测base_contact传感器是否可用（flat场景没有此传感器）
        self._has_base_contact_sensor = "base_contact" in getattr(
            self._model, 'sensor_names', []
        )

        # 初始位置生成参数
        self.spawn_center = np.array(cfg.init_state.pos, dtype=np.float32)
        # 从配置读取随机化范围（不再硬编码0.1m）
        pos_range = cfg.init_state.pos_randomization_range
        self.spawn_range_low = np.array(pos_range[:2], dtype=np.float32)
        self.spawn_range_high = np.array(pos_range[2:], dtype=np.float32)
        # 圆形平台安全半径（超出则裁剪到圆内）
        self.platform_radius = getattr(cfg, 'platform_radius', 11.0)
        # 目标生成半径（目标集中在平台中心附近）
        self.target_radius = getattr(cfg.commands, 'target_radius', 3.0)

        # 课程学习：生成距离控制
        self.spawn_inner_radius = getattr(cfg, 'spawn_inner_radius', 0.0)
        self.spawn_outer_radius = getattr(cfg, 'spawn_outer_radius', self.platform_radius)

        # 导航统计计数器
        self.navigation_stats_step = 0

    def _init_buffer(self):
        """初始化缓存和参数"""
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)

        # 归一化系数
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )

        # 设置默认关节角度
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle

        self._init_dof_pos[-self._num_action:] = self.default_angles
        self.action_filter_alpha = 0.3

    def _find_target_marker_dof_indices(self):
        """查找target_marker在dof_pos中的索引位置"""
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10

    def _find_arrow_dof_indices(self):
        """查找箭头在dof_pos中的索引位置"""
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36

        arrow_init_height = self._cfg.init_state.pos[2] + 0.5
        arrow_val = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = arrow_val
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = arrow_val

    def _init_contact_geometry(self):
        """初始化接触检测所需的几何体索引"""
        self._init_termination_contact()
        self._init_foot_contact()

    def _init_termination_contact(self):
        """初始化终止接触检测：基座geom与地面geom的碰撞"""
        termination_contact_names = self._cfg.asset.terminate_after_contacts_on

        # 获取所有地面geom
        ground_geoms = []
        ground_prefix = self._cfg.asset.ground_subtree
        for geom_name in self._model.geom_names:
            if geom_name is not None and ground_prefix in geom_name:
                ground_geoms.append(self._model.get_geom_index(geom_name))

        # 构建碰撞对：每个基座geom × 每个地面geom
        termination_contact_list = []
        for base_geom_name in termination_contact_names:
            try:
                base_geom_idx = self._model.get_geom_index(base_geom_name)
                for ground_idx in ground_geoms:
                    termination_contact_list.append([base_geom_idx, ground_idx])
            except Exception as e:
                print(f"[Warning] 无法找到基座geom '{base_geom_name}': {e}")

        if len(termination_contact_list) > 0:
            self.termination_contact = np.array(termination_contact_list, dtype=np.uint32)
            self.num_termination_check = len(termination_contact_list)
            n_base = len(termination_contact_names)
            n_ground = len(ground_geoms)
            print(f"[Info] 终止接触检测: {n_base}基座geom × {n_ground}地面geom = {self.num_termination_check}对")
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("[Warning] 未找到任何终止接触geom，基座接触检测将被禁用！")

    def _init_foot_contact(self):
        self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
        self.num_foot_check = 4

    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)

    def _extract_root_state(self, data):
        """从self._body中提取根节点状态"""
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

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # 保存上一步的关节速度（用于计算加速度）
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
        state.data.actuator_ctrls = self._compute_torques(state.info["filtered_actions"], state.data)

        return state

    def _compute_torques(self, actions, data):
        """计算PD控制力矩（VBot使用motor执行器，需要力矩控制）"""
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled

        current_pos = self.get_dof_pos(data)
        current_vel = self.get_dof_vel(data)

        # PD控制器：tau = kp * (target - current) - kv * vel
        kp = 80.0
        kv = 6.0

        pos_error = target_pos - current_pos
        torques = kp * pos_error - kv * current_vel

        # 限制力矩范围（与XML中的forcerange一致）
        # hip/thigh: ±17 N·m, calf: ±34 N·m
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)
        torques = np.clip(torques, -torque_limits, torque_limits)

        return torques

    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        """计算机器人坐标系中的重力向量"""
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)

    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        """从四元数计算yaw角（朝向）"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """更新目标位置标记的位置和朝向"""
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

    def _update_heading_arrows(
        self, data: mtx.SceneData, robot_pos: np.ndarray,
        desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray,
    ):
        """更新箭头位置（使用DOF控制freejoint，不影响物理）"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return

        num_envs = data.shape[0]
        arrow_offset = 0.5
        all_dof_pos = data.dof_pos.copy()

        for env_idx in range(num_envs):
            arrow_height = robot_pos[env_idx, 2] + arrow_offset

            # 当前运动方向箭头
            cur_v = base_lin_vel_xy[env_idx]
            if np.linalg.norm(cur_v) > 1e-3:
                cur_yaw = np.arctan2(cur_v[1], cur_v[0])
            else:
                cur_yaw = 0.0
            robot_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            quat_norm = np.linalg.norm(robot_arrow_quat)
            if quat_norm > 1e-6:
                robot_arrow_quat = robot_arrow_quat / quat_norm
            else:
                robot_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                robot_arrow_pos, robot_arrow_quat
            ])

            # 期望运动方向箭头
            des_v = desired_vel_xy[env_idx]
            if np.linalg.norm(des_v) > 1e-3:
                des_yaw = np.arctan2(des_v[1], des_v[0])
            else:
                des_yaw = 0.0
            desired_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            quat_norm = np.linalg.norm(desired_arrow_quat)
            if quat_norm > 1e-6:
                desired_arrow_quat = desired_arrow_quat / quat_norm
            else:
                desired_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                desired_arrow_pos, desired_arrow_quat
            ])

        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)

    def _euler_to_quat(self, roll, pitch, yaw):
        """欧拉角转四元数 [qx, qy, qz, qw] - Motrix格式"""
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

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """更新环境状态，计算观测、奖励和终止条件"""
        data = state.data
        cfg = self._cfg

        # 获取基础状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        # 传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # 导航目标
        pose_commands = state.info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        # 计算位置误差
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        # 计算朝向误差
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        # 达到判定 (Round4: 0.3→0.5m — wider zone gives more arrival_bonus signals during training)
        position_threshold = 0.5
        reached_all = distance_to_target < position_threshold

        # 存储robot_xy供奖励函数使用（边界惩罚需要）
        state.info["robot_xy"] = robot_position

        # 计算期望速度命令（P控制器）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)

        # 角速度命令：跟踪运动方向
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        h2m = heading_to_movement
        heading_to_movement = np.where(h2m > np.pi, h2m - 2 * np.pi, h2m)
        h2m = heading_to_movement
        heading_to_movement = np.where(h2m < -np.pi, h2m + 2 * np.pi, h2m)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(
            np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate
        )
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)

        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()

        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )

        # 归一化观测
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]

        # 任务相关观测
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)

        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)

        obs = np.concatenate(
            [
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
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 54)

        # 更新目标标记和箭头
        self._update_target_marker(data, pose_commands)
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, position_error, base_lin_vel_xy)

        # 计算终止条件（必须在奖励之前，因为终止惩罚需要此结果）
        terminated_state = self._compute_terminated(state, projected_gravity, joint_vel)
        terminated = terminated_state.terminated

        # 计算奖励
        reward = self._compute_reward(data, state.info, velocity_commands,
                                       base_lin_vel, gyro, projected_gravity,
                                       joint_vel, distance_to_target, heading_diff,
                                       position_error, reached_all, terminated,
                                       robot_heading)

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        # ========== Round2: 达到目标后成功截断 ==========
        # 当机器人到达目标且速度足够低时，连续计数；超过阈值后截断episode
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        reach_and_stopped = np.logical_and(reached_all, speed_xy < 0.15)
        reach_stop_count = state.info.get("reach_stop_count", np.zeros(self._num_envs, dtype=np.int32))
        reach_stop_count = np.where(reach_and_stopped, reach_stop_count + 1, 0)
        state.info["reach_stop_count"] = reach_stop_count
        # 连续50步（0.5秒）稳定停在目标处 → 标记成功截断
        self._success_truncate = reach_stop_count >= 50
        # Round9 REVERTED: 预算耗尽强制截断实验失败（50步/150步都过于激进，
        # 阻碍早期学习,R9/R9b低于R7)。peak-then-decline根源是policy std collapse,
        # 不是后预算exploitation。让训练跑完100M步后选最佳checkpoint。

        # 记录评估指标用于TensorBoard和Pipeline评估
        state.info["metrics"] = {
            "distance_to_target": distance_to_target,
            "reached_fraction": reached_all.astype(np.float32),
        }

        return state

    def _update_truncate(self):
        """Override: 除max_episode_steps截断外，还包括成功到达截断"""
        super()._update_truncate()
        if hasattr(self, '_success_truncate'):
            self._state.truncated = np.logical_or(self._state.truncated, self._success_truncate)

    def _compute_terminated(
        self, state: NpEnvState,
        projected_gravity: np.ndarray = None,
        joint_vel: np.ndarray = None,
    ) -> NpEnvState:
        """
        终止条件检测：
        1. 基座接触地面（翻倒）
        2. 侧翻检测（倾斜角 > 75°）
        3. 关节速度异常（> max_dof_vel 或 NaN/Inf）
        """
        data = state.data

        # 1. 基座接触地面终止（使用传感器，仅在传感器存在时读取）
        if self._has_base_contact_sensor:
            try:
                base_contact_value = self._model.get_sensor_value("base_contact", data)
                if base_contact_value.ndim == 0:
                    base_contact = np.array([base_contact_value > 0.01], dtype=bool)
                elif base_contact_value.shape[0] != self._num_envs:
                    base_contact = np.full(self._num_envs, base_contact_value.flatten()[0] > 0.01, dtype=bool)
                else:
                    base_contact = (base_contact_value > 0.01).flatten()[:self._num_envs]
            except BaseException:
                base_contact = np.zeros(self._num_envs, dtype=bool)
        else:
            base_contact = np.zeros(self._num_envs, dtype=bool)

        terminated = base_contact.copy()

        # 2. 侧翻检测：projected_gravity的xy分量过大（倾斜 > 75°）
        if projected_gravity is not None:
            gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
            gz = projected_gravity[:, 2]
            tilt_angle = np.arctan2(gxy, np.abs(gz))
            side_flip = tilt_angle > np.deg2rad(75)
            terminated = np.logical_or(terminated, side_flip)

        # 3. 关节速度异常检测
        if joint_vel is not None:
            vel_max = np.abs(joint_vel).max(axis=1)
            vel_overflow = vel_max > self._cfg.max_dof_vel
            vel_extreme = np.isnan(joint_vel).any(axis=1) | np.isinf(joint_vel).any(axis=1)
            terminated = np.logical_or(terminated, vel_overflow | vel_extreme)

        return state.replace(terminated=terminated)

    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray,
                         base_lin_vel: np.ndarray, gyro: np.ndarray, projected_gravity: np.ndarray,
                         joint_vel: np.ndarray, distance_to_target: np.ndarray, heading_diff: np.ndarray,
                         position_error: np.ndarray, reached_all: np.ndarray,
                         terminated: np.ndarray, robot_heading: np.ndarray = None) -> np.ndarray:
        """
        导航任务奖励计算，使用 RewardConfig.scales 配置权重

        奖励组成：
        - 导航跟踪（position_tracking, fine_position_tracking, heading_tracking, forward_velocity）
        - 距离接近奖励（approach_reward）
        - 到达奖励（arrival_bonus, stop_bonus）
        - 稳定性惩罚（orientation, lin_vel_z, ang_vel_xy, torques, dof_vel, dof_acc, action_rate）
        - 终止惩罚（termination）
        """
        scales = self._cfg.reward_config.scales

        # ========== 终止惩罚（使用已计算的 terminated 标志） ==========
        term_scale = scales.get("termination", -50.0)
        termination_penalty = np.where(terminated, term_scale, 0.0)

        # ========== 导航跟踪奖励（未到达时激活） ==========

        # 位置跟踪：exp(-distance / sigma) — sigma=5.0 解决远距离梯度死区
        position_tracking = np.exp(-distance_to_target / 5.0)

        # 精细位置跟踪：在距离 < 2.5m 时激活, sigma=0.5
        # Round3: 扩大范围 1.5→2.5m, sigma 0.3→0.5, 消除1.5-2.5m的梯度死区
        # 在 1.75m (当前平均距离) 处提供实质性奖励:
        #   exp(-1.75/0.5)*8 = 0.24/step, 而非之前 = 0.0
        fine_position_tracking = np.where(
            distance_to_target < 2.5,
            np.exp(-distance_to_target / 0.5),
            0.0
        )

        # 朝向跟踪：奖励机器人面朝目标方向（而非随机target_heading）
        # 使用 facing_error = target_bearing - robot_heading（面向目标位置）
        target_bearing = np.arctan2(position_error[:, 1], position_error[:, 0])
        facing_error = target_bearing - robot_heading
        # 归一化到 [-pi, pi]
        facing_error = np.where(facing_error > np.pi, facing_error - 2 * np.pi, facing_error)
        facing_error = np.where(facing_error < -np.pi, facing_error + 2 * np.pi, facing_error)
        # 到达后不需要面向目标（已经停在那里了）
        heading_tracking = np.where(reached_all, 1.0, np.exp(-np.abs(facing_error) / 0.5))

        # 前进速度奖励：速度在朝目标方向的投影
        direction_to_target = position_error / (np.linalg.norm(position_error, axis=1, keepdims=True) + 1e-8)
        forward_velocity = np.sum(base_lin_vel[:, :2] * direction_to_target, axis=1)
        forward_velocity = np.clip(forward_velocity, -1.0, 0.6)  # 限速0.6m/s，超速无额外奖励

        # 距离接近奖励：激励靠近目标
        # Round4: 改用逐步距离变化（连续梯度），替代 min_distance 一次性记录
        # 原 min_distance 方法的问题：进度停滞后信号消失，策略失去接近激励
        last_distance = info.get("last_distance", distance_to_target.copy())
        distance_delta = last_distance - distance_to_target  # 正值 = 靠近
        info["last_distance"] = distance_to_target.copy()
        raw_approach = distance_delta * scales.get("approach_scale", 4.0)
        # Round5 FIX: 移除近目标处的 retreat 免罚条款
        # 原来 d<1.5m 时后退无惩罚，导致"hovering"行为
        # Round6 FIX: 移除retreat惩罚 (clip下界0而非-0.5)
        # step-delta退后惩罚会惩罚早期探索，原始min_distance方法从未惩罚退后
        # Round8 REVERTED: 分段clip实验失败 — approach_scale=40.46 × -0.5 = -20.2/step
        # 使机器人害怕触达目标 (触达后退后惩罚太大), R8 reached% 14% vs R7 33%
        approach_reward = np.clip(raw_approach, 0.0, 1.0)

        # 线性距离递减奖励（新增）：提供全局梯度信号
        initial_distance = info.get("initial_distance", distance_to_target)
        distance_progress = np.clip(1.0 - distance_to_target / (initial_distance + 1e-8), -0.5, 1.0)

        # 存活奖励：每步固定小奖励，始终激活（移除 ever_reached 条件）
        # Round5 FIX: 原来 alive_bonus=0 after reaching 导致"touch and die"循环
        # 机器人到达后失去存活激励，会故意摔倒以重置episode获取更多奖励
        # 现在持续鼓励存活，由 success_truncation（50步停止）控制episode结束
        alive_bonus = np.ones(self._num_envs, dtype=np.float32)

        # ========== 时间压力：已移除 ==========
        # R11 FIX: 移除time_decay — Budget Audit发现它创造"早死"激励
        # 原因：time_decay使后期步骤价值降低(步0=1.0, 步500=0.75, 步1000=0.5)
        # 两个500步episode(avg decay=0.94)比一个1000步episode(avg decay=0.75)多17%奖励
        # PPO学会激进approach→摔倒→重启 以获取更高per-step奖励
        # 证据：R10 ep_len从975→528, per_step从2.49→2.99, reached从35%→14%
        steps = info.get("steps", np.zeros(self._num_envs, dtype=np.float32)).astype(np.float32)
        max_steps = float(self._cfg.max_episode_steps)
        time_decay = np.ones(self._num_envs, dtype=np.float32)  # 常数1.0，无衰减

        # ========== 到达后的停止奖励 ==========

        # 首次到达的一次性奖励
        info["ever_reached"] = info.get("ever_reached", np.zeros(self._num_envs, dtype=bool))
        first_time_reach = np.logical_and(reached_all, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached_all)
        arrival_bonus = np.where(first_time_reach, scales.get("arrival_bonus", 15.0), 0.0)

        # Round7 FIX: 记录首次到达的步数，限制stop_bonus最多持续50步
        # 原因：stop_bonus为21.2/step，600步farming=12726远超导航奖励611
        # 这导致robot学会到达后站立farming，最终变得过于保守(peak-then-decline)
        info["first_reach_step"] = info.get("first_reach_step", np.full(self._num_envs, -1.0, dtype=np.float32))
        info["first_reach_step"] = np.where(
            np.logical_and(first_time_reach, info["first_reach_step"] < 0),
            steps, info["first_reach_step"]
        )
        # 首次到达后经过的步数
        steps_since_reach = np.where(info["first_reach_step"] >= 0, steps - info["first_reach_step"], 0.0)
        stop_budget_remaining = np.clip(50.0 - steps_since_reach, 0.0, 50.0)  # 最多50步stop奖励
        stop_eligible = stop_budget_remaining > 0

        # 停止奖励：到达后鼓励静止（限50步）
        # Round5 FIX: 添加速度门控，只有真正在减速（<0.3m/s）才给stop_bonus
        # Round7 FIX: 50步预算上限，防止stop farming dominance
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        genuinely_slow = np.logical_and(reached_all, speed_xy < 0.3)
        genuinely_slow = np.logical_and(genuinely_slow, stop_eligible)  # Round7: 预算内才给奖励
        stop_base = scales.get("stop_scale", 2.0) * (
            0.8 * np.exp(-(speed_xy / 0.2) ** 2) + 1.2 * np.exp(-(np.abs(gyro[:, 2]) / 0.1) ** 4)
        )
        zero_ang_mask = np.abs(gyro[:, 2]) < 0.05
        zero_ang_bonus = np.where(
            np.logical_and(genuinely_slow, zero_ang_mask), scales.get("zero_ang_bonus", 6.0), 0.0
        )
        stop_bonus = np.where(genuinely_slow, stop_base + zero_ang_bonus, 0.0)

        # ========== 稳定性惩罚（始终激活） ==========

        # 姿态惩罚
        orientation_penalty = np.square(projected_gravity[:, 0]) + np.square(projected_gravity[:, 1])

        # Z轴线速度惩罚
        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])

        # XY轴角速度惩罚
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1)

        # 力矩惩罚
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=1)

        # 关节速度惩罚
        dof_vel_penalty = np.sum(np.square(joint_vel), axis=1)

        # 关节加速度惩罚
        last_dof_vel = info.get("last_dof_vel", np.zeros_like(joint_vel))
        dof_acc = joint_vel - last_dof_vel
        dof_acc_penalty = np.sum(np.square(dof_acc), axis=1)

        # 动作变化率惩罚
        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        # ========== 新增Phase5奖励/惩罚 ==========

        # Round5 FIX: 距离-速度耦合惩罚（替代原来仅d<0.5m的窄范围惩罚）
        # 期望速度与距离成正比：远处快跑，近处慢行
        # desired_speed: 0.6m/s at d>=1.2m, 线性递减到 0.05m/s at d=0.1m
        desired_speed = np.clip(distance_to_target * 0.5, 0.05, 0.6)
        speed_excess = np.maximum(speed_xy - desired_speed, 0.0)
        near_target_speed_penalty = scales.get("near_target_speed", -0.5) * speed_excess ** 2

        # 内围栏一次性奖励：首次进入距离<0.75m区域
        inner_fence_dist = 0.75
        in_inner_fence = distance_to_target < inner_fence_dist
        info["ever_inner_fence"] = info.get("ever_inner_fence", np.zeros(self._num_envs, dtype=bool))
        first_inner_fence = np.logical_and(in_inner_fence, ~info["ever_inner_fence"])
        info["ever_inner_fence"] = np.logical_or(info["ever_inner_fence"], in_inner_fence)
        inner_fence_bonus = np.where(first_inner_fence, scales.get("inner_fence_bonus", 0.0), 0.0)

        # 边界惩罚：距平台边缘<1m时惩罚（防掉落）
        robot_xy = info.get("robot_xy", np.zeros((self._num_envs, 2), dtype=np.float32))
        dist_from_center = np.linalg.norm(robot_xy, axis=1)
        boundary_margin = 1.0  # 距边缘1m内开始惩罚
        boundary_violation = np.maximum(dist_from_center - (self.platform_radius - boundary_margin), 0.0)
        boundary_penalty = scales.get("boundary_penalty", 0.0) * boundary_violation

        # ========== Round8: 离开中心区惩罚 ==========
        # 当机器人在目标区域内(d<0.5m)且向外移动时，立即惩罚
        # 填补架构缺口：reached分支无approach信号，not-reached分支approach无后退惩罚
        prev_distance = info.get("prev_distance", distance_to_target.copy())
        delta_d = distance_to_target - prev_distance  # 正值 = 远离目标
        info["prev_distance"] = distance_to_target.copy()
        is_departing_from_center = np.logical_and(reached_all, delta_d > 0.01)
        departure_penalty = np.where(
            is_departing_from_center,
            scales.get("departure_penalty", -5.0) * delta_d,
            0.0
        )

        # ========== 综合奖励（到达前/后分支） ==========
        # R11 FIX: fine_position_tracking仅在ever_reached后解锁
        # Budget Audit: fine_position_tracking (sigma=0.5) 在d≈0.52m时给4.24/step
        # 创造"悬停在d=0.5m边界外"的激励 — 几乎等于reaching的奖励(97%)
        # 修复：必须先触达(d<0.5m)才能解锁fine_position_tracking
        # position_tracking (sigma=5.0) 保留在所有分支 — 它提供平缓的全局梯度
        # (d=0.7m和d=2m的差异仅0.35/step，不足以创造hovering激励)
        fine_tracking_gated = np.where(
            info["ever_reached"],
            scales.get("fine_position_tracking", 2.0) * fine_position_tracking,
            0.0
        )

        # 惩罚项公共部分（到达前后都生效）
        penalties = (
            scales.get("orientation", -0.05) * orientation_penalty
            + scales.get("lin_vel_z", -0.5) * lin_vel_z_penalty
            + scales.get("ang_vel_xy", -0.05) * ang_vel_xy_penalty
            + scales.get("torques", -1e-5) * torque_penalty
            + scales.get("dof_vel", -5e-5) * dof_vel_penalty
            + scales.get("dof_acc", -2.5e-7) * dof_acc_penalty
            + scales.get("action_rate", -0.01) * action_rate_penalty
            + termination_penalty
            + near_target_speed_penalty
            + boundary_penalty
            + departure_penalty
        )

        reward = np.where(
            reached_all,
            # 到达后：停止奖励 + 到达奖励 + 位置跟踪 + fine(已解锁) + 内围栏 + 惩罚
            stop_bonus + arrival_bonus
            + scales.get("position_tracking", 1.5) * position_tracking
            + fine_tracking_gated
            + inner_fence_bonus + penalties,
            # 未到达：导航 + position_tracking(全局) + fine(仅ever_reached后) + 接近 + 内围栏 + 惩罚
            # R11: 移除time_decay(=1.0), fine_position_tracking门控behind ever_reached
            (
                scales.get("position_tracking", 1.5) * position_tracking
                + scales.get("heading_tracking", 0.8) * heading_tracking
                + scales.get("forward_velocity", 1.5) * forward_velocity
                + scales.get("distance_progress", 2.0) * distance_progress
                + scales.get("alive_bonus", 0.5) * alive_bonus
                + approach_reward
            ) + fine_tracking_gated + inner_fence_bonus + penalties
        )

        # 记录各奖励分量用于TensorBoard可视化
        info["Reward"] = {
            "position_tracking": scales.get("position_tracking", 2.0) * position_tracking,
            "fine_position_tracking": scales.get("fine_position_tracking", 2.0) * fine_position_tracking,
            "fine_tracking_gated": fine_tracking_gated,  # R11: ever_reached门控后的fine值
            "heading_tracking": scales.get("heading_tracking", 1.0) * heading_tracking,
            "forward_velocity": scales.get("forward_velocity", 0.5) * forward_velocity,
            "approach_reward": approach_reward,
            "distance_progress": scales.get("distance_progress", 2.0) * distance_progress,
            "alive_bonus": scales.get("alive_bonus", 0.5) * alive_bonus,
            "arrival_bonus": arrival_bonus,
            "inner_fence_bonus": inner_fence_bonus,
            "stop_bonus": stop_bonus,
            "near_target_speed": near_target_speed_penalty,
            "boundary_penalty": boundary_penalty,
            "departure_penalty": departure_penalty,
            "penalties": penalties,
            "termination": termination_penalty,
        }

        return reward

    def _random_point_in_circle(self, n: int, radius: float) -> np.ndarray:
        """在半径radius的圆内均匀采样n个点 (返回 [n, 2])"""
        # 均匀分布在圆内：r = sqrt(U) * R, theta = U * 2pi
        r = np.sqrt(np.random.uniform(0, 1, size=n)) * radius
        theta = np.random.uniform(0, 2 * np.pi, size=n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=1).astype(np.float32)

    def _random_point_in_annulus(self, n: int, inner_r: float, outer_r: float) -> np.ndarray:
        """在环形区域 [inner_r, outer_r] 内均匀采样n个点 (返回 [n, 2])

        用于课程学习：控制机器人生成距离。
        - Stage 1 (Easy):   inner=2, outer=5
        - Stage 2 (Medium): inner=5, outer=8
        - Stage 3 (Hard):   inner=8, outer=11
        """
        if inner_r <= 0:
            return self._random_point_in_circle(n, outer_r)
        # 环形均匀采样：r = sqrt(U * (R2^2 - R1^2) + R1^2)
        u = np.random.uniform(0, 1, size=n)
        r = np.sqrt(u * (outer_r**2 - inner_r**2) + inner_r**2)
        theta = np.random.uniform(0, 2 * np.pi, size=n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=1).astype(np.float32)

    def _clip_to_circle(self, xy: np.ndarray, radius: float) -> np.ndarray:
        """将xy坐标裁剪到半径radius的圆内"""
        dist = np.linalg.norm(xy, axis=1, keepdims=True)
        scale = np.where(dist > radius, radius / (dist + 1e-8), 1.0)
        return xy * scale

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg: VBotSection001EnvCfg = self._cfg
        num_envs = data.shape[0]

        # 在环形区域内随机生成起始位置（课程学习控制距离）
        robot_init_xy = self._random_point_in_annulus(
            num_envs, self.spawn_inner_radius, self.spawn_outer_radius
        )
        # 加上平台中心偏移（通常为0,0）
        robot_init_xy += self.spawn_center[:2]
        terrain_heights = np.full(num_envs, self.spawn_center[2], dtype=np.float32)

        robot_init_pos = robot_init_xy
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])

        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))

        # 设置 base 的 XYZ位置（DOF 3-5）
        dof_pos[:, 3:6] = robot_init_xyz

        # 随机初始朝向（全方向，不总是面向+Y）
        random_yaw = np.random.uniform(-np.pi, np.pi, size=num_envs)
        for i in range(num_envs):
            quat = self._euler_to_quat(0, 0, random_yaw[i])
            dof_pos[i, self._base_quat_start:self._base_quat_end] = quat / (np.linalg.norm(quat) + 1e-8)

        # 目标位置生成
        target_mode = getattr(cfg.commands, 'target_mode', 'relative')
        min_dist = getattr(cfg.commands, 'min_distance', 3.0)

        if target_mode == "absolute":
            # 绝对坐标模式：目标在平台中心附近小圆内随机（竞赛目标区域）
            target_positions = self._random_point_in_circle(num_envs, self.target_radius)
            target_positions += self.spawn_center[:2]
            # 保证起始与目标之间的最小距离
            for i in range(num_envs):
                attempts = 0
                while np.linalg.norm(target_positions[i] - robot_init_pos[i]) < min_dist and attempts < 20:
                    tp = self._random_point_in_circle(1, self.platform_radius)[0]
                    tp += self.spawn_center[:2]
                    target_positions[i] = tp
                    attempts += 1
        else:
            # 旧模式：相对偏移
            target_offset = np.random.uniform(
                low=cfg.commands.pose_command_range[:2],
                high=cfg.commands.pose_command_range[3:5],
                size=(num_envs, 2)
            )
            target_positions = robot_init_pos + target_offset
            # 裁剪到平台内
            target_positions = self._clip_to_circle(
                target_positions - self.spawn_center[:2], self.platform_radius
            ) + self.spawn_center[:2]

        target_headings = np.random.uniform(
            low=cfg.commands.pose_command_range[2],
            high=cfg.commands.pose_command_range[5],
            size=(num_envs, 1)
        )

        pose_commands = np.concatenate([target_positions, target_headings], axis=1)

        # 归一化箭头的四元数（base四元数已在上面随机yaw时设置）
        for env_idx in range(num_envs):
            # 归一化箭头的四元数
            if self._robot_arrow_body is not None:
                ra_s = self._robot_arrow_dof_start + 3
                ra_e = self._robot_arrow_dof_end
                robot_arrow_quat = dof_pos[env_idx, ra_s:ra_e]
                quat_norm = np.linalg.norm(robot_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, ra_s:ra_e] = robot_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, ra_s:ra_e] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

                da_s = self._desired_arrow_dof_start + 3
                da_e = self._desired_arrow_dof_end
                desired_arrow_quat = dof_pos[env_idx, da_s:da_e]
                quat_norm = np.linalg.norm(desired_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, da_s:da_e] = desired_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, da_s:da_e] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # 更新目标位置标记
        self._update_target_marker(data, pose_commands)

        # 获取根节点状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # 关节状态
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        # 传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # 计算速度命令
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        position_threshold = 0.5  # Round4: aligned with update_state
        reached_all = distance_to_target < position_threshold

        # 计算期望速度
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)

        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, position_error, base_lin_vel_xy)

        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        # 角速度跟踪运动方向
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        h2m = heading_to_movement
        heading_to_movement = np.where(h2m > np.pi, h2m - 2 * np.pi, h2m)
        h2m = heading_to_movement
        heading_to_movement = np.where(h2m < -np.pi, h2m + 2 * np.pi, h2m)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)

        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)

        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()

        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )

        # 归一化观测
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)

        # 任务相关观测
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)

        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)

        obs = np.concatenate(
            [
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
            ],
            axis=-1,
        )
        assert obs.shape == (num_envs, 54)

        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "last_distance": distance_to_target.copy(),  # Round4: step-delta approach_reward
            "initial_distance": distance_to_target.copy(),  # 用于distance_progress归一化
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=np.bool_),
        }

        return obs, info
