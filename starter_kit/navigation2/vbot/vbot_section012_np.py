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
VBot Section012 有序多航点全收集导航环境

竞赛规则 (Section 2 = 60分):
  +10: 通过波浪地形到达楼梯
  +5:  从楼梯到达河床/吊桥
  +10: 经过吊桥途径拜年红包到达楼梯口
  +5:  从楼梯口下来到达丙午大吉平台
  +5:  庆祝动作
  +3×5=15: 河床石头上贺礼红包
  +5×2=10: 桥底下拜年红包

策略: 有序航点全收集 (右侧优先)
  1) 右侧河床收集5个石头红包 (固定顺序)
  2) 桥下收集2个拜年红包
  3) 远端上桥收集桥上拜年红包
  4) 原路返回下桥
  5) 到达终点平台
  6) 庆祝动作 (~10次，可配置)

航点由 cfg.OrderedRoute 定义: 每个航点含坐标、类型
(reward/virtual/goal)、到达半径、可选z约束、和奖金配置。
机器人严格按顺序依次到达。wp_idx = 已完成航点数。
观测: 69维 (与其他section对齐, 含trunk_acc + actuator_torques)
"""

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection012EnvCfg, VBotSection012StairsEnvCfg, TerrainScaleHelper

# ============================================================
# 庆祝状态机常量 (v58: X轴行走 + 蹲坐庆祝, 与section011一致)
# ============================================================
CELEB_IDLE = 0        # 未开始庆祝
CELEB_WALKING = 1     # 走向X轴端点
CELEB_SITTING = 2     # 蹲坐中
CELEB_DONE = 3        # 庆祝完成

# 机器人躯干矩形足印半尺寸 (m) — 来自vbot.xml collision_middle_box
# 用于 footprint-contains 检测: 区域中心点是否落在机器人矩形投影内
ROBOT_HALF_X = 0.25  # 前后半长
ROBOT_HALF_Y = 0.15  # 左右半宽


@registry.env("vbot_navigation_section012", "np")
class VBotSection012Env(NpEnv):
    """
    VBot Section02 有序多航点全收集导航 + 跳跃庆祝
    地形: 入口平台 → 楼梯 → 拱桥/球障碍 → 楼梯 → 出口平台
    策略: 右侧优先收集石头红包 → 桥下红包 → 远端上桥收集桥上红包 → 原路返回 → 终点 → 庆祝
    观测: 69维 (通用对齐)
    """
    _cfg: VBotSection012EnvCfg

    def __init__(self, cfg: VBotSection012EnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()

        self._target_marker_body = self._model.get_body("target_marker")
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None

        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # 69维观测: 与section011完全对齐 (54 base + 4 foot_contact + 1 base_height
        #   + 1 celeb_progress + 3 trunk_acc + 12 actuator_torques - 6 removed old fields)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(69,), dtype=np.float32)

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)

        self._find_target_marker_dof_indices()
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()

        self._init_buffer()

        self.spawn_center = np.array(cfg.init_state.pos, dtype=np.float32)
        pr = cfg.init_state.pos_randomization_range
        self.spawn_x_range = (pr[0], pr[2])
        self.spawn_y_range = (pr[1], pr[3])

        self.navigation_stats_step = 0

        # 初始化有序路线导航系统
        self._init_ordered_route(cfg)

    # ============================================================
    # 初始化方法
    # ============================================================

    def _init_buffer(self):
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

    def _init_contact_geometry(self):
        self._init_foot_contact()

    def _init_foot_contact(self):
        self.foot_sensor_names = ["FR_foot_contact", "FL_foot_contact", "RR_foot_contact", "RL_foot_contact"]
        self.num_foot_check = 4
        self._gait_phase_counter = None
        # 关节扭矩限制 (与section011一致)
        self.torque_limits = np.array([17, 17, 34, 17, 17, 34, 17, 17, 34, 17, 17, 34], dtype=np.float32)

    def _init_ordered_route(self, cfg):
        """初始化有序航点导航系统 — 通用实现，路线由 cfg.ordered_route 定义"""
        route = cfg.ordered_route
        wps = route.waypoints

        # 航点属性 (预计算numpy数组，用于向量化操作)
        self.num_route_waypoints = len(wps)
        self.wp_xy = np.array([w.xy for w in wps], dtype=np.float32)          # [N, 2]
        self.wp_radius = np.array([w.radius for w in wps], dtype=np.float32)  # [N]
        self.wp_z_min = np.array([w.z_min for w in wps], dtype=np.float32)    # [N]
        self.wp_z_max = np.array([w.z_max for w in wps], dtype=np.float32)    # [N]
        self.wp_labels = [w.label for w in wps]
        self.wp_kinds = [w.kind for w in wps]
        # reward航点使用footprint-contains检测 (radius<=0时启用)
        self.wp_use_footprint = np.array([w.radius <= 0.0 for w in wps], dtype=bool)  # [N]

        # 预计算每个航点的奖金值 (从reward_scales查找)
        scales = cfg.reward_config.scales
        self.wp_bonus = np.array([
            scales.get(w.bonus_key, w.bonus_default) if w.bonus_key else w.bonus_default
            for w in wps
        ], dtype=np.float32)  # [N]

        # 庆祝配置 (v58: X轴行走 + 蹲坐, 与section011一致)
        # 庆祝X轴目标: goal航点X + celeb_x_offset
        goal_xy = self.wp_xy[-1]
        self.celeb_x_target = np.array([goal_xy[0] + route.celeb_x_offset, goal_xy[1]], dtype=np.float32)
        self.celeb_walk_radius = route.celeb_walk_radius
        self.celeb_sit_z = route.celeb_sit_z
        self.celeb_sit_steps = route.celeb_sit_steps

        # 终点航点 (最后一个goal类型航点的坐标，用作fallback目标)
        self.goal_xy = self.wp_xy[-1].copy()

        labels = ", ".join(f"{w.label}({w.kind})" for w in wps)
        print(f"[Info] 有序路线导航: {self.num_route_waypoints}航点, "
              f"庆祝: X轴行走+蹲坐")
        print(f"[Info] 路线: {labels}")

    # ============================================================
    # 传感器 & 物理辅助
    # ============================================================

    def _get_foot_contact_forces(self, data: mtx.SceneData) -> np.ndarray:
        forces = []
        for sensor_name in self.foot_sensor_names:
            try:
                force = self._model.get_sensor_value(sensor_name, data)
                forces.append(force)
            except Exception:
                forces.append(np.zeros((data.shape[0], 3), dtype=np.float32))
        return np.stack(forces, axis=1)

    def _get_trunk_acc(self, data: mtx.SceneData) -> np.ndarray:
        """读取trunk加速度计传感器 [n, 3] (m/s²)"""
        return self._model.get_sensor_value("trunk_acc", data)

    def _get_actuator_torques(self, data: mtx.SceneData) -> np.ndarray:
        """获取raw PD扭矩需求 [n, 12] (Nm) — 未被forcerange clip"""
        if hasattr(self, '_raw_torques'):
            return self._raw_torques
        return data.actuator_ctrls  # fallback

    def _compute_swing_contact_penalty(self, data: mtx.SceneData, joint_vel: np.ndarray) -> np.ndarray:
        foot_forces = self._get_foot_contact_forces(data)
        force_magnitudes = np.linalg.norm(foot_forces, axis=2)
        calf_indices = [2, 5, 8, 11]
        foot_vel = np.abs(joint_vel[:, calf_indices])
        has_contact = force_magnitudes > 0.5
        has_high_vel = foot_vel > 2.0
        swing_contact = np.logical_and(has_contact, has_high_vel).astype(np.float32)
        penalty = np.sum(swing_contact * np.square(foot_vel) / 10.0, axis=1)
        return penalty

    # ============================================================
    # 步态质量奖励 (与section011一致)
    # ============================================================

    def _compute_gait_rewards(self, data: mtx.SceneData, info: dict,
                               base_lin_vel: np.ndarray, robot_heading: np.ndarray,
                               projected_gravity: np.ndarray) -> dict:
        foot_forces = self._get_foot_contact_forces(data)
        force_mag = np.linalg.norm(foot_forces, axis=2)
        in_contact = (force_mag > 0.5).astype(np.float32)

        # Trot gait: diagonal pair alternation
        pair_A = in_contact[:, 0] + in_contact[:, 3]  # FR + RL
        pair_B = in_contact[:, 1] + in_contact[:, 2]  # FL + RR
        trot_score = np.abs(pair_A - pair_B)
        either_pair_full = np.maximum(pair_A, pair_B)
        feet_contact_reward = trot_score * 0.5 + (either_pair_full - 1.0) * 0.25
        feet_contact_reward = np.clip(feet_contact_reward, 0.0, 1.5)

        # Stance ratio: ~2 feet on ground
        total_contacts = np.sum(in_contact, axis=1)
        stance_raw = 1.0 - np.abs(total_contacts - 2.0) * 0.33
        stance_reward = np.clip(stance_raw, 0.0, 1.0)

        # Lateral velocity penalty
        heading_vec = np.stack([np.cos(robot_heading), np.sin(robot_heading)], axis=1)
        lateral_vec = np.stack([-np.sin(robot_heading), np.cos(robot_heading)], axis=1)
        lateral_vel = np.abs(np.sum(base_lin_vel[:, :2] * lateral_vec, axis=1))
        lateral_penalty = np.square(lateral_vel)

        # Body balance
        upright_raw = np.clip(-projected_gravity[:, 2], 0.0, 1.0)
        forward_body_vel = np.sum(base_lin_vel[:, :2] * heading_vec, axis=1)
        body_balance_reward = upright_raw * np.clip(forward_body_vel, 0.0, 0.8) / 0.8

        return {
            "feet_contact_reward": feet_contact_reward,
            "stance_reward": stance_reward,
            "lateral_penalty": lateral_penalty,
            "body_balance_reward": body_balance_reward,
        }

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

    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)

    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _footprint_contains_point(center_xy, robot_xy, robot_heading, mask):
        """检测区域中心点是否落在机器人矩形足印(XY平面投影)内。

        向量化: 每个env可能对应不同航点中心, center_xy已按env索引。
        仅对mask=True的env计算 (其余返回False)。

        Args:
            center_xy: [n, 2] 各env当前目标航点坐标
            robot_xy: [n, 2] 机器人位置
            robot_heading: [n] 机器人航向角 (yaw)
            mask: [n] bool — 仅对True的env做footprint检测

        Returns:
            [n] bool — True 表示机器人足印包含该中心点
        """
        n = robot_xy.shape[0]
        result = np.zeros(n, dtype=bool)
        if not np.any(mask):
            return result
        # 相对偏移 (世界坐标)
        dx = center_xy[:, 0] - robot_xy[:, 0]  # [n]
        dy = center_xy[:, 1] - robot_xy[:, 1]  # [n]
        # 旋转到机器人局部坐标 (反向旋转 heading)
        cos_h = np.cos(robot_heading)
        sin_h = np.sin(robot_heading)
        local_x = dx * cos_h + dy * sin_h   # 前后方向
        local_y = -dx * sin_h + dy * cos_h  # 左右方向
        # 检查是否在矩形半尺寸内
        inside = (np.abs(local_x) <= ROBOT_HALF_X) & (np.abs(local_y) <= ROBOT_HALF_Y)
        result = np.where(mask, inside, False)
        return result

    def _euler_to_quat(self, roll, pitch, yaw):
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return np.array([qx, qy, qz, qw], dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    # ============================================================
    # 可视化辅助 (与section011一致)
    # ============================================================

    @staticmethod
    def _sanitize_dof_quaternions(dof_pos, quat_indices):
        for start in quat_indices:
            q = dof_pos[:, start:start + 4]
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            bad = (norms < 1e-6).flatten() | (~np.isfinite(norms)).flatten()
            if np.any(bad):
                dof_pos[bad, start:start + 4] = np.array([0, 0, 0, 1], dtype=np.float32)
            good = ~bad
            if np.any(good):
                dof_pos[good, start:start + 4] = q[good] / norms[good]
        return dof_pos

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        try:
            num_envs = data.shape[0]
            all_dof_pos = data.dof_pos.copy()
            for env_idx in range(num_envs):
                all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                    float(pose_commands[env_idx, 0]), float(pose_commands[env_idx, 1]),
                    float(pose_commands[env_idx, 2])
                ]
            quat_indices = [self._base_quat_start]
            if hasattr(self, '_robot_arrow_dof_start') and self._robot_arrow_body is not None:
                quat_indices.append(self._robot_arrow_dof_start + 3)
                quat_indices.append(self._desired_arrow_dof_start + 3)
            all_dof_pos = self._sanitize_dof_quaternions(all_dof_pos, quat_indices)
            data.set_dof_pos(all_dof_pos, self._model)
            self._model.forward_kinematic(data)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            pass

    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray,
                                desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        num_envs = data.shape[0]
        arrow_offset = 0.5
        all_dof_pos = data.dof_pos.copy()
        for env_idx in range(num_envs):
            arrow_height = robot_pos[env_idx, 2] + arrow_offset
            cur_v = base_lin_vel_xy[env_idx]
            cur_yaw = np.arctan2(cur_v[1], cur_v[0]) if np.linalg.norm(cur_v) > 1e-3 else 0.0
            raq = self._euler_to_quat(0, 0, cur_yaw)
            raq = raq / (np.linalg.norm(raq) + 1e-8)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32), raq
            ])
            des_v = desired_vel_xy[env_idx]
            des_yaw = np.arctan2(des_v[1], des_v[0]) if np.linalg.norm(des_v) > 1e-3 else 0.0
            daq = self._euler_to_quat(0, 0, des_yaw)
            daq = daq / (np.linalg.norm(daq) + 1e-8)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32), daq
            ])
        try:
            quat_indices = [self._base_quat_start]
            if hasattr(self, '_robot_arrow_dof_start'):
                quat_indices.append(self._robot_arrow_dof_start + 3)
                quat_indices.append(self._desired_arrow_dof_start + 3)
            all_dof_pos = self._sanitize_dof_quaternions(all_dof_pos, quat_indices)
            data.set_dof_pos(all_dof_pos, self._model)
            self._model.forward_kinematic(data)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            pass

    # ============================================================
    # 桥优先导航状态机
    # ============================================================

    def _update_waypoint_state(self, info, robot_xy, robot_heading, speed_xy, gyro_z, current_z):
        """通用有序航点推进 — 严格顺序到达，到达goal后进入庆祝。

        向量化实现: 每步仅检查当前目标航点，到达后自动推进到下一个。
        wp_idx = 已完成航点数量 (单调递增, 0 → num_route_waypoints)。
        庆祝: X轴行走 + 蹲坐 (与section011一致)。
        """
        wp_current = info["wp_current"]    # [n] int32 — 当前目标航点索引
        wp_reached = info["wp_reached"]    # [n, N] bool — 各航点是否已到达
        celeb_state = info["celeb_state"]  # [n] int32
        celeb_sit_counter = info["celeb_sit_counter"]  # [n] int32
        n = self._num_envs

        milestone_bonus = np.zeros(n, dtype=np.float32)
        celeb_bonus = np.zeros(n, dtype=np.float32)
        celeb_walk_reward = np.zeros(n, dtype=np.float32)

        # --- 航点推进 (可能连续推进多步, 处理紧邻航点) ---
        max_advances = min(3, self.num_route_waypoints)  # 每step最多推进3个航点
        for _ in range(max_advances):
            not_done = wp_current < self.num_route_waypoints
            if not np.any(not_done):
                break

            # 收集当前航点属性 (clip防越界)
            safe_idx = np.clip(wp_current, 0, self.num_route_waypoints - 1)
            target_xy = self.wp_xy[safe_idx]        # [n, 2]
            target_r = self.wp_radius[safe_idx]     # [n]
            target_z_lo = self.wp_z_min[safe_idx]   # [n]
            target_z_hi = self.wp_z_max[safe_idx]   # [n]

            dist = np.linalg.norm(robot_xy - target_xy, axis=1)
            z_ok = (current_z >= target_z_lo) & (current_z <= target_z_hi)

            # footprint-contains 检测: radius<=0 的航点使用机器人矩形足印检测
            use_fp = self.wp_use_footprint[safe_idx]  # [n] bool
            if np.any(use_fp):
                fp_arrived = self._footprint_contains_point(target_xy, robot_xy, robot_heading, use_fp)
                radius_arrived = dist < target_r
                proximity = np.where(use_fp, fp_arrived, radius_arrived)
            else:
                proximity = dist < target_r
            arrived = not_done & proximity & z_ok

            # first arrival only
            already = wp_reached[np.arange(n), safe_idx]
            first = arrived & ~already

            if not np.any(first):
                break

            # 标记到达 + 发放奖金
            wp_reached[np.arange(n), safe_idx] = wp_reached[np.arange(n), safe_idx] | arrived
            milestone_bonus += np.where(first, self.wp_bonus[safe_idx], 0.0)
            wp_current = np.where(first, wp_current + 1, wp_current)

        # --- 更新 wp_idx ---
        wp_idx = np.sum(wp_reached, axis=1).astype(np.int32)
        info["wp_current"] = wp_current
        info["wp_reached"] = wp_reached
        info["wp_idx"] = wp_idx

        # --- 庆祝: 到达所有航点后 X轴行走 + 蹲坐 (与section011一致) ---
        all_done = (wp_current >= self.num_route_waypoints)
        scales = self._cfg.reward_config.scales

        # IDLE → WALKING (进入庆祝阶段)
        start_celeb = all_done & (celeb_state == CELEB_IDLE)
        if np.any(start_celeb):
            celeb_state = np.where(start_celeb, CELEB_WALKING, celeb_state)

        # WALKING: 走向X轴端点
        walking = all_done & (celeb_state == CELEB_WALKING)
        if np.any(walking):
            # 连续奖励: 接近X轴端点 (delta-based)
            d_x_target = np.linalg.norm(robot_xy - self.celeb_x_target[np.newaxis, :], axis=1)
            last_d_x = info.get("last_celeb_x_dist", d_x_target.copy())
            x_delta = last_d_x - d_x_target  # 正 = 靠近
            celeb_walk_reward += np.where(walking, np.clip(x_delta * scales.get("celeb_walk_approach", 200.0), -0.5, 2.5), 0.0)
            info["last_celeb_x_dist"] = d_x_target.copy()

            # 到达X轴端点 → 给一次性奖励, 进入SITTING
            arrived_x = walking & (d_x_target < self.celeb_walk_radius)
            if np.any(arrived_x):
                celeb_bonus += np.where(arrived_x, scales.get("celeb_walk_bonus", 30.0), 0.0)
                celeb_state = np.where(arrived_x, CELEB_SITTING, celeb_state)

        # SITTING: 蹲坐 (z低于阈值, 保持N步)
        sitting = all_done & (celeb_state == CELEB_SITTING)
        if np.any(sitting):
            # 连续奖励: z越低越好 (鼓励蹲下)
            z_below = np.maximum(1.55 - current_z, 0.0)  # standing ≈ 1.55, 越低越好
            celeb_walk_reward += np.where(sitting, scales.get("celeb_sit_reward", 5.0) * z_below, 0.0)

            # 检测蹲坐: z < 阈值时累加计数器, 否则清零
            is_low = current_z < self.celeb_sit_z
            celeb_sit_counter = np.where(sitting & is_low, celeb_sit_counter + 1, np.where(sitting, 0, celeb_sit_counter))

            # 蹲坐足够久 → DONE
            sit_done = sitting & (celeb_sit_counter >= self.celeb_sit_steps)
            if np.any(sit_done):
                celeb_bonus += np.where(sit_done, scales.get("celebration_bonus", 50.0), 0.0)
                celeb_state = np.where(sit_done, CELEB_DONE, celeb_state)

        info["celeb_sit_counter"] = celeb_sit_counter
        info["celeb_state"] = celeb_state

        return info, milestone_bonus, celeb_bonus, celeb_walk_reward

    def _get_current_target(self, info, robot_xy):
        """获取当前导航目标 — 有序航点索引直接查表"""
        wp_current = info["wp_current"]
        n = len(wp_current)
        safe_idx = np.clip(wp_current, 0, self.num_route_waypoints - 1)
        return self.wp_xy[safe_idx].copy()

    # ============================================================
    # apply_action & torques
    # ============================================================

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        actions = np.where(np.isfinite(actions), actions, 0.0)
        actions = np.clip(actions, -1.0, 1.0)
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
        kp, kv = 100.0, 8.0
        torques = kp * (target_pos - current_pos) - kv * current_vel
        self._raw_torques = torques.copy()
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)
        return np.clip(torques, -torque_limits, torque_limits)

    # ============================================================
    # update_state — 核心循环
    # ============================================================

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        cfg = self._cfg
        info = state.info

        # --- 获取物理状态 ---
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)

        # --- NaN安全防护 ---
        nan_mask = (
            np.any(~np.isfinite(joint_vel), axis=1)
            | np.any(~np.isfinite(joint_pos), axis=1)
            | np.any(~np.isfinite(root_pos), axis=1)
            | np.any(~np.isfinite(root_vel), axis=1)
        )
        if np.any(nan_mask):
            joint_vel = np.where(nan_mask[:, np.newaxis], 0.0, joint_vel)
            joint_pos = np.where(nan_mask[:, np.newaxis], self.default_angles, joint_pos)
            root_vel = np.where(nan_mask[:, np.newaxis], 0.0, root_vel)
            root_pos = np.where(nan_mask[:, np.newaxis], np.array([[0.0, 9.5, 1.8, 0.0, 0.0, 0.0]]), root_pos)
            root_quat = np.where(nan_mask[:, np.newaxis], np.array([[0.0, 0.0, 0.0, 1.0]]), root_quat)
            info["nan_terminated"] = nan_mask

        joint_pos_rel = joint_pos - self.default_angles
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        robot_xy = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        current_z = root_pos[:, 2]
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)

        # --- 航点 & 庆祝更新 ---
        info, wp_bonus, celeb_bonus, celeb_walk_reward = \
            self._update_waypoint_state(info, robot_xy, robot_heading, speed_xy, gyro[:, 2], current_z)

        # --- 当前导航目标 ---
        target_xy = self._get_current_target(info, robot_xy)
        in_celeb = (info["wp_current"] >= self.num_route_waypoints)

        position_error = target_xy - robot_xy
        distance_to_target = np.linalg.norm(position_error, axis=1)

        pose_commands = np.column_stack([target_xy, np.zeros(self._num_envs, dtype=np.float32)])
        info["pose_commands"] = pose_commands

        # 到达当前WP? (使用当前航点半径)
        safe_idx = np.clip(info["wp_current"], 0, self.num_route_waypoints - 1)
        current_wp_radius = self.wp_radius[safe_idx]
        reached_wp = distance_to_target < current_wp_radius

        # 运动命令
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where((reached_wp | in_celeb)[:, np.newaxis], 0.0, desired_vel_xy)

        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = self._wrap_angle(desired_heading - robot_heading)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_wp | in_celeb, 0.0, desired_yaw_rate)

        heading_diff = self._wrap_angle(desired_heading - robot_heading)

        # --- 观测 (69维, 与section011完全对齐) ---
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        last_actions = info["current_actions"]

        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi

        # 足端接触 + 基座高度
        foot_forces_obs = self._get_foot_contact_forces(data)
        foot_contact = (np.linalg.norm(foot_forces_obs, axis=2) > 0.5).astype(np.float32)
        base_height_norm = np.clip((current_z - 0.5) / 1.2, -1.0, 1.0)[:, np.newaxis]

        # trunk加速度计 + 关节扭矩 (v20)
        trunk_acc_raw = self._get_trunk_acc(data)
        trunk_acc_norm = np.clip(trunk_acc_raw / 20.0, -3.0, 3.0)
        actual_torques = self._get_actuator_torques(data)
        torques_normalized = actual_torques / self.torque_limits[np.newaxis, :]

        # 庆祝进度观测 (v4fix: /3.0 not /2.0 — celeb_state ∈ {0,1,2,3})
        celeb_progress = info["celeb_state"].astype(np.float32) / 3.0

        obs = np.concatenate([
            noisy_linvel,                              # 3
            noisy_gyro,                                # 3
            projected_gravity,                         # 3
            noisy_joint_angle,                         # 12
            noisy_joint_vel,                           # 12
            last_actions,                              # 12
            position_error_normalized,                 # 2
            heading_error_normalized[:, np.newaxis],   # 1
            base_height_norm,                          # 1
            celeb_progress[:, np.newaxis],             # 1
            foot_contact,                              # 4
            trunk_acc_norm,                            # 3
            torques_normalized,                        # 12
        ], axis=-1)
        assert obs.shape == (data.shape[0], 69), f"obs shape {obs.shape} != (N, 69)"

        # 可视化
        self._update_target_marker(data, pose_commands)
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel[:, :2])

        # --- 终止 ---
        terminated_state = self._compute_terminated(state, projected_gravity, joint_vel, robot_xy, current_z)
        terminated = terminated_state.terminated

        # --- 奖励 ---
        reward = self._compute_reward(
            data, info, base_lin_vel, gyro, projected_gravity,
            joint_vel, distance_to_target, position_error, reached_wp,
            terminated, robot_heading, robot_xy, current_z, speed_xy,
            wp_bonus, celeb_bonus, celeb_walk_reward, in_celeb
        )

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        # 庆祝完成截断
        celeb_done = (info["celeb_state"] == CELEB_DONE)
        self._success_truncate = celeb_done

        # v57: 停滞检测 — 若机器人长时间不动则截断 (庆祝阶段豁免)
        enable_stag = getattr(cfg, 'enable_stagnation_truncate', True)
        stag_cfg_window = getattr(cfg, 'stagnation_window_steps', 1000)
        stag_cfg_dist = getattr(cfg, 'stagnation_min_distance', 0.5)
        stag_cfg_grace = getattr(cfg, 'stagnation_grace_steps', 500)
        ep_steps = info.get("steps", np.zeros(self._num_envs, dtype=np.int32))
        anchor_xy = info["stagnation_anchor_xy"]
        anchor_step = info["stagnation_anchor_step"]
        dist_from_anchor = np.linalg.norm(robot_xy - anchor_xy, axis=1)
        # 更新锚点: 机器人移动足够远 OR 到达新航点时刷新
        moved_enough = dist_from_anchor >= stag_cfg_dist
        wp_advanced = info["wp_current"] > info.get("stagnation_last_wp", np.zeros(self._num_envs, dtype=np.int32))
        refresh = moved_enough | wp_advanced
        info["stagnation_anchor_xy"] = np.where(refresh[:, np.newaxis], robot_xy, anchor_xy)
        info["stagnation_anchor_step"] = np.where(refresh, ep_steps, anchor_step)
        info["stagnation_last_wp"] = info["wp_current"].copy()
        # 检测停滞: 超过窗口时间仍未移动, 且不在庆祝阶段, 且过了grace期
        steps_since_anchor = ep_steps - info["stagnation_anchor_step"]
        stagnant = (
            enable_stag
            & (steps_since_anchor > stag_cfg_window)
            & (ep_steps >= stag_cfg_grace)
            & ~in_celeb
        )
        self._stagnation_truncate = stagnant

        state.info["metrics"] = {
            "distance_to_target": distance_to_target,
            "reached_fraction": reached_wp.astype(np.float32),
            "stair_cleared_fraction": (info["wp_idx"] >= 2).astype(np.float32),
            "wp_idx_mean": info["wp_idx"].astype(np.float32),
            "wp_current_mean": info["wp_current"].astype(np.float32),
            "celeb_state_mean": info["celeb_state"].astype(np.float32),
        }
        return state

    def _update_truncate(self):
        super()._update_truncate()
        if hasattr(self, '_success_truncate'):
            self._state.truncated = np.logical_or(self._state.truncated, self._success_truncate)
        if hasattr(self, '_stagnation_truncate'):
            self._state.truncated = np.logical_or(self._state.truncated, self._stagnation_truncate)

    # ============================================================
    # 终止条件 (与section011一致: hard/soft分层 + grace + OOB)
    # ============================================================

    def _compute_terminated(self, state, projected_gravity=None, joint_vel=None, robot_xy=None, current_z=None):
        data = state.data
        n = self._num_envs

        # === HARD terminations (never grace-protected) ===
        hard_terminated = np.zeros(n, dtype=bool)

        # 摔倒检测: 倾斜角度过大 — 侧躺/翻倒必须立即终止
        if projected_gravity is not None:
            gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
            gz = projected_gravity[:, 2]
            tilt_angle = np.arctan2(gxy, np.abs(gz))
            hard_tilt = getattr(self._cfg, 'hard_tilt_deg', 70.0)
            hard_terminated |= tilt_angle > np.deg2rad(hard_tilt)

        # 超出边界 — 竞赛规则: 直接终止
        bounds = getattr(self._cfg, 'course_bounds', None)
        if bounds is not None and robot_xy is not None and current_z is not None:
            oob_x = (robot_xy[:, 0] < bounds.x_min) | (robot_xy[:, 0] > bounds.x_max)
            oob_y = (robot_xy[:, 1] < bounds.y_min) | (robot_xy[:, 1] > bounds.y_max)
            oob_z = current_z < bounds.z_min
            oob = oob_x | oob_y | oob_z
            hard_terminated |= oob
            state.info["oob_terminated"] = oob

        # 关节速度异常 / NaN — 物理爆炸, 必须终止
        if joint_vel is not None:
            vel_max = np.abs(joint_vel).max(axis=1)
            vel_overflow = vel_max > self._cfg.max_dof_vel
            vel_extreme = np.isnan(joint_vel).any(axis=1) | np.isinf(joint_vel).any(axis=1)
            hard_terminated |= vel_overflow | vel_extreme
            # 关节加速度异常 — 单步速度变化>80 rad/s = 物理不稳定
            last_dof_vel = state.info.get("last_dof_vel", np.zeros_like(joint_vel))
            dof_acc_max = np.abs(joint_vel - np.clip(last_dof_vel, -100.0, 100.0)).max(axis=1)
            hard_terminated |= dof_acc_max > 80.0

        nan_terminated = state.info.get("nan_terminated", np.zeros(n, dtype=bool))
        hard_terminated |= nan_terminated

        # === SOFT terminations (grace-protected for initial stabilization) ===
        soft_terminated = np.zeros(n, dtype=bool)

        # 基座接触地面 — grace period内可能是初始着陆抖动
        if getattr(self._cfg, 'enable_base_contact_term', True):
            try:
                base_contact_value = self._model.get_sensor_value("base_contact", data)
                # motrixsim 0.5+ returns 3D force vector (n, 3); compute magnitude
                if base_contact_value.ndim >= 2 and base_contact_value.shape[-1] == 3:
                    force_mag = np.linalg.norm(base_contact_value, axis=-1)
                    base_contact = (force_mag > 0.01).flatten()[:n]
                elif base_contact_value.ndim == 0:
                    base_contact = np.array([float(base_contact_value) > 0.01], dtype=bool)
                else:
                    base_contact = (base_contact_value.flatten() > 0.01)[:n]
            except Exception:
                base_contact = np.zeros(n, dtype=bool)
            soft_terminated |= base_contact

        # 中等倾斜 — grace期间允许恢复, 之后终止
        soft_tilt = getattr(self._cfg, 'soft_tilt_deg', 50.0)
        if projected_gravity is not None and soft_tilt > 0:
            gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
            gz = projected_gravity[:, 2]
            tilt_angle = np.arctan2(gxy, np.abs(gz))
            soft_terminated |= tilt_angle > np.deg2rad(soft_tilt)

        # Apply grace period only to soft terminations
        grace_steps = getattr(self._cfg, 'grace_period_steps', 0)
        if grace_steps > 0:
            ep_steps = state.info.get("steps", np.zeros(n, dtype=np.int32))
            in_grace = ep_steps < grace_steps
            soft_terminated = np.where(in_grace, False, soft_terminated)

        terminated = hard_terminated | soft_terminated
        return state.replace(terminated=terminated)

    # ============================================================
    # 奖励计算
    # ============================================================

    def _compute_reward(self, data, info, base_lin_vel, gyro,
                         projected_gravity, joint_vel, distance_to_target, position_error,
                         reached_wp, terminated, robot_heading, robot_xy, current_z,
                         speed_xy, wp_bonus, celeb_bonus, celeb_walk_reward, in_celeb):
        scales = self._cfg.reward_config.scales
        n = self._num_envs

        # ===== 累积奖金追踪 =====
        accumulated_bonus = info.get("accumulated_bonus", np.zeros(n, dtype=np.float32))

        # ===== 导航跟踪 =====
        position_tracking = np.exp(-distance_to_target / 5.0)

        target_bearing = np.arctan2(position_error[:, 1], position_error[:, 0])
        facing_error = self._wrap_angle(target_bearing - robot_heading)
        heading_tracking = np.where(reached_wp | in_celeb, 1.0, np.exp(-np.abs(facing_error) / 0.5))

        direction_to_target = position_error / (np.linalg.norm(position_error, axis=1, keepdims=True) + 1e-8)
        forward_velocity = np.clip(np.sum(base_lin_vel[:, :2] * direction_to_target, axis=1), -0.5, 0.8)

        # 航点approach: step-delta
        last_wp_dist = info.get("last_wp_distance", distance_to_target.copy())
        wp_delta = last_wp_dist - distance_to_target
        info["last_wp_distance"] = distance_to_target.copy()
        wp_approach = np.clip(wp_delta * scales.get("waypoint_approach", 100.0), -0.5, 5.0)

        wp_facing = scales.get("waypoint_facing", 0.15) * heading_tracking

        # 存活奖励 (条件式 + 时间衰减)
        gz = np.clip(-projected_gravity[:, 2], 0.0, 1.0)
        upright_factor = np.where(gz > 0.9, 1.0, np.where(gz > 0.7, 0.5, 0.0))
        alive_decay_horizon = scales.get("alive_decay_horizon", 3000.0)
        alive_time_decay = np.clip(1.0 - ep_steps / alive_decay_horizon, 0.0, 1.0)
        alive_bonus = scales.get("alive_bonus", 0.05) * upright_factor * alive_time_decay

        # ===== 稳定性惩罚 =====
        orientation_penalty = np.sum(np.square(projected_gravity[:, :2]), axis=1)
        
        # v59b: 多区域方向补偿 — 入口楼梯 + 河谷斜坡 + 远端楼梯
        # 使用方向无关匹配: min(|gy-mag|, |gy+mag|), 因为robot在河谷和远端楼梯双向行走
        current_y_for_slope = robot_xy[:, 1]
        on_stairs = (current_y_for_slope > 12.0) & (current_y_for_slope < 14.5)           # 入口右楼梯 26.5°
        on_valley_south = (current_y_for_slope > 14.33) & (current_y_for_slope < 17.0)    # 南侧河谷斜坡 ~14°
        on_valley_north = (current_y_for_slope > 19.0) & (current_y_for_slope < 21.33)    # 北侧河谷斜坡 ~14°
        on_far_stairs = (current_y_for_slope > 21.33) & (current_y_for_slope < 23.33)     # 远端左梯37°+右梯27°
        # 各区域期望|gy|幅度 (取sin of slope angle)
        expected_gy_mag = np.where(on_stairs, 0.447,
                         np.where(on_valley_south | on_valley_north, 0.201,
                         np.where(on_far_stairs, 0.600, 0.0)))
        on_any_slope = on_stairs | on_valley_south | on_valley_north | on_far_stairs
        # 方向无关误差: 允许正/负倾斜都能获得补偿 (远端楼梯上下、河谷锯齿路线)
        gy_actual = projected_gravity[:, 1]
        gy_error = np.minimum(np.abs(gy_actual - expected_gy_mag), np.abs(gy_actual + expected_gy_mag))
        slope_compensation = np.where(on_any_slope, np.exp(-np.square(gy_error) / 0.05), 0.0)
        slope_orientation_reward = scales.get("slope_orientation", 0.04) * slope_compensation

        lin_vel_z_penalty = np.square(np.clip(base_lin_vel[:, 2], -50.0, 50.0))
        ang_vel_xy_penalty = np.sum(np.square(np.clip(gyro[:, :2], -50.0, 50.0)), axis=1)

        actual_torques_r = self._get_actuator_torques(data)
        torque_penalty = np.sum(np.square(np.clip(actual_torques_r, -200.0, 200.0)), axis=1)
        # v4fix: dof_pos_penalty — deviation from default standing angles (was ghost parameter)
        _joint_pos = self.get_dof_pos(data)
        dof_pos_penalty = np.sum(np.square(_joint_pos - self.default_angles), axis=1)
        safe_joint_vel = np.clip(joint_vel, -100.0, 100.0)
        dof_vel_penalty = np.sum(np.square(safe_joint_vel), axis=1)
        last_dof_vel = info.get("last_dof_vel", np.zeros_like(joint_vel))
        dof_acc_penalty = np.sum(np.square(safe_joint_vel - np.clip(last_dof_vel, -100.0, 100.0)), axis=1)
        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        # v20: 冲击惩罚
        trunk_acc_r = self._get_trunk_acc(data)
        trunk_acc_mag = np.linalg.norm(trunk_acc_r, axis=1)
        impact_excess = np.maximum(trunk_acc_mag - 15.0, 0.0)
        impact_penalty = np.square(impact_excess) / 100.0

        # v20: 扭矩饱和惩罚
        saturation_ratio = np.abs(actual_torques_r) / self.torque_limits[np.newaxis, :]
        torque_sat_penalty = np.sum(np.maximum(saturation_ratio - 0.9, 0.0) ** 2, axis=1)

        # ===== 爬高高度进步 =====
        last_z = info.get("last_z", current_z.copy())
        z_delta = current_z - last_z
        info["last_z"] = current_z.copy()
        height_progress = scales.get("height_progress", 12.0) * np.maximum(z_delta, 0.0)

        # v4fix: 高度振荡惩罚 — penalize rapid Z bouncing on stairs (ported from section011)
        z_osc = np.abs(z_delta)
        height_osc_penalty = scales.get("height_oscillation", -2.0) * np.maximum(z_osc - 0.015, 0.0)

        # ===== 地形里程碑 (Y轴进度) =====
        current_y = robot_xy[:, 1]
        milestones_reached = info.get("milestones_reached", np.zeros((n, 4), dtype=bool))
        traversal_total = np.zeros(n, dtype=np.float32)
        # m1: 到达楼梯区域 (y > 12.0)
        m1 = (current_y > 12.0)
        m1_first = m1 & ~milestones_reached[:, 0]
        milestones_reached[:, 0] |= m1
        traversal_total += np.where(m1_first, scales.get("traversal_bonus", 20.0), 0.0)
        # m2: 到达桥/球区域 (y > 15.0 且 z > 1.5)
        m2 = (current_y > 15.0) & (current_z > 1.5)
        m2_first = m2 & ~milestones_reached[:, 1]
        milestones_reached[:, 1] |= m2
        traversal_total += np.where(m2_first, scales.get("traversal_bonus", 20.0), 0.0)
        # m3: 过了桥中段 (y > 18.0 且 z > 2.0)
        m3 = (current_y > 18.0) & (current_z > 2.0)
        m3_first = m3 & ~milestones_reached[:, 2]
        milestones_reached[:, 2] |= m3
        traversal_total += np.where(m3_first, scales.get("traversal_bonus", 20.0), 0.0)
        # m4: 到达出口楼梯区域 (y > 21.0)
        m4 = (current_y > 21.0)
        m4_first = m4 & ~milestones_reached[:, 3]
        milestones_reached[:, 3] |= m4
        traversal_total += np.where(m4_first, scales.get("traversal_bonus", 20.0), 0.0)
        info["milestones_reached"] = milestones_reached

        # ===== 楼梯区抬脚 =====
        # v54: TerrainZone-driven clearance boost with pre-zone transition (replaces hardcoded on_stair)
        foot_clearance_scale = scales.get("foot_clearance", 0.02)
        if foot_clearance_scale > 0:
            foot_forces = self._get_foot_contact_forces(data)
            force_mag = np.linalg.norm(foot_forces, axis=2)
            in_swing = force_mag < 0.5
            calf_indices = [2, 5, 8, 11]
            calf_vel = np.abs(joint_vel[:, calf_indices])
            clearance_boost = self._terrain_scale.compute_clearance_boost(current_y, scales)
            clearance_scale_vec = foot_clearance_scale * clearance_boost
            foot_clearance_reward = clearance_scale_vec * np.sum(
                in_swing.astype(np.float32) * np.clip(calf_vel, 0.0, 5.0) * 0.2, axis=1
            )
        else:
            foot_clearance_reward = np.zeros(n, dtype=np.float32)

        # ===== 摆动相接触惩罚 =====
        # v54: TerrainZone-driven swing contact scaling (replaces hardcoded on_stair)
        terrain_swing_scale = self._terrain_scale.compute_swing_scale(current_y, scales)
        swing_penalty = (
            scales.get("swing_contact_penalty", -0.025)
            * self._compute_swing_contact_penalty(data, joint_vel)
            * terrain_swing_scale
        )

        # ===== 拖脚惩罚 (Drag-Foot Penalty) =====
        # 腿有接触 + 小腿关节角速度低 = 拖地行为 (foot_clearance的盲区)
        drag_foot_scale = scales.get("drag_foot_penalty", 0.0)
        if drag_foot_scale < 0:
            if foot_clearance_scale > 0:
                drag_in_contact = ~in_swing
            else:
                foot_forces_df = self._get_foot_contact_forces(data)
                force_mag_df = np.linalg.norm(foot_forces_df, axis=2)
                drag_in_contact = force_mag_df > 0.5
                calf_indices = [2, 5, 8, 11]
                calf_vel = np.abs(joint_vel[:, calf_indices])
            low_vel = calf_vel < 1.0
            dragging = drag_in_contact & low_vel
            drag_foot_raw = np.sum(dragging.astype(np.float32), axis=1)
            drag_foot_penalty = drag_foot_scale * drag_foot_raw
        else:
            drag_foot_penalty = np.zeros(n, dtype=np.float32)

        # ===== 停滞渐进惩罚 (Stagnation Ramp Penalty) =====
        # 停滞检测只会截断episode, 不给惩罚信号 → 添加渐进惩罚
        stagnation_penalty_scale = scales.get("stagnation_penalty", 0.0)
        if stagnation_penalty_scale < 0:
            stag_window = getattr(self._cfg, 'stagnation_window_steps', 1000)
            stag_grace = getattr(self._cfg, 'stagnation_grace_steps', 500)
            stag_anchor_step = info.get("stagnation_anchor_step", np.zeros(n, dtype=np.int32))
            stag_ep_steps = info.get("steps", np.zeros(n, dtype=np.int32))
            stag_since = stag_ep_steps - stag_anchor_step
            stag_ratio = np.clip((stag_since.astype(np.float32) / stag_window - 0.5) * 2.0, 0.0, 1.0)
            past_grace = stag_ep_steps >= stag_grace
            stagnation_penalty = stagnation_penalty_scale * stag_ratio * past_grace.astype(np.float32)
            stagnation_penalty = np.where(in_celeb, 0.0, stagnation_penalty)
        else:
            stagnation_penalty = np.zeros(n, dtype=np.float32)

        # ===== 蹲坐惩罚 (Crouch Penalty) =====
        # 机器人坐下时base高度低于正常站立高度
        crouch_penalty_scale = scales.get("crouch_penalty", 0.0)
        if crouch_penalty_scale < 0:
            # section012地形: 楼梯区y∈12-14.3 z从1.3→2.3, 河谷区y>14.3 z≈1.8-2.8
            terrain_z_est = np.where(
                current_y < 12.0, 1.29,                                    # 入口平台
                np.where(current_y < 14.3,
                         1.32 + (current_y - 12.0) / 2.3 * 0.97,          # 右楼梯线性插值 1.32→2.29
                np.where(current_y < 16.0, 1.80,                           # 河谷底部 (保守)
                         1.50)))                                            # 河谷远端
            clearance = current_z - terrain_z_est
            min_clearance = 0.20  # 正常站立clearance≈0.25m, 低于0.20m=蹲坐
            crouch_penalty = np.where(clearance < min_clearance, crouch_penalty_scale, 0.0)
        else:
            crouch_penalty = np.zeros(n, dtype=np.float32)

        # ===== 步态质量 =====
        gait = self._compute_gait_rewards(data, info, base_lin_vel, robot_heading, projected_gravity)
        gait_stance = scales.get("stance_ratio", 0.08) * gait["stance_reward"]

        # ===== 累积奖金更新 =====
        step_bonus = np.where(terminated, 0.0, wp_bonus + celeb_bonus + traversal_total)
        accumulated_bonus = accumulated_bonus + step_bonus
        info["accumulated_bonus"] = accumulated_bonus

        # ===== 终止惩罚 + 得分清零 =====
        base_termination = scales.get("termination", -100.0)
        score_clear = scales.get("score_clear_factor", 0.3)
        score_clear_penalty = np.where(terminated, np.maximum(-score_clear * accumulated_bonus, -100.0), 0.0)
        termination_penalty = np.where(terminated, base_termination, 0.0) + score_clear_penalty

        # ===== 惩罚汇总 =====
        penalties = (
            scales.get("orientation", -0.015) * orientation_penalty
            + scales.get("lin_vel_z", -0.06) * lin_vel_z_penalty
            + scales.get("ang_vel_xy", -0.01) * ang_vel_xy_penalty
            + scales.get("torques", -5e-6) * torque_penalty
            + scales.get("dof_pos", 0.0) * dof_pos_penalty
            + scales.get("dof_vel", -3e-5) * dof_vel_penalty
            + scales.get("dof_acc", -1.5e-7) * dof_acc_penalty
            + scales.get("action_rate", -0.005) * action_rate_penalty
            + scales.get("impact_penalty", -0.02) * impact_penalty
            + scales.get("torque_saturation", -0.01) * torque_sat_penalty
            + termination_penalty
            + swing_penalty
            + height_osc_penalty
            + drag_foot_penalty
            + stagnation_penalty
            + crouch_penalty
        )

        # ===== 综合奖励 =====
        nav_reward = (
            scales.get("position_tracking", 0.05) * position_tracking
            + wp_approach
            + wp_facing
            + scales.get("forward_velocity", 3.0) * forward_velocity
            + alive_bonus
        )
        # v4fix: 只在SITTING/DONE阶段停止导航奖励, WALKING阶段保持导航 (match section011)
        celeb_stop = in_celeb & (info["celeb_state"] >= CELEB_SITTING)
        nav_reward = np.where(celeb_stop, alive_bonus, nav_reward)

        reward = (
            nav_reward
            + wp_bonus
            + celeb_bonus
            + celeb_walk_reward
            + height_progress
            + traversal_total
            + foot_clearance_reward
            + slope_orientation_reward
            + gait_stance
            + penalties
        )

        # 终止时只保留惩罚
        reward = np.where(terminated, termination_penalty, reward)
        reward = np.where(np.isfinite(reward), reward, -50.0)

        # TensorBoard
        info["Reward"] = {
            "position_tracking": scales.get("position_tracking", 0.05) * position_tracking,
            "heading_tracking": wp_facing,
            "forward_velocity": scales.get("forward_velocity", 3.0) * forward_velocity,
            "wp_approach": wp_approach,
            "alive_bonus": alive_bonus,
            "wp_bonus": wp_bonus,
            "celeb_bonus": celeb_bonus,
            "celeb_walk_reward": celeb_walk_reward,
            "height_progress": height_progress,
            "slope_orientation": slope_orientation_reward,
            "traversal_bonus": traversal_total,
            "penalties": penalties,
            "termination": termination_penalty,
            "swing_contact_penalty": swing_penalty,
            "foot_clearance": foot_clearance_reward,
            "score_clear_penalty": score_clear_penalty,
            "gait_stance": gait_stance,
            "impact_penalty": scales.get("impact_penalty", -0.02) * impact_penalty,
            "torque_saturation": scales.get("torque_saturation", -0.01) * torque_sat_penalty,
            "dof_pos_penalty": scales.get("dof_pos", 0.0) * dof_pos_penalty,
            "height_osc_penalty": height_osc_penalty,
            "drag_foot_penalty": drag_foot_penalty,
            "stagnation_penalty": stagnation_penalty,
            "crouch_penalty": crouch_penalty,
        }
        return reward

    # ============================================================
    # reset
    # ============================================================

    def reset(self, data: mtx.SceneData, done: np.ndarray = None):
        cfg = self._cfg
        num_envs = data.shape[0]

        random_x = np.random.uniform(self.spawn_x_range[0], self.spawn_x_range[1], size=(num_envs,))
        random_y = np.random.uniform(self.spawn_y_range[0], self.spawn_y_range[1], size=(num_envs,))
        robot_init_xy = self.spawn_center[:2] + np.column_stack([random_x, random_y])
        terrain_heights = np.full(num_envs, self.spawn_center[2], dtype=np.float32)
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])

        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))
        dof_pos[:, 3:6] = robot_init_xyz

        # 初始朝向: +Y方向 (yaw=π/2) + 随机扰动, 与section011一致
        reset_yaw_scale = getattr(cfg, 'reset_yaw_scale', 0.1)
        base_yaw = np.pi / 2
        yaw_noise = np.random.uniform(-np.pi * reset_yaw_scale, np.pi * reset_yaw_scale, size=(num_envs,))
        for env_idx in range(num_envs):
            init_quat = self._euler_to_quat(0, 0, base_yaw + yaw_noise[env_idx])
            dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = init_quat
            if self._robot_arrow_body is not None:
                for s, e in [(self._robot_arrow_dof_start + 3, self._robot_arrow_dof_end),
                             (self._desired_arrow_dof_start + 3, self._desired_arrow_dof_end)]:
                    q = dof_pos[env_idx, s:e]
                    qn = np.linalg.norm(q)
                    dof_pos[env_idx, s:e] = q / qn if qn > 1e-6 else [0, 0, 0, 1]

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # 初始目标: 第一个航点 (wp_current=0)
        temp_info = {"wp_current": np.zeros(num_envs, dtype=np.int32)}
        first_target = self._get_current_target(temp_info, robot_init_xy)
        pose_commands = np.column_stack([first_target, np.zeros((num_envs, 1), dtype=np.float32)])
        self._update_target_marker(data, pose_commands)

        # 物理状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel_r = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        robot_heading = self._get_heading_from_quat(root_quat)

        target_xy = pose_commands[:, :2]
        position_error = target_xy - root_pos[:, :2]
        distance_to_target = np.linalg.norm(position_error, axis=1)
        reached = distance_to_target < self.wp_radius[0]  # 初始时wp_current=0
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached[:, np.newaxis], 0.0, desired_vel_xy)
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel[:, :2])

        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_diff = self._wrap_angle(desired_heading - robot_heading)

        # 观测 (69维, 与update_state一致)
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        base_height_norm = np.clip((robot_init_xyz[:, 2] - 0.5) / 1.2, -1.0, 1.0)[:, np.newaxis]

        obs = np.concatenate([
            base_lin_vel * cfg.normalization.lin_vel,                   # 3
            gyro * cfg.normalization.ang_vel,                           # 3
            projected_gravity,                                          # 3
            joint_pos_rel * cfg.normalization.dof_pos,                  # 12
            joint_vel_r * cfg.normalization.dof_vel,                    # 12
            np.zeros((num_envs, self._num_action), dtype=np.float32),   # 12 last_actions
            position_error_normalized,                                  # 2
            heading_error_normalized[:, np.newaxis],                    # 1
            base_height_norm,                                           # 1
            np.zeros((num_envs, 1), dtype=np.float32),                 # 1 celeb_progress
            np.zeros((num_envs, 4), dtype=np.float32),                 # 4 foot_contact
            np.zeros((num_envs, 3), dtype=np.float32),                 # 3 trunk_acc
            np.zeros((num_envs, 12), dtype=np.float32),                # 12 actuator_torques
        ], axis=-1)
        assert obs.shape == (num_envs, 69), f"reset obs shape {obs.shape} != ({num_envs}, 69)"

        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=np.bool_),
            "last_z": terrain_heights.copy(),
            "last_wp_distance": distance_to_target.copy(),
            "milestones_reached": np.zeros((num_envs, 4), dtype=bool),
            # 有序航点导航状态
            "wp_current": np.zeros(num_envs, dtype=np.int32),
            "wp_reached": np.zeros((num_envs, self.num_route_waypoints), dtype=bool),
            "wp_idx": np.zeros(num_envs, dtype=np.int32),
            # 庆祝状态机 (v58: X轴行走 + 蹲坐)
            "celeb_state": np.full(num_envs, CELEB_IDLE, dtype=np.int32),
            "celeb_sit_counter": np.zeros(num_envs, dtype=np.int32),
            # 累积奖金 (终止清零用)
            "accumulated_bonus": np.zeros(num_envs, dtype=np.float32),
            "oob_terminated": np.zeros(num_envs, dtype=bool),
            # 停滞检测锚点 (与section011一致)
            "stagnation_anchor_xy": robot_init_xy.copy(),
            "stagnation_anchor_step": np.zeros(num_envs, dtype=np.int32),
            "stagnation_last_wp": np.zeros(num_envs, dtype=np.int32),
        }

        return obs, info


# 聚焦楼梯+河谷训练: 复用同一Env类, 不同cfg → 不同航点路线
registry.register_env("vbot_navigation_section012_stairs", VBotSection012Env, "np")
