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
VBot Section011 分阶段区域收集导航环境 - 竞赛得分区 + 跳跃庆祝

竞赛规则:
  Phase 0: 收集全部3个笑脸得分区 (任意顺序, 全部收集才进入下一阶段)
  Phase 1: 全部笑脸后收集全部3个红包区 (任意顺序)
  Phase 2: 全部红包后爬到高台庆祝区
  Phase 3: 在高台上跳跃庆祝

导航目标: 当前阶段最近的未收集区域中心
wp_idx = smileys_count + red_packets_count + platform_reached (0-7)
"""

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection011EnvCfg, TerrainScaleHelper

# ============================================================
# 庆祝状态机常量
# ============================================================
CELEB_IDLE = 0        # 未开始庆祝
CELEB_JUMP = 1        # 在高台上, 准备/正在跳跃
CELEB_LANDING = 2     # 跳到高点后等待落地 (z回到standing以下)
CELEB_DONE = 3        # 所有跳跃完成

# 机器人躯干矩形足印半尺寸 (m) — 来自vbot.xml collision_middle_box
# 用于 footprint-contains 检测: 区域中心点是否落在机器人矩形投影内
ROBOT_HALF_X = 0.25  # 前后半长 (含head_box+bumper, 实际0.2685/0.2551, 用0.25)
ROBOT_HALF_Y = 0.15  # 左右半宽 (含thigh motor, 实际0.163, 用0.15)


@registry.env("vbot_navigation_section011", "np")
class VBotSection011Env(NpEnv):
    """
    VBot Section01 多航点导航 + 跳跃庆祝
    地形: hfield + 15度坡道 + 高台(顶面z=1.294), 起点z=0
    """
    _cfg: VBotSection011EnvCfg

    def __init__(self, cfg: VBotSection011EnvCfg, num_envs: int = 1):
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
        # v20: 观测空间69维 = 54(base) + 3(trunk_acc) + 12(actuator_torques)
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
        self.spawn_x_range = (pr[0], pr[2])  # (x_min, x_max) for per-axis randomization
        self.spawn_y_range = (pr[1], pr[3])  # (y_min, y_max) — set to (0,0) for fixed Y

        self.navigation_stats_step = 0

        # 初始化得分区 & 航点
        self._init_scoring_zones(cfg)
        self._init_waypoints(cfg)

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
        return self._terrain_scale.update(info, probe_y, self._num_envs)

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
        # 基座接触终止统一使用 base_contact 传感器 (见 _compute_terminated)
        # 旧的 geom-pair 检测 (_init_termination_contact) 已移除
        self._init_foot_contact()

    def _init_foot_contact(self):
        self.foot_sensor_names = ["FR_foot_contact", "FL_foot_contact", "RR_foot_contact", "RL_foot_contact"]
        self.num_foot_check = 4
        # Gait tracking buffers (initialized per-env in reset)
        self._gait_phase_counter = None  # steps since last contact switch per foot
        # 关节raw PD扭矩归一化 (含饱和信息)
        # motrixsim不支持jointactuatorfrc传感器, 改用raw PD输出 (unclipped by forcerange)
        self.torque_limits = np.array([17, 17, 34, 17, 17, 34, 17, 17, 34, 17, 17, 34], dtype=np.float32)

    def _init_scoring_zones(self, cfg):
        sz = cfg.scoring_zones
        self.smiley_centers = np.array(sz.smiley_centers, dtype=np.float32)
        self.red_packet_centers = np.array(sz.red_packet_centers, dtype=np.float32)
        self.celebration_center = np.array(sz.celebration_center, dtype=np.float32)
        self.celebration_radius = sz.celebration_radius
        self.celebration_min_z = sz.celebration_min_z
        self.has_scoring_zones = True
        self.num_smileys = len(sz.smiley_centers)
        self.num_red_packets = len(sz.red_packet_centers)
        # 预计算zone分类 (left/middle/right by x-coordinate)
        # 笑脸
        smiley_xs = self.smiley_centers[:, 0]
        self._smiley_middle_idx = int(np.argmin(np.abs(smiley_xs)))
        side_mask = np.ones(self.num_smileys, dtype=bool)
        side_mask[self._smiley_middle_idx] = False
        side_idxs = np.where(side_mask)[0]
        self._smiley_left_idx = int(side_idxs[0] if smiley_xs[side_idxs[0]] < smiley_xs[side_idxs[1]] else side_idxs[1])
        self._smiley_right_idx = int(side_idxs[0] if smiley_xs[side_idxs[0]] > smiley_xs[side_idxs[1]] else side_idxs[1])
        # 红包
        rp_xs = self.red_packet_centers[:, 0]
        self._rp_middle_idx = int(np.argmin(np.abs(rp_xs)))
        rp_side_mask = np.ones(self.num_red_packets, dtype=bool)
        rp_side_mask[self._rp_middle_idx] = False
        rp_side_idxs = np.where(rp_side_mask)[0]
        self._rp_left_idx = int(rp_side_idxs[0] if rp_xs[rp_side_idxs[0]] < rp_xs[rp_side_idxs[1]] else rp_side_idxs[1])
        self._rp_right_idx = int(rp_side_idxs[0] if rp_xs[rp_side_idxs[0]] > rp_xs[rp_side_idxs[1]] else rp_side_idxs[1])
        # 碰撞检测模式: radius==0 → 机器人矩形足印包含区域中心 (精确); radius>0 → 距离阈值
        self.smiley_radius = sz.smiley_radius
        self.red_packet_radius = sz.red_packet_radius
        self._use_footprint_smileys = (sz.smiley_radius <= 0.0)
        self._use_footprint_rp = (sz.red_packet_radius <= 0.0)

    def _init_waypoints(self, cfg):
        """初始化分阶段区域收集导航系统
        竞赛规则:
          Phase 0 (COLLECT_SMILEYS): 收集全部3个笑脸区 (任意顺序)
          Phase 1 (COLLECT_RED_PACKETS): 全部笑脸后, 收集全部3个红包区 (任意顺序)
          Phase 2 (CLIMB_TO_PLATFORM): 全部红包后, 爬到高台庆祝区
          Phase 3 (CELEBRATION): 在高台上跳跃庆祝
        wp_idx = smileys_collected + red_packets_collected + platform_reached
        num_waypoints = 7 (3+3+1), 当 wp_idx >= 7 进入庆祝阶段
        """
        # 阶段常量
        self.PHASE_APPROACH = -1     # START平台 → 接近bump区
        self.PHASE_SMILEYS = 0
        self.PHASE_RED_PACKETS = 1
        self.PHASE_CLIMB = 2
        self.PHASE_CELEBRATION = 3
        self.NUM_WAYPOINTS = 7  # 航点总数 = 3笑脸 + 3红包 + 1高台

        # 庆祝参数 (多次跳跃庆祝)
        wn = cfg.waypoint_nav
        self.wp_radius = wn.waypoint_radius
        self.wp_final_radius = wn.final_radius
        self.celeb_jump_threshold = getattr(wn, 'celebration_jump_threshold', 1.55)
        self.required_jumps = getattr(wn, 'required_jumps', 3)
        self.celeb_landing_z = getattr(wn, 'celebration_landing_z', 1.50)
        print(f"[Info] 分阶段导航: {self.num_smileys}笑脸 → {self.num_red_packets}红包 → 高台 → 庆祝")

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
        """获取实际力矩"""
        return data.actuator_ctrls

    def _compute_swing_contact_penalty(self, data: mtx.SceneData, joint_vel: np.ndarray) -> np.ndarray:
        foot_forces = self._get_foot_contact_forces(data)
        force_magnitudes = np.linalg.norm(foot_forces, axis=2)
        
        calf_indices = [2, 5, 8, 11]
        foot_vel = np.abs(joint_vel[:, calf_indices])
        # motrixsim 0.5+: contact sensors return unit normal (mag 0 or 1), not force in N
        has_contact = force_magnitudes > 0.5
        has_high_vel = foot_vel > 2.0
        swing_contact = np.logical_and(has_contact, has_high_vel).astype(np.float32)
        penalty = np.sum(swing_contact * np.square(foot_vel) / 10.0, axis=1)  # quadratic: heavier at high vel
        return penalty

    # ============================================================
    # 步态质量奖励 (Gait Quality Rewards)
    # ============================================================

    def _compute_gait_rewards(self, data: mtx.SceneData) -> dict:
        """Compute stance ratio reward: ~2 feet on ground is ideal for blind terrain traversal."""
        foot_forces = self._get_foot_contact_forces(data)  # [n, 4, 3]
        force_mag = np.linalg.norm(foot_forces, axis=2)    # [n, 4]
        # motrixsim 0.5+: contact sensors return unit normal (mag 0 or 1)
        in_contact = (force_mag > 0.5).astype(np.float32)  # [n, 4] binary

        # Stance Ratio: ~2 feet on ground is ideal for blind terrain traversal
        total_contacts = np.sum(in_contact, axis=1)  # [n]
        # Ideal: 2 feet on ground. Penalize 0 (airborne) or 4 (all feet planted = not walking)
        stance_raw = 1.0 - np.abs(total_contacts - 2.0) * 0.33
        stance_reward = np.clip(stance_raw, 0.0, 1.0)

        return {
            "stance_reward": stance_reward,
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
        """将角度归一化到[-pi, pi]"""
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _footprint_contains_point(center_2d, robot_xy, robot_heading):
        """检测区域中心点是否落在机器人矩形足印(XY平面投影)内。

        将区域中心变换到机器人局部坐标系, 检查是否在trunk半尺寸内。

        Args:
            center_2d: [2] 区域中心坐标 (x, y)
            robot_xy: [n, 2] 机器人位置
            robot_heading: [n] 机器人航向角 (yaw)

        Returns:
            [n] bool — True 表示机器人足印包含该中心点
        """
        # 相对偏移 (世界坐标)
        dx = center_2d[0] - robot_xy[:, 0]  # [n]
        dy = center_2d[1] - robot_xy[:, 1]  # [n]
        # 旋转到机器人局部坐标 (反向旋转 heading)
        cos_h = np.cos(robot_heading)
        sin_h = np.sin(robot_heading)
        local_x = dx * cos_h + dy * sin_h   # 前后方向
        local_y = -dx * sin_h + dy * cos_h  # 左右方向
        # 检查是否在矩形半尺寸内
        return (np.abs(local_x) <= ROBOT_HALF_X) & (np.abs(local_y) <= ROBOT_HALF_Y)

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
    # 可视化辅助
    # ============================================================

    @staticmethod
    def _sanitize_dof_quaternions(dof_pos, quat_indices):
        """归一化dof_pos中指定位置的四元数，防止Rust panic"""
        for start in quat_indices:
            q = dof_pos[:, start:start+4]
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            # bad: NaN, Inf, 或极小模长
            bad = (norms < 1e-6).flatten() | (~np.isfinite(norms)).flatten()
            if np.any(bad):
                dof_pos[bad, start:start+4] = np.array([0, 0, 0, 1], dtype=np.float32)
            good = ~bad
            if np.any(good):
                dof_pos[good, start:start+4] = q[good] / norms[good]
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
            # 归一化所有已知的四元数位置: 机器人基座, 两个箭头
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
            pass  # 可视化失败不影响训练

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
            pass  # 可视化失败不影响训练

    # ============================================================
    # 航点 & 庆祝状态转换
    # ============================================================

    def _update_waypoint_state(self, info, robot_xy, robot_heading, current_z):
        """
        分阶段区域收集 & 庆祝状态机。

        竞赛规则:
          Phase 0: 收集全部3个笑脸 (任意顺序, 进入半径内即收集)
          Phase 1: 全部笑脸后收集全部3个红包 (任意顺序, 门控于全部笑脸)
          Phase 2: 全部红包后爬到高台 (z > celebration_min_z)
          Phase 3: 跳跃庆祝 (JUMP → DONE)

        wp_idx = smileys_count + red_packets_count + platform_reached (0-7)
        当 wp_idx >= 7 (num_waypoints) → 进入庆祝阶段
        """
        nav_phase = info["nav_phase"]
        celeb_state = info["celeb_state"]

        n = self._num_envs
        scales = self._cfg.reward_config.scales

        zone_bonus = np.zeros(n, dtype=np.float32)
        smiley_bonus_tb = np.zeros(n, dtype=np.float32)  # TensorBoard individual tracking
        red_packet_bonus_tb = np.zeros(n, dtype=np.float32)  # TensorBoard individual tracking
        phase_bonus = np.zeros(n, dtype=np.float32)
        celeb_bonus = np.zeros(n, dtype=np.float32)
        jump_reward = np.zeros(n, dtype=np.float32)

        # --- Phase APPROACH → 0: 进入bump区 (y >= -1.5) ---
        in_approach = (nav_phase == self.PHASE_APPROACH)
        entered_bump = in_approach & (robot_xy[:, 1] >= -1.5)
        if np.any(entered_bump):
            nav_phase = np.where(entered_bump, self.PHASE_SMILEYS, nav_phase)
            phase_bonus += np.where(entered_bump, scales.get("phase_bonus", 15.0), 0.0)

        # --- Phase 0 & 1: 笑脸收集 (Phase-independent: 任何阶段都可收集笑脸) ---
        # footprint-contains检测: 机器人矩形足印(XY平面投影)包含区域中心点
        smileys_reached = info["smileys_reached"]
        can_collect_smileys = (nav_phase <= self.PHASE_RED_PACKETS)  # Phase 0 + Phase 1
        if np.any(can_collect_smileys) and self.has_scoring_zones:
            for i in range(self.num_smileys):
                if self._use_footprint_smileys:
                    touched = self._footprint_contains_point(
                        self.smiley_centers[i], robot_xy, robot_heading
                    )
                else:
                    d = np.linalg.norm(robot_xy - self.smiley_centers[i][:2], axis=1)
                    touched = d < self.smiley_radius
                first_collect = can_collect_smileys & touched & ~smileys_reached[:, i]
                smileys_reached[:, i] |= (can_collect_smileys & touched)
                smiley_val = np.where(first_collect, scales.get("waypoint_bonus", 20.0), 0.0)
                zone_bonus += smiley_val
                smiley_bonus_tb += smiley_val
        info["smileys_reached"] = smileys_reached

        # Phase 0 → 1: 全部3个笑脸收集完成 (v16: strict ALL for full score 20/20)
        all_smileys = np.all(smileys_reached, axis=1)
        in_phase0 = (nav_phase == self.PHASE_SMILEYS)
        phase0_to_1 = in_phase0 & all_smileys
        if np.any(phase0_to_1):
            nav_phase = np.where(phase0_to_1, self.PHASE_RED_PACKETS, nav_phase)
            phase_bonus += np.where(phase0_to_1, scales.get("phase_bonus", 15.0), 0.0)

        # --- Phase 1: 红包收集 (门控: 全部笑脸已收集) ---
        red_packets_reached = info["red_packets_reached"]
        in_phase1 = (nav_phase == self.PHASE_RED_PACKETS)
        if np.any(in_phase1) and self.has_scoring_zones:
            for i in range(self.num_red_packets):
                if self._use_footprint_rp:
                    touched = self._footprint_contains_point(
                        self.red_packet_centers[i], robot_xy, robot_heading
                    )
                else:
                    d = np.linalg.norm(robot_xy - self.red_packet_centers[i][:2], axis=1)
                    touched = d < self.red_packet_radius
                first_collect = in_phase1 & touched & ~red_packets_reached[:, i]
                red_packets_reached[:, i] |= (in_phase1 & touched)
                rp_val = np.where(first_collect, scales.get("waypoint_bonus", 20.0), 0.0)
                zone_bonus += rp_val
                red_packet_bonus_tb += rp_val
        info["red_packets_reached"] = red_packets_reached

        # Phase 1 → 2: 全部红包收集完成
        all_red_packets = np.all(red_packets_reached, axis=1)
        phase1_to_2 = in_phase1 & all_red_packets
        if np.any(phase1_to_2):
            nav_phase = np.where(phase1_to_2, self.PHASE_CLIMB, nav_phase)
            phase_bonus += np.where(phase1_to_2, scales.get("phase_bonus", 15.0), 0.0)

        # --- Phase 2: 爬到高台 ---
        platform_reached = info.get("platform_reached", np.zeros(n, dtype=bool))
        in_phase2 = (nav_phase == self.PHASE_CLIMB)
        if np.any(in_phase2):
            celeb_xy = self.celebration_center[:2]
            d_celeb = np.linalg.norm(robot_xy - celeb_xy[np.newaxis, :], axis=1)
            arrived = in_phase2 & (d_celeb < self.wp_final_radius) & (current_z > self.celebration_min_z)
            first_arrive = arrived & ~platform_reached
            platform_reached |= arrived
            zone_bonus += np.where(first_arrive, scales.get("phase_bonus", 15.0), 0.0)
            # Phase 2 → 3
            nav_phase = np.where(arrived, self.PHASE_CELEBRATION, nav_phase)
        info["platform_reached"] = platform_reached

        # --- 更新 wp_idx (进度指标, 取代旧的顺序航点索引) ---
        smileys_count = np.sum(smileys_reached, axis=1).astype(np.int32)
        red_packets_count = np.sum(red_packets_reached, axis=1).astype(np.int32)
        wp_idx = smileys_count + red_packets_count + platform_reached.astype(np.int32)
        info["wp_idx"] = wp_idx
        info["nav_phase"] = nav_phase

        # --- Phase 3: 多次跳跃庆祝 (v27: 跳N次) ---
        in_celeb = (nav_phase == self.PHASE_CELEBRATION)
        jump_count = info["jump_count"]

        # IDLE -> JUMP (进入庆祝阶段)
        start_celeb = in_celeb & (celeb_state == CELEB_IDLE)
        if np.any(start_celeb):
            celeb_state = np.where(start_celeb, CELEB_JUMP, celeb_state)

        # JUMP: 奖励向上运动, 检测跳跃峰值
        jumping = in_celeb & (celeb_state == CELEB_JUMP)
        if np.any(jumping):
            # 连续奖励: z越高越好 (platform standing z ≈ 1.55)
            z_above_standing = np.maximum(current_z - 1.5, 0.0)
            jump_reward += np.where(jumping, scales.get("jump_reward", 8.0) * z_above_standing, 0.0)

            # 跳跃检测: z超过阈值 → 转入LANDING等待落地
            jumped = jumping & (current_z > self.celeb_jump_threshold)
            if np.any(jumped):
                jump_count = np.where(jumped, jump_count + 1, jump_count)
                # 每次跳跃给一次性奖励
                celeb_bonus += np.where(jumped, scales.get("per_jump_bonus", 15.0), 0.0)
                # 检查是否完成所有跳跃
                all_done = jumped & (jump_count >= self.required_jumps)
                still_jumping = jumped & (jump_count < self.required_jumps)
                celeb_state = np.where(all_done, CELEB_DONE, celeb_state)
                celeb_state = np.where(still_jumping, CELEB_LANDING, celeb_state)
                # 完成全部跳跃的额外大奖
                celeb_bonus += np.where(all_done, scales.get("celebration_bonus", 50.0), 0.0)

        # LANDING: 等待落地 (z回到standing以下), 然后重新进入JUMP
        landing = in_celeb & (celeb_state == CELEB_LANDING)
        if np.any(landing):
            landed = landing & (current_z < self.celeb_landing_z)
            if np.any(landed):
                celeb_state = np.where(landed, CELEB_JUMP, celeb_state)

        info["jump_count"] = jump_count
        info["celeb_state"] = celeb_state

        # zone_bonus 包含笑脸+红包+高台一次性奖励
        # phase_bonus 包含阶段完成奖励
        wp_bonus = zone_bonus + phase_bonus
        info["_smiley_bonus_tb"] = smiley_bonus_tb
        info["_red_packet_bonus_tb"] = red_packet_bonus_tb
        info["_phase_bonus_tb"] = phase_bonus
        return info, wp_bonus, celeb_bonus, jump_reward

    def _get_current_target(self, info, robot_xy):
        """v25 Ordered targeting: nearest side → middle → far side.

        Based on robot's current X position:
          x < 0  → Left → Middle → Right
          x >= 0 → Right → Middle → Left

        Picks the first uncollected zone in this priority order.
        No state stored — purely determined by robot position + reached flags.
        """
        nav_phase = info["nav_phase"]
        n = len(nav_phase)
        celeb_xy = self.celebration_center[:2]
        targets = np.tile(celeb_xy, (n, 1))

        if not self.has_scoring_zones:
            return targets

        def _apply_ordered_targets(env_mask, reached, centers,
                                    left_idx, middle_idx, right_idx):
            """Pick first uncollected zone in order: nearest_side → middle → far_side."""
            p = np.where(env_mask)[0]
            if len(p) == 0:
                return

            robot_x = robot_xy[p, 0]  # [K]
            on_left = robot_x < 0     # [K] bool

            # Priority order per env: [near_side, middle, far_side]
            first = np.where(on_left, left_idx, right_idx)    # near side
            second = np.full(len(p), middle_idx, dtype=np.int32)  # middle always second
            third = np.where(on_left, right_idx, left_idx)    # far side
            priority = np.stack([first, second, third], axis=1)  # [K, 3]

            # Check reached status for each priority slot
            reached_p = reached[p]  # [K, num_zones]
            reached_by_priority = np.array([
                reached_p[np.arange(len(p)), priority[:, r]] for r in range(3)
            ]).T  # [K, 3]

            uncollected = ~reached_by_priority  # [K, 3]
            has_any = np.any(uncollected, axis=1)  # [K]

            # argmax on bool gives index of first True = first uncollected
            first_uncollected_rank = np.argmax(uncollected, axis=1)  # [K]
            target_zone = priority[np.arange(len(p)), first_uncollected_rank]  # [K]

            # Set targets for envs that still have uncollected zones
            assign = p[has_any]
            targets[assign] = centers[target_zone[has_any]]

        # Phase APPROACH + Phase 0: 目标 = 最近笑脸 (approach阶段也朝笑脸方向走)
        _apply_ordered_targets(
            (nav_phase == self.PHASE_APPROACH) | (nav_phase == self.PHASE_SMILEYS),
            info["smileys_reached"],
            self.smiley_centers,
            self._smiley_left_idx, self._smiley_middle_idx, self._smiley_right_idx)

        # Phase 1: 红包 (nearest side → middle → far side)
        _apply_ordered_targets(
            nav_phase == self.PHASE_RED_PACKETS, info["red_packets_reached"],
            self.red_packet_centers,
            self._rp_left_idx, self._rp_middle_idx, self._rp_right_idx)

        return targets

    # ============================================================
    # apply_action & torques
    # ============================================================

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # 安全防护: NaN/Inf actions → 归零(保持上一帧动作)
        actions = np.where(np.isfinite(actions), actions, 0.0)
        actions = np.clip(actions, -1.0, 1.0)  # 动作空间严格裁剪
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
        # PD控制: kp=100 提供足够关节刚度支撑9kg体重
        # kp×action_scale=50Nm 虽超过hip forcerange(17Nm), 但高kp对姿态稳定至关重要
        # 大动作时clip到forcerange(bang-bang), 小动作(<0.17rad)仍为线性响应
        kp, kv = 100.0, 8.0
        torques = kp * (target_pos - current_pos) - kv * current_vel
        # v20: 保存raw PD输出 (unclipped), 供obs和reward使用
        # data.actuator_ctrls仍然clip到forcerange用于物理引擎
        self._raw_torques = torques.copy()
        # 与XML forcerange对齐: hip/thigh=±17Nm, calf=±34Nm
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)
        return np.clip(torques, -torque_limits, torque_limits)

    # ============================================================
    # update_state - 核心循环
    # ============================================================

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        cfg = self._cfg
        info = state.info

        # --- 获取物理状态 ---
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)

        # --- NaN安全防护: 物理引擎偶发NaN, 必须在所有计算之前清理 ---
        nan_mask = (
            np.any(~np.isfinite(joint_vel), axis=1)
            | np.any(~np.isfinite(joint_pos), axis=1)
            | np.any(~np.isfinite(root_pos), axis=1)
            | np.any(~np.isfinite(root_vel), axis=1)
        )
        if np.any(nan_mask):
            # 用安全默认值替换NaN envs (这些env会被terminated)
            joint_vel = np.where(nan_mask[:, np.newaxis], 0.0, joint_vel)
            joint_pos = np.where(nan_mask[:, np.newaxis], self.default_angles, joint_pos)
            root_vel = np.where(nan_mask[:, np.newaxis], 0.0, root_vel)
            root_pos = np.where(nan_mask[:, np.newaxis], np.array([[0.0, -2.5, 0.5, 0.0, 0.0, 0.0]]), root_pos)
            root_quat = np.where(nan_mask[:, np.newaxis], np.array([[1.0, 0.0, 0.0, 0.0]]), root_quat)
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
        info, wp_bonus, celeb_bonus, jump_reward = \
            self._update_waypoint_state(info, robot_xy, robot_heading, current_z)

        # --- 当前导航目标 (基于阶段 + 最近未收集区域) ---
        target_xy = self._get_current_target(info, robot_xy)
        in_celeb = (info["nav_phase"] >= self.PHASE_CELEBRATION)

        # 导航命令
        position_error = target_xy - robot_xy
        distance_to_target = np.linalg.norm(position_error, axis=1)

        # 更新pose_commands供可视化
        pose_commands = np.column_stack([target_xy, np.zeros(self._num_envs, dtype=np.float32)])
        info["pose_commands"] = pose_commands

        # 到达当前目标? (阶段感知半径)
        current_wp_radius = np.where(
            info["nav_phase"] >= self.PHASE_CLIMB,
            self.wp_final_radius,
            self.wp_radius
        )
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

        velocity_commands = np.column_stack([desired_vel_xy, desired_yaw_rate[:, np.newaxis] if desired_yaw_rate.ndim == 1 else desired_yaw_rate])

        # 朝向误差 (朝向当前导航目标, 而非固定heading=0)
        heading_diff = self._wrap_angle(desired_heading - robot_heading)

        # --- 观测 ---
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        last_actions = info["current_actions"]

        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi

        # v19: 盲走越障关键感知特征 (foot_contact + base_height)
        foot_forces_obs = self._get_foot_contact_forces(data)
        foot_contact = (np.linalg.norm(foot_forces_obs, axis=2) > 0.5).astype(np.float32)  # [n, 4]
        base_height_norm = np.clip((current_z - 0.5) / 1.2, -1.0, 1.0)[:, np.newaxis]      # [n, 1]

        # v20: trunk加速度计 (冲击/地形感知) + 关节raw扭矩需求 (力觉反馈, 含饱和信息)
        trunk_acc_raw = self._get_trunk_acc(data)                                            # [n, 3]
        trunk_acc_norm = np.clip(trunk_acc_raw / 20.0, -3.0, 3.0)                           # 归一化: 重力≈0.5, clip at ±60m/s²
        actual_torques = self._get_actuator_torques(data)                                    # [n, 12] raw unclipped PD
        torques_normalized = actual_torques / self.torque_limits[np.newaxis, :]               # can exceed ±1 when saturated

        # 庆祝进度观测: 0=导航中, 0.5=在平台跳跃中, 1.0=跳跃完成
        celeb_progress = info["celeb_state"].astype(np.float32) / 2.0

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
            trunk_acc_norm,                            # 3  (v20)
            torques_normalized,                        # 12 (v20)
        ], axis=-1)
        assert obs.shape == (data.shape[0], 69), f"obs shape {obs.shape} != (N, 69)"

        # 可视化
        self._update_target_marker(data, pose_commands)
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel[:, :2])

        # --- 步数追踪 ---
        # NOTE: 不在这里自增 info["steps"], 基类 NpEnv.step() 已经做了 info["steps"] += 1
        # 之前的双重自增导致 grace_period 和 max_episode_steps 实际减半

        # --- 终止 (包括摔倒+超出边界) ---
        terminated_state = self._compute_terminated(state, projected_gravity, joint_vel, robot_xy, current_z)
        terminated = terminated_state.terminated

        # --- 奖励 ---
        reward = self._compute_reward(
            data, info, velocity_commands, base_lin_vel, gyro, projected_gravity,
            joint_vel, distance_to_target, position_error, reached_wp,
            terminated, robot_heading, robot_xy, current_z, speed_xy,
            wp_bonus, celeb_bonus, jump_reward, in_celeb
        )

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        # 庆祝完成后截断
        celeb_done = (info["celeb_state"] == CELEB_DONE)
        self._success_truncate = celeb_done

        # v44: 停滞检测 — 若机器人长时间不动则截断
        stag_cfg_window = getattr(cfg, 'stagnation_window_steps', 1000)
        stag_cfg_dist = getattr(cfg, 'stagnation_min_distance', 0.5)
        stag_cfg_grace = getattr(cfg, 'stagnation_grace_steps', 500)
        ep_steps = info.get("steps", np.zeros(self._num_envs, dtype=np.int32))
        anchor_xy = info["stagnation_anchor_xy"]
        anchor_step = info["stagnation_anchor_step"]
        dist_from_anchor = np.linalg.norm(robot_xy - anchor_xy, axis=1)
        # 更新锚点: 机器人移动足够远就刷新锚点
        moved_enough = dist_from_anchor >= stag_cfg_dist
        info["stagnation_anchor_xy"] = np.where(moved_enough[:, np.newaxis], robot_xy, anchor_xy)
        info["stagnation_anchor_step"] = np.where(moved_enough, ep_steps, anchor_step)
        # 检测停滞: 超过窗口时间仍未移动, 且不在庆祝阶段, 且过了grace期
        steps_since_anchor = ep_steps - info["stagnation_anchor_step"]
        stagnant = (
            (steps_since_anchor > stag_cfg_window)
            & (ep_steps >= stag_cfg_grace)
            & ~in_celeb
        )
        self._stagnation_truncate = stagnant

        # 诊断指标: 机器人是否进入bump区 + 最远Y推进
        current_y = robot_xy[:, 1]
        in_bump_zone = ((current_y >= -1.5) & (current_y <= 1.5)).astype(np.float32)
        max_y_reached = info.get("max_y_reached", current_y.copy())
        max_y_reached = np.maximum(max_y_reached, current_y)
        info["max_y_reached"] = max_y_reached

        state.info["metrics"] = {
            "distance_to_target": distance_to_target,
            "reached_fraction": reached_wp.astype(np.float32),
            "wp_idx_mean": info["wp_idx"].astype(np.float32),
            "nav_phase_mean": info["nav_phase"].astype(np.float32),
            "celeb_state_mean": info["celeb_state"].astype(np.float32),
            "jump_count_mean": info["jump_count"].astype(np.float32),
            "action_scale_mean": info["current_action_scale"].astype(np.float32).reshape(-1),
            "bump_entry_frac": in_bump_zone,
            "max_y_progress": max_y_reached,
        }
        return state

    def _update_truncate(self):
        super()._update_truncate()
        if hasattr(self, '_success_truncate'):
            self._state.truncated = np.logical_or(self._state.truncated, self._success_truncate)
        if hasattr(self, '_stagnation_truncate'):
            self._state.truncated = np.logical_or(self._state.truncated, self._stagnation_truncate)

    # ============================================================
    # 终止条件
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
            hard_terminated |= tilt_angle > np.deg2rad(70)  # 70°: 明确摔倒, 坡道15°不受影响

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
        try:
            base_contact_value = self._model.get_sensor_value("base_contact", data)
            # motrixsim 0.5+ returns 3D force vector (n, 3); compute magnitude
            if base_contact_value.ndim >= 2 and base_contact_value.shape[-1] == 3:
                force_mag = np.linalg.norm(base_contact_value, axis=-1)  # (n,)
                base_contact = (force_mag > 0.01).flatten()[:n]
            elif base_contact_value.ndim == 0:
                base_contact = np.array([float(base_contact_value) > 0.01], dtype=bool)
            else:
                base_contact = (base_contact_value.flatten() > 0.01)[:n]
        except Exception:
            base_contact = np.zeros(n, dtype=bool)
        soft_terminated |= base_contact

        # 中等倾斜 (50-70°) — grace期间允许恢复, 之后终止
        if projected_gravity is not None:
            gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
            gz = projected_gravity[:, 2]
            tilt_angle = np.arctan2(gxy, np.abs(gz))
            soft_terminated |= tilt_angle > np.deg2rad(50)  # 50°: 不稳定但可能恢复

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

    def _compute_reward(self, data, info, velocity_commands, base_lin_vel, gyro,
                         projected_gravity, joint_vel, distance_to_target, position_error,
                         reached_wp, terminated, robot_heading, robot_xy, current_z,
                         speed_xy, wp_bonus, celeb_bonus, jump_reward, in_celeb):
        scales = self._cfg.reward_config.scales
        n = self._num_envs

        # ===== 累积奖金追踪 (终止时得分清零用) =====
        accumulated_bonus = info.get("accumulated_bonus", np.zeros(n, dtype=np.float32))

        # ===== 导航跟踪 (仅非庆祝阶段) =====
        position_tracking = np.exp(-distance_to_target / 5.0)

        # 面朝当前航点
        target_bearing = np.arctan2(position_error[:, 1], position_error[:, 0])
        facing_error = self._wrap_angle(target_bearing - robot_heading)
        heading_tracking = np.where(reached_wp | in_celeb, 1.0, np.exp(-np.abs(facing_error) / 0.5))

        direction_to_target = position_error / (np.linalg.norm(position_error, axis=1, keepdims=True) + 1e-8)
        forward_velocity = np.clip(np.sum(base_lin_vel[:, :2] * direction_to_target, axis=1), -0.5, 0.8)

        # 航点approach: step-delta
        last_wp_dist = info.get("last_wp_distance", distance_to_target.copy())
        wp_delta = last_wp_dist - distance_to_target
        info["last_wp_distance"] = distance_to_target.copy()
        wp_approach = np.clip(wp_delta * scales.get("waypoint_approach", 200.0), -0.5, 2.5)

        # 面朝航点奖励
        wp_facing = scales.get("waypoint_facing", 0.61) * heading_tracking

        # 存活奖励 (分段式: 明确区分站立/倾斜/摔倒)
        # gz>0.9 (tilt<26°,含15°坡道): 全额; gz>0.7 (tilt<45°): 半额; gz<0.7: 零
        gz = np.clip(-projected_gravity[:, 2], 0.0, 1.0)
        upright_factor = np.where(gz > 0.9, 1.0, np.where(gz > 0.7, 0.5, 0.0))
        # v39: time-decay — alive_bonus线性衰减至0, 防止"存活即奖"的lazy robot exploit
        alive_decay_horizon = scales.get("alive_decay_horizon", 3000.0)
        ep_steps = info.get("steps", np.zeros(n, dtype=np.int32)).astype(np.float32)
        alive_time_decay = np.clip(1.0 - ep_steps / alive_decay_horizon, 0.0, 1.0)
        alive_bonus = scales.get("alive_bonus", 0.05) * upright_factor * alive_time_decay

        # ===== 稳定性惩罚 (数值安全: 先clip再平方, 防止NaN溢出) =====
        # Standard orientation penalty (equal roll + pitch)
        # v17: 恢复 -0.05 强度 (防摔), 但添加坡道补偿奖励
        orientation_penalty = np.sum(np.square(projected_gravity[:, :2]), axis=1)

        # v17: 坡道方向补偿 — 在坡道上(y∈[2.0,7.0])期望pitch≈15°
        # 当机器人正确倾斜时, 补偿抵消orientation惩罚
        current_y_for_slope = robot_xy[:, 1]
        on_ramp = (current_y_for_slope > 2.0) & (current_y_for_slope < 7.0)
        expected_gy = np.where(on_ramp, 0.259, 0.0)  # sin(15°) ≈ 0.259
        # 补偿: 实际gy接近expected_gy时给正奖励
        gy_error = np.abs(projected_gravity[:, 1] - expected_gy)
        slope_compensation = np.where(on_ramp, np.exp(-np.square(gy_error) / 0.05), 0.0)
        slope_orientation_reward = scales.get("slope_orientation", 0.04) * slope_compensation
        lin_vel_z_penalty = np.square(np.clip(base_lin_vel[:, 2], -50.0, 50.0))
        ang_vel_xy_penalty = np.sum(np.square(np.clip(gyro[:, :2], -50.0, 50.0)), axis=1)
        # torque_penalty: 使用clipped扭矩 (物理引擎实际施加的力矩)
        clipped_torques = self._get_actuator_torques(data)
        torque_penalty = np.sum(np.square(np.clip(clipped_torques, -200.0, 200.0)), axis=1)
        safe_joint_vel = np.clip(joint_vel, -100.0, 100.0)
        dof_vel_penalty = np.sum(np.square(safe_joint_vel), axis=1)
        last_dof_vel = info.get("last_dof_vel", np.zeros_like(joint_vel))
        dof_acc_penalty = np.sum(np.square(safe_joint_vel - np.clip(last_dof_vel, -100.0, 100.0)), axis=1)
        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        # v20: 冲击惩罚 (trunk加速度计 — 感知身体受到的冲击/碰撞)
        trunk_acc_r = self._get_trunk_acc(data)
        trunk_acc_mag = np.linalg.norm(trunk_acc_r, axis=1)
        # 正常行走加速度 ≈ 10-15 m/s² (含重力), 超过阈值视为冲击
        impact_excess = np.maximum(trunk_acc_mag - 15.0, 0.0)
        impact_penalty = np.square(impact_excess) / 100.0  # 温和二次惩罚

        # v20: 扭矩饱和惩罚 (raw PD需求超过forcerange极限 → 控制饱和)
        raw_torques = self._raw_torques  # unclipped PD output, stored in _compute_torques
        saturation_ratio = np.abs(raw_torques) / self.torque_limits[np.newaxis, :]
        torque_sat_penalty = np.sum(np.maximum(saturation_ratio - 0.9, 0.0) ** 2, axis=1)

        # ===== 爬坡高度进步 =====
        last_z = info.get("last_z", current_z.copy())
        z_delta = current_z - last_z
        info["last_z"] = current_z.copy()
        height_progress = scales.get("height_progress", 8.0) * np.maximum(z_delta, 0.0)

        # ===== v17: 目标高度接近奖励 (减少 |z_target - z_robot|) =====
        # 根据当前目标y位置估算目标z高度
        target_xy_for_z = position_error + robot_xy  # recover target_xy
        target_y_for_z = target_xy_for_z[:, 1]
        target_z_est = np.where(
            target_y_for_z < 1.5, 0.15,  # 高度场区域: z≈0-0.277, 均值≈0.15
            np.where(target_y_for_z < 7.0,
                     (target_y_for_z - 1.5) / (7.0 - 1.5) * 1.294,  # 坡道线性插值
                     1.294)  # 高台顶面
        )
        z_error = np.abs(current_z - target_z_est)
        last_z_error = info.get("last_z_error", z_error.copy())
        z_error_delta = last_z_error - z_error  # 正 = 接近目标高度
        info["last_z_error"] = z_error.copy()
        height_approach = scales.get("height_approach", 5.0) * np.clip(z_error_delta, -0.1, 0.5)

        # ===== v17: 高度振荡惩罚 (penalize rapid z bouncing) =====
        z_osc = np.abs(z_delta)
        # 正常行走z变化 < 0.015m/step, 超过阈值的部分视为振荡
        height_osc_penalty = scales.get("height_oscillation", -2.0) * np.maximum(z_osc - 0.015, 0.0)

        # ===== 地形里程碑 =====
        current_y = robot_xy[:, 1]
        milestones_reached = info.get("milestones_reached", np.zeros((n, 2), dtype=bool))
        traversal_total = np.zeros(n, dtype=np.float32)
        m1 = (current_y > 4.0) & (current_z > 0.3)
        m1_first = m1 & ~milestones_reached[:, 0]
        milestones_reached[:, 0] |= m1
        traversal_total += np.where(m1_first, 15.0, 0.0)
        m2 = (current_y > 6.5) & (current_z > 0.8)
        m2_first = m2 & ~milestones_reached[:, 1]
        milestones_reached[:, 1] |= m2
        traversal_total += np.where(m2_first, 15.0, 0.0)
        info["milestones_reached"] = milestones_reached

        # ===== 竞赛得分区 (收集逻辑已移至 _update_waypoint_state, 此处仅统计TensorBoard) =====
        # zone_bonus 和 phase_bonus 已包含在 wp_bonus 中
        smiley_total = info.get("_smiley_bonus_tb", np.zeros(n, dtype=np.float32))
        red_packet_total = info.get("_red_packet_bonus_tb", np.zeros(n, dtype=np.float32))
        phase_bonus_total = info.get("_phase_bonus_tb", np.zeros(n, dtype=np.float32))

        # ===== Zone吸引力: 当前阶段未收集zone产生接近delta奖励 =====
        # Delta-based: 只奖励靠近zone的运动, 站着不动=0 (Anti-Lazy)
        # Stage 1B: Phase-independent — 吸引所有未收集zone, 允许路过收集
        zone_approach_reward = np.zeros(n, dtype=np.float32)
        zone_approach_scale = scales.get("zone_approach", 3.0)
        if self.has_scoring_zones and zone_approach_scale > 0:
            smileys_reached = info["smileys_reached"]
            red_packets_reached = info["red_packets_reached"]

            # 笑脸zone吸引力 (Phase 0 + 1: 仅在未爬坡前有效)
            pre_climb = (info["nav_phase"] <= self.PHASE_RED_PACKETS)
            if np.any(pre_climb):
                for i in range(self.num_smileys):
                    d = np.linalg.norm(robot_xy - self.smiley_centers[i][np.newaxis, :], axis=1)
                    last_key = f"last_smiley_dist_{i}"
                    last_d = info.get(last_key, d.copy())
                    delta = last_d - d  # 正 = 靠近
                    in_range = d < 5.0  # v16: wider range for side-smiley pull (was 3.5)
                    uncollected = ~smileys_reached[:, i]
                    active = pre_climb & in_range & uncollected
                    delta_reward = np.clip(delta * zone_approach_scale * 10.0, -0.3, 2.0)  # Stage 7: raised ceiling (was -0.1,0.5) — remove artificial cap on lateral pull
                    zone_approach_reward += np.where(active, delta_reward, 0.0)
                    info[last_key] = d.copy()

            # 红包zone吸引力 (Phase 1+: 进入红包阶段后始终有效)
            post_smileys = (info["nav_phase"] >= self.PHASE_RED_PACKETS)
            if np.any(post_smileys):
                for i in range(self.num_red_packets):
                    d = np.linalg.norm(robot_xy - self.red_packet_centers[i][np.newaxis, :], axis=1)
                    last_key = f"last_rp_dist_{i}"
                    last_d = info.get(last_key, d.copy())
                    delta = last_d - d
                    in_range = d < 5.0  # v16: wider range for side-zone pull (was 3.5)
                    uncollected = ~red_packets_reached[:, i]
                    active = post_smileys & in_range & uncollected
                    delta_reward = np.clip(delta * zone_approach_scale * 10.0, -0.3, 2.0)  # Stage 7: raised ceiling (was -0.1,0.5)
                    zone_approach_reward += np.where(active, delta_reward, 0.0)
                    info[last_key] = d.copy()

        # ===== 摆动相接触惩罚 =====
        # v19: bump区降低惩罚强度, 避免在高度场因必要触地被过度惩罚
        on_bump = (current_y > -1.8) & (current_y < 1.8)
        bump_swing_scale = np.where(on_bump, scales.get("swing_contact_bump_scale", 0.6), 1.0)
        swing_penalty = (
            scales.get("swing_contact_penalty", -0.15)
            * self._compute_swing_contact_penalty(data, joint_vel)
            * bump_swing_scale
        )

        # ===== 脚部离地高度奖励 (鼓励抬脚过障碍) =====
        foot_clearance_scale = scales.get("foot_clearance", 0.0)
        if foot_clearance_scale > 0:
            foot_forces = self._get_foot_contact_forces(data)
            force_mag = np.linalg.norm(foot_forces, axis=2)  # [n, 4]
            in_swing = force_mag < 0.5  # motrixsim 0.5+: 摆动相 = 无接触 (mag < 0.5)
            calf_indices = [2, 5, 8, 11]
            calf_vel = np.abs(joint_vel[:, calf_indices])  # [n, 4]
            # v46: 三级抬脚boost — pre-bump过渡区(y>-2.5) + bump区 + 坡道前(y<2.5)
            # 解决边界处机器人不知道抬脚的问题: 在进入bump前0.5-1m就开始鼓励高抬腿
            pre_bump = (current_y > -2.5) & (current_y <= -1.5)  # 接近bump的过渡区
            bump_boost_val = scales.get("foot_clearance_bump_boost", 2.5)
            clearance_boost = np.where(
                on_bump, bump_boost_val,                    # bump区: 全额boost
                np.where(pre_bump, bump_boost_val * 0.5,    # 过渡区: 半额boost
                         1.0)                                # 其他: 无boost
            )
            clearance_scale_vec = foot_clearance_scale * clearance_boost
            # 奖励摆动相的小腿关节角速度 (鼓励积极抬腿, 而非拖地)
            foot_clearance_reward = clearance_scale_vec * np.sum(
                in_swing.astype(np.float32) * np.clip(calf_vel, 0.0, 5.0) * 0.2, axis=1
            )
        else:
            foot_clearance_reward = np.zeros(n, dtype=np.float32)

        # ===== 拖脚惩罚 (Drag-Foot Penalty) =====
        # 腿有接触 + 小腿关节角速度低 = 拖地行为 (foot_clearance的盲区)
        # 与foot_clearance互补: 后者奖励摆动相抬腿, 此处惩罚支撑相拖腿
        drag_foot_scale = scales.get("drag_foot_penalty", 0.0)
        if drag_foot_scale < 0:
            if foot_clearance_scale > 0:
                # 复用foot_clearance已计算的变量
                drag_in_contact = ~in_swing  # 支撑相 (有接触力)
            else:
                foot_forces_df = self._get_foot_contact_forces(data)
                force_mag_df = np.linalg.norm(foot_forces_df, axis=2)
                drag_in_contact = force_mag_df > 0.5
                calf_indices = [2, 5, 8, 11]
                calf_vel = np.abs(joint_vel[:, calf_indices])
            low_vel = calf_vel < 1.0  # 小腿几乎不动
            dragging = drag_in_contact & low_vel  # [n, 4]
            # bump区boost: bump区拖脚更有害
            drag_bump_boost = np.where(on_bump, 2.0, 1.0)
            drag_foot_raw = np.sum(dragging.astype(np.float32), axis=1)  # 0~4: 拖地腿数
            drag_foot_penalty = drag_foot_scale * drag_foot_raw * drag_bump_boost
        else:
            drag_foot_penalty = np.zeros(n, dtype=np.float32)

        # ===== 停滞渐进惩罚 (Stagnation Ramp Penalty) =====
        # 停滞检测只会截断episode, 不给惩罚信号 → 添加渐进惩罚
        # 随着接近停滞截断阈值, 惩罚从0线性增长 → 提供"赶紧动"的梯度
        stagnation_penalty_scale = scales.get("stagnation_penalty", 0.0)
        if stagnation_penalty_scale < 0:
            stag_window = getattr(self._cfg, 'stagnation_window_steps', 1000)
            stag_grace = getattr(self._cfg, 'stagnation_grace_steps', 500)
            stag_anchor_step = info.get("stagnation_anchor_step", np.zeros(n, dtype=np.int32))
            stag_ep_steps = info.get("steps", np.zeros(n, dtype=np.int32))
            stag_since = stag_ep_steps - stag_anchor_step
            # 从50%窗口开始线性增长: 0→0.5window=0, 0.5window→window=0→1
            stag_ratio = np.clip((stag_since.astype(np.float32) / stag_window - 0.5) * 2.0, 0.0, 1.0)
            past_grace = stag_ep_steps >= stag_grace
            stagnation_penalty = stagnation_penalty_scale * stag_ratio * past_grace.astype(np.float32)
            stagnation_penalty = np.where(in_celeb, 0.0, stagnation_penalty)  # 庆祝阶段不惩罚
        else:
            stagnation_penalty = np.zeros(n, dtype=np.float32)

        # ===== 步态质量奖励 (Gait Quality) =====
        gait = self._compute_gait_rewards(data)
        gait_stance = scales.get("stance_ratio", 0.0) * gait["stance_reward"]

        # ===== 累积奖金更新 (用于终止时得分清零) =====
        # 本步新增的所有一次性奖金 (终止的env不计入新奖金)
        step_bonus = np.where(terminated, 0.0, wp_bonus + celeb_bonus + traversal_total)
        accumulated_bonus = accumulated_bonus + step_bonus
        info["accumulated_bonus"] = accumulated_bonus

        # ===== 终止惩罚 + 得分清零 =====
        # 竞赛规则: "超出边界/摔倒行为 → 扣除本Section所有得分"
        # 训练时软化: 扣除30%累积奖金 (Stage 3: 从60%→30%, 因为smiley_bonus 150使得
        # 60%下收集笑脸的期望值 < 0, 导致机器人学会回避笑脸)
        # Cap at -100 防止高累积奖金后极端惩罚
        base_termination = scales.get("termination", -50.0)
        score_clear_penalty = np.where(terminated, np.maximum(-0.3 * accumulated_bonus, -100.0), 0.0)
        termination_penalty = np.where(terminated, base_termination, 0.0) + score_clear_penalty

        # ===== 惩罚汇总 =====
        penalties = (
            scales.get("orientation", -0.05) * orientation_penalty
            + scales.get("lin_vel_z", -0.15) * lin_vel_z_penalty
            + scales.get("ang_vel_xy", -0.02) * ang_vel_xy_penalty
            + scales.get("torques", -1e-5) * torque_penalty
            + scales.get("dof_vel", -5e-5) * dof_vel_penalty
            + scales.get("dof_acc", -2.5e-7) * dof_acc_penalty
            + scales.get("action_rate", -0.01) * action_rate_penalty
            + scales.get("impact_penalty", -0.02) * impact_penalty       # v20: trunk冲击
            + scales.get("torque_saturation", -0.01) * torque_sat_penalty # v20: 控制饱和
            + termination_penalty
            + swing_penalty
            + height_osc_penalty
            + drag_foot_penalty
            + stagnation_penalty
        )

        # ===== 综合奖励 =====
        nav_reward = (
            scales.get("position_tracking", 1.5) * position_tracking
            + wp_approach
            + wp_facing
            + scales.get("forward_velocity", 3.5) * forward_velocity
            + alive_bonus
        )
        # 庆祝期间导航奖励降低, 用旋转奖励替代
        nav_reward = np.where(in_celeb, alive_bonus, nav_reward)

        reward = (
            nav_reward
            + wp_bonus
            + celeb_bonus
            + jump_reward
            + height_progress
            + height_approach
            + slope_orientation_reward
            + traversal_total
            + zone_approach_reward
            + foot_clearance_reward
            + gait_stance
            + penalties
        )

        # 竞赛规则强制: 终止时只保留惩罚, 清零所有正向奖励
        # 摔倒/越界的step不应获得任何正向奖励
        reward = np.where(terminated, termination_penalty, reward)

        # 数值安全: 防止NaN/Inf传播到policy
        reward = np.where(np.isfinite(reward), reward, -50.0)

        # TensorBoard
        info["Reward"] = {
            "position_tracking": scales.get("position_tracking", 1.5) * position_tracking,
            "heading_tracking": wp_facing,
            "forward_velocity": scales.get("forward_velocity", 3.5) * forward_velocity,
            "wp_approach": wp_approach,
            "alive_bonus": alive_bonus,
            "wp_bonus": wp_bonus,
            "celeb_bonus": celeb_bonus,
            "jump_reward": jump_reward,
            "smiley_bonus": smiley_total,
            "red_packet_bonus": red_packet_total,
            "zone_approach": zone_approach_reward,
            "height_progress": height_progress,
            "height_approach": height_approach,
            "height_oscillation": height_osc_penalty,
            "slope_orientation": slope_orientation_reward,
            "traversal_bonus": traversal_total,
            "penalties": penalties,
            "termination": termination_penalty,
            "swing_contact_penalty": swing_penalty,
            "foot_clearance": foot_clearance_reward,
            "score_clear_penalty": score_clear_penalty,
            "phase_completion_bonus": phase_bonus_total,
            "gait_stance": gait_stance,
            "impact_penalty": scales.get("impact_penalty", -0.02) * impact_penalty,
            "torque_saturation": scales.get("torque_saturation", -0.01) * torque_sat_penalty,
            "drag_foot_penalty": drag_foot_penalty,
            "stagnation_penalty": stagnation_penalty,
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

        # v24: 设置初始朝向为+Y方向 (yaw=π/2), 让机器人面向目标
        # 默认四元数[0,0,0,1]朝向+X, 需要绕Z轴旋转90°
        init_quat = self._euler_to_quat(0, 0, np.pi / 2)  # [qx, qy, qz, qw]
        for env_idx in range(num_envs):
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

        # 初始目标: Phase 0 最近的未收集笑脸 (从起点(0,-2.5)出发, 通常是中心笑脸(0,0))
        # 构建临时info用于 _get_current_target
        init_smileys = np.zeros((num_envs, self.num_smileys), dtype=bool)
        init_rp = np.zeros((num_envs, self.num_red_packets), dtype=bool)
        init_nav_phase = np.full(num_envs, self.PHASE_APPROACH, dtype=np.int32)
        temp_info = {"nav_phase": init_nav_phase, "smileys_reached": init_smileys, "red_packets_reached": init_rp}
        first_target = self._get_current_target(temp_info, robot_init_xy)
        pose_commands = np.column_stack([
            first_target,
            np.zeros((num_envs, 1), dtype=np.float32)
        ])
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

        # 运动命令
        target_xy = pose_commands[:, :2]
        position_error = target_xy - root_pos[:, :2]
        distance_to_target = np.linalg.norm(position_error, axis=1)
        reached = distance_to_target < self.wp_radius
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached[:, np.newaxis], 0.0, desired_vel_xy)
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel[:, :2])

        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = self._wrap_angle(desired_heading - robot_heading)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < np.deg2rad(8), 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached, 0.0, desired_yaw_rate)

        velocity_commands = np.column_stack([desired_vel_xy, desired_yaw_rate[:, np.newaxis]])
        heading_diff = self._wrap_angle(desired_heading - robot_heading)

        # 观测 (must match update_state obs layout exactly)
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        base_height_norm = np.clip((robot_init_xyz[:, 2] - 0.5) / 1.2, -1.0, 1.0)[:, np.newaxis]

        obs = np.concatenate([
            base_lin_vel * cfg.normalization.lin_vel,                 # 3
            gyro * cfg.normalization.ang_vel,                          # 3
            projected_gravity,                                         # 3
            joint_pos_rel * cfg.normalization.dof_pos,                 # 12
            joint_vel_r * cfg.normalization.dof_vel,                   # 12
            np.zeros((num_envs, self._num_action), dtype=np.float32),  # 12 last_actions
            position_error_normalized,                                 # 2
            heading_error_normalized[:, np.newaxis],                   # 1
            base_height_norm,                                          # 1
            np.zeros((num_envs, 1), dtype=np.float32),                # 1 celeb_progress
            np.zeros((num_envs, 4), dtype=np.float32),                # 4 foot_contact
            np.zeros((num_envs, 3), dtype=np.float32),                # 3 trunk_acc (v20)
            np.zeros((num_envs, 12), dtype=np.float32),               # 12 actuator_torques (v20)
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
            "milestones_reached": np.zeros((num_envs, 2), dtype=bool),
            # 分阶段区域收集导航
            "nav_phase": np.full(num_envs, self.PHASE_APPROACH, dtype=np.int32),
            "wp_idx": np.zeros(num_envs, dtype=np.int32),
            "platform_reached": np.zeros(num_envs, dtype=bool),
            # 庆祝状态机 (v27: 多次跳跃)
            "celeb_state": np.full(num_envs, CELEB_IDLE, dtype=np.int32),
            "jump_count": np.zeros(num_envs, dtype=np.int32),
            # 得分区追踪
            "smileys_reached": np.zeros((num_envs, self.num_smileys), dtype=bool),
            "red_packets_reached": np.zeros((num_envs, self.num_red_packets), dtype=bool),
            # (footprint-contains检测不需要帧间状态)
            # 竞赛规则: 累积奖金追踪 (终止时清零用)
            "accumulated_bonus": np.zeros(num_envs, dtype=np.float32),
            "oob_terminated": np.zeros(num_envs, dtype=bool),
            # v44: 停滞检测 (stagnation timeout)
            "stagnation_anchor_xy": robot_init_xy.copy(),
            "stagnation_anchor_step": np.zeros(num_envs, dtype=np.int32),
        }

        return obs, info
