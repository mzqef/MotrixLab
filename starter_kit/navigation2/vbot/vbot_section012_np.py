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
VBot Section012 桥优先多目标导航环境 — 基于 Section011 状态机架构

竞赛规则 (Section 2 = 60分):
  +10: 通过波浪地形到达楼梯
  +5:  从左楼梯到达吊桥
  +10: 经过吊桥途径拜年红包到达楼梯口
  +5:  从楼梯口下来到达丙午大吉平台
  +5:  庆祝动作
  +3×5: 河床石头贺礼红包 (右路线, 本策略暂不主动追求)
  +5×2: 桥底下拜年红包 (过桥后激活收集)

桥优先导航状态机:
  Phase 0 (WAVE_TO_STAIR):       通过波浪地形到达左楼梯底
  Phase 1 (CLIMB_STAIR):         爬左楼梯到达桥入口
  Phase 2 (CROSS_BRIDGE):        过桥 — 虚拟导航点 entry→mid→exit
  Phase 3 (DESCEND_STAIR):       下楼梯到达底部
  Phase 4 (COLLECT_UNDER_BRIDGE): 过桥后收集桥下红包
  Phase 5 (REACH_EXIT):          到达丙午大吉平台
  Phase 6 (CELEBRATION):         在平台上跳跃庆祝

wp_idx = 里程碑计数 (0-N, 追踪总进度)
观测: 69维 (与section011对齐, 含trunk_acc + actuator_torques)
"""

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection012EnvCfg, TerrainScaleHelper

# ============================================================
# 庆祝状态机常量 (与section011一致)
# ============================================================
CELEB_IDLE = 0
CELEB_JUMP = 1
CELEB_DONE = 2

# ============================================================
# 导航阶段常量
# ============================================================
PHASE_WAVE_TO_STAIR = 0       # 波浪地形→左楼梯底
PHASE_CLIMB_STAIR = 1         # 爬左楼梯
PHASE_CROSS_BRIDGE = 2        # 过桥 (3个虚拟WP)
PHASE_DESCEND_STAIR = 3       # 下楼梯
PHASE_COLLECT_UNDER_BRIDGE = 4  # 收集桥下红包
PHASE_REACH_EXIT = 5          # 到达终点平台
PHASE_CELEBRATION = 6         # 庆祝


def generate_repeating_array(num_period, num_reset, period_counter):
    """生成重复数组，用于在固定位置中循环选择"""
    idx = []
    for i in range(num_reset):
        idx.append((period_counter + i) % num_period)
    return np.array(idx)


@registry.env("vbot_navigation_section012", "np")
class VBotSection012Env(NpEnv):
    """
    VBot Section02 桥优先多目标导航 + 跳跃庆祝
    地形: 入口平台 → 楼梯 → 拱桥/球障碍 → 楼梯 → 出口平台
    观测: 69维 (与section011对齐)
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

        # 初始化得分区 & 导航系统
        self._init_scoring_zones(cfg)
        self._init_bridge_nav(cfg)

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

    def _init_scoring_zones(self, cfg):
        """初始化竞赛得分区 — 桥上红包、桥下红包、石头红包、庆祝区"""
        if hasattr(cfg, 'scoring_zones'):
            sz = cfg.scoring_zones
            # 桥上拜年红包
            self.bridge_hongbao_center = np.array(sz.bridge_hongbao_center, dtype=np.float32)
            self.bridge_hongbao_radius = sz.bridge_hongbao_radius
            self.bridge_hongbao_min_z = sz.bridge_hongbao_min_z
            # 桥下拜年红包 (2个)
            self.under_bridge_centers = np.array(sz.under_bridge_centers, dtype=np.float32)
            self.under_bridge_radius = sz.under_bridge_radius
            self.under_bridge_max_z = sz.under_bridge_max_z
            self.num_under_bridge = len(sz.under_bridge_centers)
            # 河床石头红包 (5个)
            self.stone_hongbao_centers = np.array(sz.stone_hongbao_centers, dtype=np.float32)
            self.stone_hongbao_radius = sz.stone_hongbao_radius
            self.num_stone_hongbao = len(sz.stone_hongbao_centers)
            # 庆祝区
            self.celebration_center = np.array(sz.celebration_center, dtype=np.float32)
            self.celebration_radius = sz.celebration_radius
            self.celebration_min_z = sz.celebration_min_z
            self.has_scoring_zones = True
            print(f"[Info] Section012得分区: 桥上1红包, {self.num_under_bridge}桥下红包, "
                  f"{self.num_stone_hongbao}石头红包, 庆祝区")
        else:
            self.has_scoring_zones = False
            self.num_under_bridge = 2
            self.num_stone_hongbao = 5

    def _init_bridge_nav(self, cfg):
        """初始化桥优先导航系统"""
        if hasattr(cfg, 'bridge_nav'):
            bn = cfg.bridge_nav
            # 虚拟航点
            self.wp_wave_to_stair = np.array(bn.wave_to_stair, dtype=np.float32)
            self.wp_stair_top = np.array(bn.stair_top, dtype=np.float32)
            self.wp_stair_top_min_z = bn.stair_top_min_z
            self.wp_bridge_entry = np.array(bn.bridge_entry, dtype=np.float32)
            self.wp_bridge_mid = np.array(bn.bridge_mid, dtype=np.float32)
            self.wp_bridge_exit = np.array(bn.bridge_exit, dtype=np.float32)
            self.wp_bridge_min_z = bn.bridge_min_z
            self.wp_stair_down_bottom = np.array(bn.stair_down_bottom, dtype=np.float32)
            self.wp_exit_platform = np.array(bn.exit_platform, dtype=np.float32)
            self.wp_radius = bn.waypoint_radius
            self.bridge_wp_radius = bn.bridge_wp_radius
            self.wp_final_radius = bn.final_radius
            self.celeb_jump_threshold = bn.celebration_jump_threshold
        else:
            # 默认值
            self.wp_wave_to_stair = np.array([-3.0, 12.3], dtype=np.float32)
            self.wp_stair_top = np.array([-3.0, 14.5], dtype=np.float32)
            self.wp_stair_top_min_z = 2.3
            self.wp_bridge_entry = np.array([-3.0, 15.8], dtype=np.float32)
            self.wp_bridge_mid = np.array([-3.0, 17.83], dtype=np.float32)
            self.wp_bridge_exit = np.array([-3.0, 20.0], dtype=np.float32)
            self.wp_bridge_min_z = 2.3
            self.wp_stair_down_bottom = np.array([-3.0, 23.2], dtype=np.float32)
            self.wp_exit_platform = np.array([0.0, 24.33], dtype=np.float32)
            self.wp_radius = 1.2
            self.bridge_wp_radius = 1.5
            self.wp_final_radius = 0.8
            self.celeb_jump_threshold = 1.55

        # 过桥虚拟WP序列 (Phase 2内的sub-index)
        self.bridge_waypoints = np.array([
            self.wp_bridge_entry,
            self.wp_bridge_mid,
            self.wp_bridge_exit
        ], dtype=np.float32)  # [3, 2]

        print(f"[Info] 桥优先导航: 波浪→左楼梯→过桥(3WP)→下楼梯→桥下红包→终点→庆祝")

    # ============================================================
    # 传感器 & 物理辅助 (与section011一致)
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
        """桥优先多目标导航状态机

        Phase 0 (WAVE_TO_STAIR):       到达左楼梯底 wp_wave_to_stair
        Phase 1 (CLIMB_STAIR):         爬到楼梯顶 wp_stair_top (z>2.3)
        Phase 2 (CROSS_BRIDGE):        过桥3个虚拟WP (bridge_sub_idx 0→1→2)
        Phase 3 (DESCEND_STAIR):       下楼梯底 wp_stair_down_bottom
        Phase 4 (COLLECT_UNDER_BRIDGE): 收集桥下红包 (2个, 过桥后才激活)
        Phase 5 (REACH_EXIT):          到达终点平台 wp_exit_platform
        Phase 6 (CELEBRATION):         跳跃庆祝
        """
        nav_phase = info["nav_phase"]
        celeb_state = info["celeb_state"]
        n = self._num_envs
        scales = self._cfg.reward_config.scales

        milestone_bonus = np.zeros(n, dtype=np.float32)
        phase_bonus = np.zeros(n, dtype=np.float32)
        celeb_bonus = np.zeros(n, dtype=np.float32)
        jump_reward = np.zeros(n, dtype=np.float32)
        bridge_hongbao_tb = np.zeros(n, dtype=np.float32)
        under_bridge_tb = np.zeros(n, dtype=np.float32)
        stone_hongbao_tb = np.zeros(n, dtype=np.float32)

        # --- Phase 0: 波浪地形→左楼梯底 ---
        in_p0 = (nav_phase == PHASE_WAVE_TO_STAIR)
        if np.any(in_p0):
            d = np.linalg.norm(robot_xy - self.wp_wave_to_stair[np.newaxis, :], axis=1)
            arrived = in_p0 & (d < self.wp_radius)
            first = arrived & ~info.get("wave_reached", np.zeros(n, dtype=bool))
            info["wave_reached"] = info.get("wave_reached", np.zeros(n, dtype=bool)) | arrived
            milestone_bonus += np.where(first, scales.get("wave_traversal_bonus", 30.0), 0.0)
            nav_phase = np.where(arrived, PHASE_CLIMB_STAIR, nav_phase)
            phase_bonus += np.where(first, scales.get("phase_completion_bonus", 15.0), 0.0)

        # --- Phase 1: 爬左楼梯 (需要z>stair_top_min_z) ---
        in_p1 = (nav_phase == PHASE_CLIMB_STAIR)
        if np.any(in_p1):
            d = np.linalg.norm(robot_xy - self.wp_stair_top[np.newaxis, :], axis=1)
            z_ok = current_z > self.wp_stair_top_min_z
            arrived = in_p1 & (d < self.wp_radius) & z_ok
            first = arrived & ~info.get("stair_top_reached", np.zeros(n, dtype=bool))
            info["stair_top_reached"] = info.get("stair_top_reached", np.zeros(n, dtype=bool)) | arrived
            milestone_bonus += np.where(first, scales.get("stair_top_bonus", 25.0), 0.0)
            nav_phase = np.where(arrived, PHASE_CROSS_BRIDGE, nav_phase)
            phase_bonus += np.where(first, scales.get("phase_completion_bonus", 15.0), 0.0)
            # 进入过桥阶段时初始化bridge_sub_idx
            info["bridge_sub_idx"] = np.where(arrived, 0, info.get("bridge_sub_idx", np.zeros(n, dtype=np.int32)))

        # --- Phase 2: 过桥 (3个虚拟WP, bridge_sub_idx 0→1→2) ---
        in_p2 = (nav_phase == PHASE_CROSS_BRIDGE)
        if np.any(in_p2):
            bridge_sub_idx = info.get("bridge_sub_idx", np.zeros(n, dtype=np.int32))
            for sub_i in range(3):
                sub_mask = in_p2 & (bridge_sub_idx == sub_i)
                if np.any(sub_mask):
                    wp = self.bridge_waypoints[sub_i]
                    d = np.linalg.norm(robot_xy - wp[np.newaxis, :], axis=1)
                    z_ok = current_z > self.wp_bridge_min_z
                    arrived = sub_mask & (d < self.bridge_wp_radius) & z_ok
                    if np.any(arrived):
                        bridge_sub_idx = np.where(arrived, bridge_sub_idx + 1, bridge_sub_idx)
            info["bridge_sub_idx"] = bridge_sub_idx

            # 所有3个WP通过 → 完成过桥
            bridge_done = in_p2 & (bridge_sub_idx >= 3)
            first_bridge = bridge_done & ~info.get("bridge_crossed", np.zeros(n, dtype=bool))
            info["bridge_crossed"] = info.get("bridge_crossed", np.zeros(n, dtype=bool)) | bridge_done
            milestone_bonus += np.where(first_bridge, scales.get("bridge_crossing_bonus", 50.0), 0.0)
            phase_bonus += np.where(first_bridge, scales.get("phase_completion_bonus", 15.0), 0.0)
            nav_phase = np.where(bridge_done, PHASE_DESCEND_STAIR, nav_phase)

            # 桥上红包: 过桥中途自然收集 (bridge_mid附近)
            if self.has_scoring_zones:
                d_hb = np.linalg.norm(robot_xy - self.bridge_hongbao_center[np.newaxis, :], axis=1)
                z_hb = current_z > self.bridge_hongbao_min_z
                hongbao_reached = in_p2 & (d_hb < self.bridge_hongbao_radius) & z_hb
                first_hb = hongbao_reached & ~info.get("bridge_hongbao_collected", np.zeros(n, dtype=bool))
                info["bridge_hongbao_collected"] = info.get("bridge_hongbao_collected", np.zeros(n, dtype=bool)) | hongbao_reached
                hb_val = np.where(first_hb, scales.get("bridge_hongbao_bonus", 30.0), 0.0)
                milestone_bonus += hb_val
                bridge_hongbao_tb += hb_val

        # --- Phase 3: 下楼梯 ---
        in_p3 = (nav_phase == PHASE_DESCEND_STAIR)
        if np.any(in_p3):
            d = np.linalg.norm(robot_xy - self.wp_stair_down_bottom[np.newaxis, :], axis=1)
            arrived = in_p3 & (d < self.wp_radius)
            first = arrived & ~info.get("stair_down_reached", np.zeros(n, dtype=bool))
            info["stair_down_reached"] = info.get("stair_down_reached", np.zeros(n, dtype=bool)) | arrived
            milestone_bonus += np.where(first, scales.get("stair_down_bonus", 20.0), 0.0)
            nav_phase = np.where(arrived, PHASE_COLLECT_UNDER_BRIDGE, nav_phase)
            phase_bonus += np.where(first, scales.get("phase_completion_bonus", 15.0), 0.0)

        # --- Phase 4: 收集桥下红包 (过桥后才激活, 2个) ---
        in_p4 = (nav_phase == PHASE_COLLECT_UNDER_BRIDGE)
        if np.any(in_p4) and self.has_scoring_zones:
            ub_reached = info.get("under_bridge_reached", np.zeros((n, self.num_under_bridge), dtype=bool))
            for i in range(self.num_under_bridge):
                d = np.linalg.norm(robot_xy - self.under_bridge_centers[i][np.newaxis, :], axis=1)
                z_ok = current_z < self.under_bridge_max_z
                first_collect = in_p4 & (d < self.under_bridge_radius) & z_ok & ~ub_reached[:, i]
                ub_reached[:, i] |= (in_p4 & (d < self.under_bridge_radius) & z_ok)
                ub_val = np.where(first_collect, scales.get("under_bridge_bonus", 15.0), 0.0)
                milestone_bonus += ub_val
                under_bridge_tb += ub_val
            info["under_bridge_reached"] = ub_reached
            # 两个都收集或y已过桥区 → 晋级
            all_ub = np.all(ub_reached, axis=1)
            past_bridge = robot_xy[:, 1] > 21.5
            can_advance = in_p4 & (all_ub | past_bridge)
            nav_phase = np.where(can_advance, PHASE_REACH_EXIT, nav_phase)
        elif np.any(in_p4):
            # 无得分区定义时直接跳过
            nav_phase = np.where(in_p4, PHASE_REACH_EXIT, nav_phase)

        # --- 石头红包: 任何阶段都可以顺路收集 (不影响主线进度) ---
        if self.has_scoring_zones:
            stone_reached = info.get("stone_hongbao_reached", np.zeros((n, self.num_stone_hongbao), dtype=bool))
            for i in range(self.num_stone_hongbao):
                d = np.linalg.norm(robot_xy - self.stone_hongbao_centers[i][np.newaxis, :], axis=1)
                first_collect = (d < self.stone_hongbao_radius) & ~stone_reached[:, i]
                stone_reached[:, i] |= (d < self.stone_hongbao_radius)
                sv = np.where(first_collect, scales.get("stone_hongbao_bonus", 8.0), 0.0)
                milestone_bonus += sv
                stone_hongbao_tb += sv
            info["stone_hongbao_reached"] = stone_reached

        # --- Phase 5: 到达终点平台 ---
        in_p5 = (nav_phase == PHASE_REACH_EXIT)
        if np.any(in_p5):
            d = np.linalg.norm(robot_xy - self.wp_exit_platform[np.newaxis, :], axis=1)
            arrived = in_p5 & (d < self.wp_final_radius)
            first = arrived & ~info.get("exit_reached", np.zeros(n, dtype=bool))
            info["exit_reached"] = info.get("exit_reached", np.zeros(n, dtype=bool)) | arrived
            milestone_bonus += np.where(first, scales.get("phase_completion_bonus", 15.0), 0.0)
            nav_phase = np.where(arrived, PHASE_CELEBRATION, nav_phase)

        # --- 更新wp_idx (进度指标) ---
        wp_idx = np.zeros(n, dtype=np.int32)
        wp_idx += info.get("wave_reached", np.zeros(n, dtype=bool)).astype(np.int32)
        wp_idx += info.get("stair_top_reached", np.zeros(n, dtype=bool)).astype(np.int32)
        wp_idx += np.clip(info.get("bridge_sub_idx", np.zeros(n, dtype=np.int32)), 0, 3)
        wp_idx += info.get("stair_down_reached", np.zeros(n, dtype=bool)).astype(np.int32)
        wp_idx += np.sum(info.get("under_bridge_reached", np.zeros((n, self.num_under_bridge), dtype=bool)), axis=1).astype(np.int32)
        wp_idx += info.get("exit_reached", np.zeros(n, dtype=bool)).astype(np.int32)
        info["wp_idx"] = wp_idx
        info["nav_phase"] = nav_phase

        # --- Phase 6: 庆祝 (跳跃) ---
        in_celeb = (nav_phase == PHASE_CELEBRATION)
        start_celeb = in_celeb & (celeb_state == CELEB_IDLE)
        if np.any(start_celeb):
            celeb_state = np.where(start_celeb, CELEB_JUMP, celeb_state)
            info["celeb_anchor_xy"] = np.where(
                start_celeb[:, np.newaxis], robot_xy,
                info.get("celeb_anchor_xy", np.zeros((n, 2), dtype=np.float32))
            )

        jumping = in_celeb & (celeb_state == CELEB_JUMP)
        if np.any(jumping):
            z_above_standing = np.maximum(current_z - 1.5, 0.0)
            jump_reward += np.where(jumping, scales.get("jump_reward", 8.0) * z_above_standing, 0.0)
            jumped = jumping & (current_z > self.celeb_jump_threshold)
            if np.any(jumped):
                celeb_state = np.where(jumped, CELEB_DONE, celeb_state)
                celeb_bonus += np.where(jumped, scales.get("celebration_bonus", 80.0), 0.0)

        info["celeb_state"] = celeb_state

        wp_bonus = milestone_bonus + phase_bonus
        info["_bridge_hongbao_tb"] = bridge_hongbao_tb
        info["_under_bridge_tb"] = under_bridge_tb
        info["_stone_hongbao_tb"] = stone_hongbao_tb
        info["_phase_bonus_tb"] = phase_bonus
        return info, wp_bonus, celeb_bonus, jump_reward

    def _get_current_target(self, info, robot_xy):
        """获取当前导航目标 — 基于Phase和sub-index"""
        nav_phase = info["nav_phase"]
        n = len(nav_phase)
        targets = np.tile(self.wp_exit_platform, (n, 1))  # 默认: 终点

        # Phase 0: 波浪→左楼梯底
        m0 = nav_phase == PHASE_WAVE_TO_STAIR
        if np.any(m0):
            targets[m0] = self.wp_wave_to_stair

        # Phase 1: 爬楼梯
        m1 = nav_phase == PHASE_CLIMB_STAIR
        if np.any(m1):
            targets[m1] = self.wp_stair_top

        # Phase 2: 过桥 (基于bridge_sub_idx选择当前WP)
        m2 = nav_phase == PHASE_CROSS_BRIDGE
        if np.any(m2):
            bridge_sub = info.get("bridge_sub_idx", np.zeros(n, dtype=np.int32))
            for sub_i in range(3):
                sub_mask = m2 & (bridge_sub == sub_i)
                if np.any(sub_mask):
                    targets[sub_mask] = self.bridge_waypoints[sub_i]
            # bridge_sub >= 3 的env已经晋级, 但防御性处理
            done_mask = m2 & (bridge_sub >= 3)
            if np.any(done_mask):
                targets[done_mask] = self.wp_bridge_exit

        # Phase 3: 下楼梯
        m3 = nav_phase == PHASE_DESCEND_STAIR
        if np.any(m3):
            targets[m3] = self.wp_stair_down_bottom

        # Phase 4: 收集桥下红包 — 导航到最近的未收集桥下红包
        m4 = nav_phase == PHASE_COLLECT_UNDER_BRIDGE
        if np.any(m4) and self.has_scoring_zones:
            ub_reached = info.get("under_bridge_reached", np.zeros((n, self.num_under_bridge), dtype=bool))
            p4_envs = np.where(m4)[0]
            for env_i in p4_envs:
                uncollected = ~ub_reached[env_i]
                if np.any(uncollected):
                    dists = np.linalg.norm(robot_xy[env_i] - self.under_bridge_centers, axis=1)
                    dists = np.where(uncollected, dists, 1e6)
                    nearest_idx = np.argmin(dists)
                    targets[env_i] = self.under_bridge_centers[nearest_idx]
                else:
                    targets[env_i] = self.wp_exit_platform

        # Phase 5: 终点
        m5 = nav_phase == PHASE_REACH_EXIT
        if np.any(m5):
            targets[m5] = self.wp_exit_platform

        # Phase 6: 庆祝 (原地)
        m6 = nav_phase == PHASE_CELEBRATION
        if np.any(m6):
            targets[m6] = self.wp_exit_platform

        return targets

    # ============================================================
    # apply_action & torques (与section011一致)
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
        info, wp_bonus, celeb_bonus, jump_reward = \
            self._update_waypoint_state(info, robot_xy, robot_heading, speed_xy, gyro[:, 2], current_z)

        # --- 当前导航目标 ---
        target_xy = self._get_current_target(info, robot_xy)
        in_celeb = (info["nav_phase"] >= PHASE_CELEBRATION)

        position_error = target_xy - robot_xy
        distance_to_target = np.linalg.norm(position_error, axis=1)

        pose_commands = np.column_stack([target_xy, np.zeros(self._num_envs, dtype=np.float32)])
        info["pose_commands"] = pose_commands

        # 到达当前WP?
        current_wp_radius = np.where(
            info["nav_phase"] >= PHASE_REACH_EXIT, self.wp_final_radius,
            np.where(info["nav_phase"] == PHASE_CROSS_BRIDGE, self.bridge_wp_radius, self.wp_radius)
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

        # 庆祝进度观测
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
            wp_bonus, celeb_bonus, jump_reward, in_celeb
        )

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        # 庆祝完成截断
        celeb_done = (info["celeb_state"] == CELEB_DONE)
        self._success_truncate = celeb_done

        state.info["metrics"] = {
            "distance_to_target": distance_to_target,
            "reached_fraction": reached_wp.astype(np.float32),
            "wp_idx_mean": info["wp_idx"].astype(np.float32),
            "nav_phase_mean": info["nav_phase"].astype(np.float32),
            "celeb_state_mean": info["celeb_state"].astype(np.float32),
        }
        return state

    def _update_truncate(self):
        super()._update_truncate()
        if hasattr(self, '_success_truncate'):
            self._state.truncated = np.logical_or(self._state.truncated, self._success_truncate)

    # ============================================================
    # 终止条件 (与section011一致: hard/soft分层 + grace + OOB)
    # ============================================================

    def _compute_terminated(self, state, projected_gravity=None, joint_vel=None, robot_xy=None, current_z=None):
        data = state.data
        n = self._num_envs

        # === HARD terminations ===
        hard_terminated = np.zeros(n, dtype=bool)

        if projected_gravity is not None:
            gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
            gz = projected_gravity[:, 2]
            tilt_angle = np.arctan2(gxy, np.abs(gz))
            hard_terminated |= tilt_angle > np.deg2rad(70)

        bounds = getattr(self._cfg, 'course_bounds', None)
        if bounds is not None and robot_xy is not None and current_z is not None:
            oob_x = (robot_xy[:, 0] < bounds.x_min) | (robot_xy[:, 0] > bounds.x_max)
            oob_y = (robot_xy[:, 1] < bounds.y_min) | (robot_xy[:, 1] > bounds.y_max)
            oob_z = current_z < bounds.z_min
            oob = oob_x | oob_y | oob_z
            hard_terminated |= oob
            state.info["oob_terminated"] = oob

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

        # === SOFT terminations (grace protected) ===
        soft_terminated = np.zeros(n, dtype=bool)

        try:
            base_contact_value = self._model.get_sensor_value("base_contact", data)
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

        if projected_gravity is not None:
            gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
            gz = projected_gravity[:, 2]
            tilt_angle = np.arctan2(gxy, np.abs(gz))
            soft_terminated |= tilt_angle > np.deg2rad(50)

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
                         speed_xy, wp_bonus, celeb_bonus, jump_reward, in_celeb):
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

        # 存活奖励 (条件式)
        gz = np.clip(-projected_gravity[:, 2], 0.0, 1.0)
        upright_factor = np.where(gz > 0.9, 1.0, np.where(gz > 0.7, 0.5, 0.0))
        alive_bonus = scales.get("alive_bonus", 0.05) * upright_factor

        # ===== 稳定性惩罚 =====
        orientation_penalty = np.sum(np.square(projected_gravity[:, :2]), axis=1)
        lin_vel_z_penalty = np.square(np.clip(base_lin_vel[:, 2], -50.0, 50.0))
        ang_vel_xy_penalty = np.sum(np.square(np.clip(gyro[:, :2], -50.0, 50.0)), axis=1)

        actual_torques_r = self._get_actuator_torques(data)
        torque_penalty = np.sum(np.square(np.clip(actual_torques_r, -200.0, 200.0)), axis=1)
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

        # ===== Zone吸引力: 桥下红包接近delta (仅Phase 4) =====
        zone_approach_reward = np.zeros(n, dtype=np.float32)
        zone_approach_scale = scales.get("zone_approach", 5.0)
        if self.has_scoring_zones and zone_approach_scale > 0:
            in_p4 = (info["nav_phase"] == PHASE_COLLECT_UNDER_BRIDGE)
            if np.any(in_p4):
                ub_reached = info.get("under_bridge_reached", np.zeros((n, self.num_under_bridge), dtype=bool))
                for i in range(self.num_under_bridge):
                    d = np.linalg.norm(robot_xy - self.under_bridge_centers[i][np.newaxis, :], axis=1)
                    last_key = f"last_ub_dist_{i}"
                    last_d = info.get(last_key, d.copy())
                    delta = last_d - d
                    uncollected = ~ub_reached[:, i]
                    active = in_p4 & uncollected
                    delta_reward = np.clip(delta * zone_approach_scale * 10.0, -0.3, 2.0)
                    zone_approach_reward += np.where(active, delta_reward, 0.0)
                    info[last_key] = d.copy()

        # ===== 楼梯区抬脚 =====
        on_stair = ((current_y > 12.0) & (current_y < 14.5)) | ((current_y > 21.0) & (current_y < 23.5))
        foot_clearance_scale = scales.get("foot_clearance", 0.02)
        if foot_clearance_scale > 0:
            foot_forces = self._get_foot_contact_forces(data)
            force_mag = np.linalg.norm(foot_forces, axis=2)
            in_swing = force_mag < 0.5
            calf_indices = [2, 5, 8, 11]
            calf_vel = np.abs(joint_vel[:, calf_indices])
            stair_boost = np.where(on_stair, scales.get("foot_clearance_stair_boost", 3.0), 1.0)
            clearance_scale_vec = foot_clearance_scale * stair_boost
            foot_clearance_reward = clearance_scale_vec * np.sum(
                in_swing.astype(np.float32) * np.clip(calf_vel, 0.0, 5.0) * 0.2, axis=1
            )
        else:
            foot_clearance_reward = np.zeros(n, dtype=np.float32)

        # ===== 摆动相接触惩罚 =====
        stair_swing_scale = np.where(on_stair, scales.get("swing_contact_stair_scale", 0.5), 1.0)
        swing_penalty = (
            scales.get("swing_contact_penalty", -0.025)
            * self._compute_swing_contact_penalty(data, joint_vel)
            * stair_swing_scale
        )

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
            + scales.get("dof_vel", -3e-5) * dof_vel_penalty
            + scales.get("dof_acc", -1.5e-7) * dof_acc_penalty
            + scales.get("action_rate", -0.005) * action_rate_penalty
            + scales.get("impact_penalty", -0.02) * impact_penalty
            + scales.get("torque_saturation", -0.01) * torque_sat_penalty
            + termination_penalty
            + swing_penalty
        )

        # ===== 综合奖励 =====
        nav_reward = (
            scales.get("position_tracking", 0.05) * position_tracking
            + wp_approach
            + wp_facing
            + scales.get("forward_velocity", 3.0) * forward_velocity
            + alive_bonus
        )
        nav_reward = np.where(in_celeb, alive_bonus, nav_reward)

        reward = (
            nav_reward
            + wp_bonus
            + celeb_bonus
            + jump_reward
            + height_progress
            + traversal_total
            + zone_approach_reward
            + foot_clearance_reward
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
            "jump_reward": jump_reward,
            "bridge_hongbao": info.get("_bridge_hongbao_tb", np.zeros(n, dtype=np.float32)),
            "under_bridge": info.get("_under_bridge_tb", np.zeros(n, dtype=np.float32)),
            "stone_hongbao": info.get("_stone_hongbao_tb", np.zeros(n, dtype=np.float32)),
            "zone_approach": zone_approach_reward,
            "height_progress": height_progress,
            "traversal_bonus": traversal_total,
            "penalties": penalties,
            "termination": termination_penalty,
            "swing_contact_penalty": swing_penalty,
            "foot_clearance": foot_clearance_reward,
            "score_clear_penalty": score_clear_penalty,
            "phase_completion_bonus": info.get("_phase_bonus_tb", np.zeros(n, dtype=np.float32)),
            "gait_stance": gait_stance,
            "impact_penalty": scales.get("impact_penalty", -0.02) * impact_penalty,
            "torque_saturation": scales.get("torque_saturation", -0.01) * torque_sat_penalty,
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

        for env_idx in range(num_envs):
            q = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            qn = np.linalg.norm(q)
            dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = q / qn if qn > 1e-6 else [0, 0, 0, 1]
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

        # 初始目标: Phase 0 → 左楼梯底
        init_nav_phase = np.full(num_envs, PHASE_WAVE_TO_STAIR, dtype=np.int32)
        temp_info = {"nav_phase": init_nav_phase}
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
        reached = distance_to_target < self.wp_radius
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
            # 桥优先导航状态
            "nav_phase": np.full(num_envs, PHASE_WAVE_TO_STAIR, dtype=np.int32),
            "wp_idx": np.zeros(num_envs, dtype=np.int32),
            "bridge_sub_idx": np.zeros(num_envs, dtype=np.int32),
            # 里程碑追踪
            "wave_reached": np.zeros(num_envs, dtype=bool),
            "stair_top_reached": np.zeros(num_envs, dtype=bool),
            "bridge_crossed": np.zeros(num_envs, dtype=bool),
            "bridge_hongbao_collected": np.zeros(num_envs, dtype=bool),
            "stair_down_reached": np.zeros(num_envs, dtype=bool),
            "under_bridge_reached": np.zeros((num_envs, self.num_under_bridge), dtype=bool),
            "stone_hongbao_reached": np.zeros((num_envs, self.num_stone_hongbao), dtype=bool),
            "exit_reached": np.zeros(num_envs, dtype=bool),
            # 庆祝状态机
            "celeb_state": np.full(num_envs, CELEB_IDLE, dtype=np.int32),
            "celeb_anchor_xy": np.zeros((num_envs, 2), dtype=np.float32),
            # 累积奖金 (终止清零用)
            "accumulated_bonus": np.zeros(num_envs, dtype=np.float32),
            "oob_terminated": np.zeros(num_envs, dtype=bool),
        }

        return obs, info
