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
VBot Section011 多航点导航环境 - 竞赛得分区 + 庆祝旋转

航点序列: START(y=-2.5) -> 笑脸区(y=0) -> 红包区(y=4.4) -> 高台(y=7.83) -> 庆祝(原地旋转)
庆祝动作: 在高台上右转180 -> 左转180 -> 停住 (安全稳定, 四脚着地)
"""

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection011EnvCfg

# ============================================================
# 庆祝状态机常量
# ============================================================
CELEB_IDLE = 0        # 未开始庆祝
CELEB_SPIN_RIGHT = 1  # 阶段1: 右转180
CELEB_SPIN_LEFT = 2   # 阶段2: 左转180回到原方向
CELEB_HOLD = 3        # 阶段3: 保持静止
CELEB_DONE = 4        # 完成


@registry.env("vbot_navigation_section011", "np")
class VBotSection011Env(NpEnv):
    """
    VBot Section01 多航点导航 + 庆祝旋转
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
        # 观测空间: 54维 (与之前保持一致, 通过pose_commands切换航点目标)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)

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
        self._init_termination_contact()
        self._init_foot_contact()

    def _init_termination_contact(self):
        termination_contact_names = self._cfg.asset.terminate_after_contacts_on
        ground_geoms = []
        ground_prefix = self._cfg.asset.ground_subtree
        for geom_name in self._model.geom_names:
            if geom_name is not None and ground_prefix in geom_name:
                ground_geoms.append(self._model.get_geom_index(geom_name))
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
            print(f"[Info] 终止接触检测: {len(termination_contact_names)}x{len(ground_geoms)}={self.num_termination_check}对")
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0

    def _init_foot_contact(self):
        self.foot_sensor_names = ["FR_foot_contact", "FL_foot_contact", "RR_foot_contact", "RL_foot_contact"]
        self.num_foot_check = 4

    def _init_scoring_zones(self, cfg):
        if hasattr(cfg, 'scoring_zones'):
            sz = cfg.scoring_zones
            self.smiley_centers = np.array(sz.smiley_centers, dtype=np.float32)
            self.smiley_radius = sz.smiley_radius
            self.red_packet_centers = np.array(sz.red_packet_centers, dtype=np.float32)
            self.red_packet_radius = sz.red_packet_radius
            self.celebration_center = np.array(sz.celebration_center, dtype=np.float32)
            self.celebration_radius = sz.celebration_radius
            self.celebration_min_z = sz.celebration_min_z
            self.has_scoring_zones = True
            self.num_smileys = len(sz.smiley_centers)
            self.num_red_packets = len(sz.red_packet_centers)
            print(f"[Info] 得分区: {self.num_smileys}笑脸, {self.num_red_packets}红包, 庆祝区")
        else:
            self.has_scoring_zones = False
            self.num_smileys = 3
            self.num_red_packets = 3

    def _init_waypoints(self, cfg):
        """初始化多航点导航系统"""
        if hasattr(cfg, 'waypoint_nav'):
            wn = cfg.waypoint_nav
            self.waypoints = np.array(wn.waypoints, dtype=np.float32)  # [N_wp, 2]
            self.num_waypoints = len(wn.waypoints)
            self.wp_radius = wn.waypoint_radius
            self.wp_final_radius = wn.final_radius
            self.celeb_spin_angle = wn.celebration_spin_angle
            self.celeb_spin_tol = wn.celebration_spin_tolerance
            self.celeb_speed_limit = wn.celebration_spin_speed_limit
            self.celeb_hold_steps = wn.celebration_hold_steps
            print(f"[Info] 航点导航: {self.num_waypoints}个航点 + 庆祝旋转")
        else:
            # fallback: 单目标
            self.waypoints = np.array([[0.0, 7.83]], dtype=np.float32)
            self.num_waypoints = 1
            self.wp_radius = 1.0
            self.wp_final_radius = 0.5
            self.celeb_spin_angle = np.pi
            self.celeb_spin_tol = 0.3
            self.celeb_speed_limit = 0.3
            self.celeb_hold_steps = 30

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

    def _compute_swing_contact_penalty(self, data: mtx.SceneData, joint_vel: np.ndarray) -> np.ndarray:
        foot_forces = self._get_foot_contact_forces(data)
        force_magnitudes = np.linalg.norm(foot_forces, axis=2)
        calf_indices = [2, 5, 8, 11]
        foot_vel = np.abs(joint_vel[:, calf_indices])
        has_contact = force_magnitudes > 1.0
        has_high_vel = foot_vel > 2.0
        swing_contact = np.logical_and(has_contact, has_high_vel).astype(np.float32)
        penalty = np.sum(swing_contact * force_magnitudes * foot_vel / 100.0, axis=1)
        return penalty

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

    def _update_waypoint_state(self, info, robot_xy, robot_heading, speed_xy, gyro_z, current_z):
        """
        更新航点索引 & 庆祝状态机。返回更新后的 info + 各种奖励分量。

        航点转换逻辑:
          - wp_idx < num_waypoints: 导航到waypoints[wp_idx], 到达后wp_idx++
          - wp_idx == num_waypoints: 进入庆祝阶段

        庆祝状态机:
          IDLE -> SPIN_RIGHT (记录初始heading, 目标=heading-pi)
          SPIN_RIGHT -> SPIN_LEFT (到达后目标回到初始heading)
          SPIN_LEFT -> HOLD (到达后开始计数)
          HOLD -> DONE (计数满celebration_hold_steps)
        """
        wp_idx = info["wp_idx"]
        celeb_state = info["celeb_state"]
        celeb_init_heading = info["celeb_init_heading"]
        celeb_target_heading = info["celeb_target_heading"]
        celeb_hold_count = info["celeb_hold_count"]

        n = self._num_envs
        scales = self._cfg.reward_config.scales

        wp_bonus = np.zeros(n, dtype=np.float32)
        celeb_bonus = np.zeros(n, dtype=np.float32)
        spin_progress_reward = np.zeros(n, dtype=np.float32)
        spin_hold_reward = np.zeros(n, dtype=np.float32)

        # --- 航点到达检测 ---
        for wp_i in range(self.num_waypoints):
            is_at_this_wp = (wp_idx == wp_i)
            if not np.any(is_at_this_wp):
                continue
            wp_pos = self.waypoints[wp_i]
            dist_to_wp = np.linalg.norm(robot_xy - wp_pos[np.newaxis, :], axis=1)
            r = self.wp_final_radius if wp_i == self.num_waypoints - 1 else self.wp_radius
            reached = is_at_this_wp & (dist_to_wp < r)
            # 高台航点额外要求z>1.0
            if wp_i == self.num_waypoints - 1:
                reached = reached & (current_z > self.celebration_min_z)
            wp_bonus += np.where(reached, scales.get("waypoint_bonus", 25.0), 0.0)
            wp_idx = np.where(reached, wp_idx + 1, wp_idx)

        # --- 庆祝状态机 ---
        in_celeb = (wp_idx >= self.num_waypoints)

        # IDLE -> SPIN_RIGHT
        start_celeb = in_celeb & (celeb_state == CELEB_IDLE)
        if np.any(start_celeb):
            celeb_init_heading = np.where(start_celeb, robot_heading, celeb_init_heading)
            target_r = self._wrap_angle(celeb_init_heading - self.celeb_spin_angle)
            celeb_target_heading = np.where(start_celeb, target_r, celeb_target_heading)
            celeb_state = np.where(start_celeb, CELEB_SPIN_RIGHT, celeb_state)

        # SPIN_RIGHT: 奖励朝目标角度旋转的进步
        spinning_right = in_celeb & (celeb_state == CELEB_SPIN_RIGHT)
        if np.any(spinning_right):
            angle_err = np.abs(self._wrap_angle(robot_heading - celeb_target_heading))
            spin_prog = np.exp(-angle_err / 0.5)
            spin_progress_reward += np.where(spinning_right, scales.get("spin_progress", 3.0) * spin_prog, 0.0)
            reached_right = spinning_right & (angle_err < self.celeb_spin_tol)
            if np.any(reached_right):
                target_l = celeb_init_heading
                celeb_target_heading = np.where(reached_right, target_l, celeb_target_heading)
                celeb_state = np.where(reached_right, CELEB_SPIN_LEFT, celeb_state)

        # SPIN_LEFT
        spinning_left = in_celeb & (celeb_state == CELEB_SPIN_LEFT)
        if np.any(spinning_left):
            angle_err = np.abs(self._wrap_angle(robot_heading - celeb_target_heading))
            spin_prog = np.exp(-angle_err / 0.5)
            spin_progress_reward += np.where(spinning_left, scales.get("spin_progress", 3.0) * spin_prog, 0.0)
            reached_left = spinning_left & (angle_err < self.celeb_spin_tol)
            if np.any(reached_left):
                celeb_state = np.where(reached_left, CELEB_HOLD, celeb_state)
                celeb_hold_count = np.where(reached_left, 0, celeb_hold_count)

        # HOLD: 奖励保持静止
        holding = in_celeb & (celeb_state == CELEB_HOLD)
        if np.any(holding):
            is_still = (speed_xy < 0.15) & (np.abs(gyro_z) < 0.1)
            celeb_hold_count = np.where(holding & is_still, celeb_hold_count + 1, celeb_hold_count)
            spin_hold_reward += np.where(holding & is_still, scales.get("spin_hold", 5.0), 0.0)
            done_hold = holding & (celeb_hold_count >= self.celeb_hold_steps)
            if np.any(done_hold):
                celeb_state = np.where(done_hold, CELEB_DONE, celeb_state)
                celeb_bonus += np.where(done_hold, scales.get("celebration_bonus", 30.0), 0.0)

        # 庆祝期间惩罚过快平移
        celeb_speed_penalty = np.zeros(n, dtype=np.float32)
        is_spinning = in_celeb & ((celeb_state == CELEB_SPIN_RIGHT) | (celeb_state == CELEB_SPIN_LEFT))
        if np.any(is_spinning):
            excess_speed = np.maximum(speed_xy - self.celeb_speed_limit, 0.0)
            celeb_speed_penalty = np.where(is_spinning, -2.0 * excess_speed ** 2, 0.0)

        info["wp_idx"] = wp_idx
        info["celeb_state"] = celeb_state
        info["celeb_init_heading"] = celeb_init_heading
        info["celeb_target_heading"] = celeb_target_heading
        info["celeb_hold_count"] = celeb_hold_count

        return info, wp_bonus, celeb_bonus, spin_progress_reward, spin_hold_reward, celeb_speed_penalty

    def _get_current_target(self, info):
        """根据当前航点索引返回每个env的当前目标XY坐标 [num_envs, 2]"""
        wp_idx = info["wp_idx"]
        clamped_idx = np.clip(wp_idx, 0, self.num_waypoints - 1)
        targets = self.waypoints[clamped_idx]
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
        state.data.actuator_ctrls = self._compute_torques(state.info["filtered_actions"], state.data)
        return state

    def _compute_torques(self, actions, data):
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled
        current_pos = self.get_dof_pos(data)
        current_vel = self.get_dof_vel(data)
        kp, kv = 100.0, 8.0  # 更强PD控制: 配合action_scale=0.5, 足够力矩驱动关节
        torques = kp * (target_pos - current_pos) - kv * current_vel
        torque_limits = np.array([23, 23, 45] * 4, dtype=np.float32)  # 提高扭矩限制: 支持更大动作幅度
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
        info, wp_bonus, celeb_bonus, spin_progress, spin_hold, celeb_speed_pen = \
            self._update_waypoint_state(info, robot_xy, robot_heading, speed_xy, gyro[:, 2], current_z)

        # --- 当前导航目标 (基于航点) ---
        target_xy = self._get_current_target(info)
        in_celeb = (info["wp_idx"] >= self.num_waypoints)

        # 导航命令
        position_error = target_xy - robot_xy
        distance_to_target = np.linalg.norm(position_error, axis=1)

        # 更新pose_commands供可视化
        pose_commands = np.column_stack([target_xy, np.zeros(self._num_envs, dtype=np.float32)])
        info["pose_commands"] = pose_commands

        # 到达当前航点?
        wp_idx = info["wp_idx"]
        current_wp_radius = np.where(
            wp_idx >= self.num_waypoints - 1,
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

        # 庆祝时: 告诉policy需要旋转
        is_spinning = in_celeb & ((info["celeb_state"] == CELEB_SPIN_RIGHT) | (info["celeb_state"] == CELEB_SPIN_LEFT))
        if np.any(is_spinning):
            celeb_heading_err = self._wrap_angle(info["celeb_target_heading"] - robot_heading)
            celeb_yaw_cmd = np.clip(celeb_heading_err * 2.0, -1.0, 1.0)
            desired_yaw_rate = np.where(is_spinning, celeb_yaw_cmd, desired_yaw_rate)

        velocity_commands = np.column_stack([desired_vel_xy, desired_yaw_rate[:, np.newaxis] if desired_yaw_rate.ndim == 1 else desired_yaw_rate])

        # 朝向误差
        heading_diff = self._wrap_angle(np.zeros(self._num_envs) - robot_heading)

        # --- 观测 ---
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = info["current_actions"]

        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_wp.astype(np.float32)

        # 庆祝进度观测: 0=导航中, 0.25=spin_right, 0.5=spin_left, 0.75=hold, 1.0=done
        celeb_progress = info["celeb_state"].astype(np.float32) / 4.0

        obs = np.concatenate([
            noisy_linvel,                              # 3
            noisy_gyro,                                # 3
            projected_gravity,                         # 3
            noisy_joint_angle,                         # 12
            noisy_joint_vel,                           # 12
            last_actions,                              # 12
            command_normalized,                        # 3
            position_error_normalized,                 # 2
            heading_error_normalized[:, np.newaxis],   # 1
            distance_normalized[:, np.newaxis],        # 1
            reached_flag[:, np.newaxis],               # 1
            celeb_progress[:, np.newaxis],             # 1
        ], axis=-1)
        assert obs.shape == (data.shape[0], 54), f"obs shape {obs.shape} != (N, 54)"

        # 可视化
        self._update_target_marker(data, pose_commands)
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel[:, :2])

        # --- 步数追踪 (grace period用) ---
        info["steps"] = info["steps"] + 1

        # --- 终止 (包括摔倒+超出边界) ---
        terminated_state = self._compute_terminated(state, projected_gravity, joint_vel, robot_xy, current_z)
        terminated = terminated_state.terminated

        # --- 奖励 ---
        reward = self._compute_reward(
            data, info, velocity_commands, base_lin_vel, gyro, projected_gravity,
            joint_vel, distance_to_target, position_error, reached_wp,
            terminated, robot_heading, robot_xy, current_z, speed_xy,
            wp_bonus, celeb_bonus, spin_progress, spin_hold, celeb_speed_pen, in_celeb
        )

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        # 庆祝完成后截断
        celeb_done = (info["celeb_state"] == CELEB_DONE)
        self._success_truncate = celeb_done

        state.info["metrics"] = {
            "distance_to_target": distance_to_target,
            "reached_fraction": reached_wp.astype(np.float32),
            "wp_idx_mean": info["wp_idx"].astype(np.float32),
            "celeb_state_mean": info["celeb_state"].astype(np.float32),
        }
        return state

    def _update_truncate(self):
        super()._update_truncate()
        if hasattr(self, '_success_truncate'):
            self._state.truncated = np.logical_or(self._state.truncated, self._success_truncate)

    # ============================================================
    # 终止条件
    # ============================================================

    def _compute_terminated(self, state, projected_gravity=None, joint_vel=None, robot_xy=None, current_z=None):
        data = state.data
        try:
            base_contact_value = self._model.get_sensor_value("base_contact", data)
            if base_contact_value.ndim == 0:
                base_contact = np.array([base_contact_value > 0.01], dtype=bool)
            elif base_contact_value.shape[0] != self._num_envs:
                base_contact = np.full(self._num_envs, base_contact_value.flatten()[0] > 0.01, dtype=bool)
            else:
                base_contact = (base_contact_value > 0.01).flatten()[:self._num_envs]
        except Exception:
            base_contact = np.zeros(self._num_envs, dtype=bool)
        terminated = base_contact.copy()
        # 摔倒检测: 倾斜角度过大
        if projected_gravity is not None:
            gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
            gz = projected_gravity[:, 2]
            tilt_angle = np.arctan2(gxy, np.abs(gz))
            terminated = np.logical_or(terminated, tilt_angle > np.deg2rad(80))  # 80°: 坡道+bumps需要宽松阈值
        # 关节速度异常
        if joint_vel is not None:
            vel_max = np.abs(joint_vel).max(axis=1)
            vel_overflow = vel_max > self._cfg.max_dof_vel
            vel_extreme = np.isnan(joint_vel).any(axis=1) | np.isinf(joint_vel).any(axis=1)
            terminated = np.logical_or(terminated, vel_overflow | vel_extreme)
        # ===== 超出边界检测 (竞赛规则: 超出边界→扣除所有得分) =====
        bounds = getattr(self._cfg, 'course_bounds', None)
        if bounds is not None and robot_xy is not None and current_z is not None:
            oob_x = (robot_xy[:, 0] < bounds.x_min) | (robot_xy[:, 0] > bounds.x_max)
            oob_y = (robot_xy[:, 1] < bounds.y_min) | (robot_xy[:, 1] > bounds.y_max)
            oob_z = current_z < bounds.z_min  # 跌落判定
            oob = oob_x | oob_y | oob_z
            terminated = np.logical_or(terminated, oob)
            # 记录OOB到info中供reward使用
            state.info["oob_terminated"] = oob
        # Grace period: 前N步不判终止，让agent学会站立
        grace_steps = getattr(self._cfg, 'grace_period_steps', 0)
        if grace_steps > 0:
            ep_steps = state.info.get("steps", np.zeros(self._num_envs, dtype=np.int32))
            in_grace = ep_steps < grace_steps
            terminated = np.where(in_grace, False, terminated)
        # NaN envs ALWAYS terminate (bypass grace period)
        nan_terminated = state.info.get("nan_terminated", np.zeros(self._num_envs, dtype=bool))
        terminated = np.logical_or(terminated, nan_terminated)
        return state.replace(terminated=terminated)

    # ============================================================
    # 奖励计算
    # ============================================================

    def _compute_reward(self, data, info, velocity_commands, base_lin_vel, gyro,
                         projected_gravity, joint_vel, distance_to_target, position_error,
                         reached_wp, terminated, robot_heading, robot_xy, current_z,
                         speed_xy, wp_bonus, celeb_bonus, spin_progress, spin_hold,
                         celeb_speed_pen, in_celeb):
        scales = self._cfg.reward_config.scales
        n = self._num_envs

        # ===== 累积奖金追踪 (终止时得分清零用) =====
        accumulated_bonus = info.get("accumulated_bonus", np.zeros(n, dtype=np.float32))

        # ===== 导航跟踪 (仅非庆祝阶段) =====
        position_tracking = np.exp(-distance_to_target / 5.0)
        fine_position_tracking = np.where(distance_to_target < 2.5, np.exp(-distance_to_target / 0.5), 0.0)

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
        wp_facing = scales.get("waypoint_facing", 0.6) * heading_tracking

        # 存活奖励
        alive_bonus = scales.get("alive_bonus", 0.05) * np.ones(n, dtype=np.float32)

        # ===== 稳定性惩罚 (数值安全: 先clip再平方, 防止NaN溢出) =====
        orientation_penalty = np.square(projected_gravity[:, 0]) + np.square(projected_gravity[:, 1])
        lin_vel_z_penalty = np.square(np.clip(base_lin_vel[:, 2], -50.0, 50.0))
        ang_vel_xy_penalty = np.sum(np.square(np.clip(gyro[:, :2], -50.0, 50.0)), axis=1)
        torque_penalty = np.sum(np.square(np.clip(data.actuator_ctrls, -200.0, 200.0)), axis=1)
        safe_joint_vel = np.clip(joint_vel, -100.0, 100.0)
        dof_vel_penalty = np.sum(np.square(safe_joint_vel), axis=1)
        last_dof_vel = info.get("last_dof_vel", np.zeros_like(joint_vel))
        dof_acc_penalty = np.sum(np.square(safe_joint_vel - np.clip(last_dof_vel, -100.0, 100.0)), axis=1)
        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        # ===== 速度-距离耦合 =====
        desired_speed = np.clip(distance_to_target * 0.5, 0.05, 0.6)
        speed_excess = np.maximum(speed_xy - desired_speed, 0.0)
        near_target_speed_pen = scales.get("near_target_speed", -0.5) * speed_excess ** 2
        near_target_speed_pen = np.where(in_celeb, 0.0, near_target_speed_pen)

        # ===== 爬坡高度进步 =====
        last_z = info.get("last_z", current_z.copy())
        z_delta = current_z - last_z
        info["last_z"] = current_z.copy()
        height_progress = scales.get("height_progress", 8.0) * np.maximum(z_delta, 0.0)

        # ===== 地形里程碑 =====
        current_y = robot_xy[:, 1]
        milestones_reached = info.get("milestones_reached", np.zeros((n, 2), dtype=bool))
        traversal_total = np.zeros(n, dtype=np.float32)
        m1 = (current_y > 4.0) & (current_z > 0.3)
        m1_first = m1 & ~milestones_reached[:, 0]
        milestones_reached[:, 0] |= m1
        traversal_total += np.where(m1_first, scales.get("traversal_bonus", 15.0), 0.0)
        m2 = (current_y > 6.5) & (current_z > 0.8)
        m2_first = m2 & ~milestones_reached[:, 1]
        milestones_reached[:, 1] |= m2
        traversal_total += np.where(m2_first, scales.get("traversal_bonus", 15.0), 0.0)
        info["milestones_reached"] = milestones_reached

        # ===== 竞赛得分区 (被动收集) =====
        smiley_total = np.zeros(n, dtype=np.float32)
        red_packet_total = np.zeros(n, dtype=np.float32)
        if self.has_scoring_zones:
            smileys_reached = info.get("smileys_reached", np.zeros((n, self.num_smileys), dtype=bool))
            for i in range(self.num_smileys):
                d = np.linalg.norm(robot_xy - self.smiley_centers[i][np.newaxis, :], axis=1)
                first = (d < self.smiley_radius) & ~smileys_reached[:, i]
                smileys_reached[:, i] |= (d < self.smiley_radius)
                smiley_total += np.where(first, scales.get("smiley_bonus", 20.0), 0.0)
            info["smileys_reached"] = smileys_reached

            red_packets_reached = info.get("red_packets_reached", np.zeros((n, self.num_red_packets), dtype=bool))
            for i in range(self.num_red_packets):
                d = np.linalg.norm(robot_xy - self.red_packet_centers[i][np.newaxis, :], axis=1)
                first = (d < self.red_packet_radius) & ~red_packets_reached[:, i]
                red_packets_reached[:, i] |= (d < self.red_packet_radius)
                red_packet_total += np.where(first, scales.get("red_packet_bonus", 10.0), 0.0)
            info["red_packets_reached"] = red_packets_reached

        scoring_rewards = smiley_total + red_packet_total

        # ===== Zone吸引力: 未收集zone产生接近delta奖励 =====
        # Delta-based: 只奖励靠近zone的运动, 站着不动=0 (Anti-Lazy)
        zone_approach_reward = np.zeros(n, dtype=np.float32)
        zone_approach_scale = scales.get("zone_approach", 3.0)
        if self.has_scoring_zones and zone_approach_scale > 0:
            # 计算当前到每个zone的距离
            num_zones = self.num_smileys + self.num_red_packets
            current_zone_dists = np.zeros((n, num_zones), dtype=np.float32)
            zone_collected = np.zeros((n, num_zones), dtype=bool)
            
            smileys_reached = info.get("smileys_reached", np.zeros((n, self.num_smileys), dtype=bool))
            for i in range(self.num_smileys):
                current_zone_dists[:, i] = np.linalg.norm(robot_xy - self.smiley_centers[i][np.newaxis, :], axis=1)
                zone_collected[:, i] = smileys_reached[:, i]
            
            red_packets_reached = info.get("red_packets_reached", np.zeros((n, self.num_red_packets), dtype=bool))
            for i in range(self.num_red_packets):
                idx = self.num_smileys + i
                current_zone_dists[:, idx] = np.linalg.norm(robot_xy - self.red_packet_centers[i][np.newaxis, :], axis=1)
                zone_collected[:, idx] = red_packets_reached[:, i]
            
            # Delta = last_distance - current_distance (正值 = 靠近)
            last_zone_dists = info.get("last_zone_dists", current_zone_dists.copy())
            zone_deltas = last_zone_dists - current_zone_dists  # 正 = 靠近zone
            
            # 只在3m内激活, 只奖励接近方向, clip防止冲击
            for i in range(num_zones):
                in_range = current_zone_dists[:, i] < 3.5  # 3.5m激活半径
                uncollected = ~zone_collected[:, i]
                active = in_range & uncollected
                delta_reward = np.clip(zone_deltas[:, i] * zone_approach_scale * 10.0, -0.1, 0.5)
                zone_approach_reward += np.where(active, delta_reward, 0.0)
            
            info["last_zone_dists"] = current_zone_dists

        # ===== 摆动相接触惩罚 =====
        swing_penalty = scales.get("swing_contact_penalty", -0.15) * self._compute_swing_contact_penalty(data, joint_vel)

        # ===== 脚部离地高度奖励 (鼓励抬脚过障碍) =====
        foot_clearance_scale = scales.get("foot_clearance", 0.0)
        if foot_clearance_scale > 0:
            foot_forces = self._get_foot_contact_forces(data)
            force_mag = np.linalg.norm(foot_forces, axis=2)  # [n, 4]
            in_swing = force_mag < 1.0  # 摆动相 = 无地面接触力
            calf_indices = [2, 5, 8, 11]
            calf_vel = np.abs(joint_vel[:, calf_indices])  # [n, 4]
            # 奖励摆动相的小腿关节角速度 (鼓励积极抬腿, 而非拖地)
            foot_clearance_reward = foot_clearance_scale * np.sum(
                in_swing.astype(np.float32) * np.clip(calf_vel, 0.0, 5.0) * 0.2, axis=1
            )
        else:
            foot_clearance_reward = np.zeros(n, dtype=np.float32)

        # ===== 累积奖金更新 (用于终止时得分清零) =====
        # 本步新增的所有一次性奖金 (终止的env不计入新奖金)
        step_bonus = np.where(terminated, 0.0, wp_bonus + celeb_bonus + scoring_rewards + traversal_total)
        accumulated_bonus = accumulated_bonus + step_bonus
        info["accumulated_bonus"] = accumulated_bonus

        # ===== 终止惩罚 + 得分清零 =====
        # 竞赛规则: "超出边界/摔倒行为 → 扣除本Section所有得分"
        # 实现: 基础惩罚 + 扣回所有累积的一次性奖金 (包括本步)
        base_termination = scales.get("termination", -50.0)
        score_clear_penalty = np.where(terminated, -accumulated_bonus, 0.0)
        termination_penalty = np.where(terminated, base_termination, 0.0) + score_clear_penalty

        # ===== 惩罚汇总 =====
        penalties = (
            scales.get("orientation", -0.03) * orientation_penalty
            + scales.get("lin_vel_z", -0.15) * lin_vel_z_penalty
            + scales.get("ang_vel_xy", -0.02) * ang_vel_xy_penalty
            + scales.get("torques", -1e-5) * torque_penalty
            + scales.get("dof_vel", -5e-5) * dof_vel_penalty
            + scales.get("dof_acc", -2.5e-7) * dof_acc_penalty
            + scales.get("action_rate", -0.01) * action_rate_penalty
            + termination_penalty
            + near_target_speed_pen
            + swing_penalty
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
            + spin_progress
            + spin_hold
            + celeb_speed_pen
            + height_progress
            + traversal_total
            + scoring_rewards
            + zone_approach_reward
            + foot_clearance_reward
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
            "spin_progress": spin_progress,
            "spin_hold": spin_hold,
            "celeb_speed_penalty": celeb_speed_pen,
            "smiley_bonus": smiley_total,
            "red_packet_bonus": red_packet_total,
            "zone_approach": zone_approach_reward,
            "height_progress": height_progress,
            "traversal_bonus": traversal_total,
            "penalties": penalties,
            "termination": termination_penalty,
            "swing_contact_penalty": swing_penalty,
            "foot_clearance": foot_clearance_reward,
            "score_clear_penalty": score_clear_penalty,
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

        # 归一化四元数
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

        # 初始目标: 第一个航点
        first_wp = self.waypoints[0]
        pose_commands = np.column_stack([
            np.tile(first_wp, (num_envs, 1)),
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
        heading_diff = self._wrap_angle(np.zeros(num_envs) - robot_heading)

        # 观测
        obs = np.concatenate([
            base_lin_vel * cfg.normalization.lin_vel,
            gyro * cfg.normalization.ang_vel,
            projected_gravity,
            joint_pos_rel * cfg.normalization.dof_pos,
            joint_vel_r * cfg.normalization.dof_vel,
            np.zeros((num_envs, self._num_action), dtype=np.float32),
            velocity_commands * self.commands_scale,
            position_error / 5.0,
            (heading_diff / np.pi)[:, np.newaxis],
            np.clip(distance_to_target / 5.0, 0, 1)[:, np.newaxis],
            reached.astype(np.float32)[:, np.newaxis],
            np.zeros((num_envs, 1), dtype=np.float32),  # celeb_progress = 0
        ], axis=-1)
        assert obs.shape == (num_envs, 54), f"reset obs shape {obs.shape} != ({num_envs}, 54)"

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
            # 航点导航
            "wp_idx": np.zeros(num_envs, dtype=np.int32),
            # 庆祝状态机
            "celeb_state": np.full(num_envs, CELEB_IDLE, dtype=np.int32),
            "celeb_init_heading": np.zeros(num_envs, dtype=np.float32),
            "celeb_target_heading": np.zeros(num_envs, dtype=np.float32),
            "celeb_hold_count": np.zeros(num_envs, dtype=np.int32),
            # 得分区追踪
            "smileys_reached": np.zeros((num_envs, self.num_smileys), dtype=bool),
            "red_packets_reached": np.zeros((num_envs, self.num_red_packets), dtype=bool),
            # 竞赛规则: 累积奖金追踪 (终止时清零用)
            "accumulated_bonus": np.zeros(num_envs, dtype=np.float32),
            "oob_terminated": np.zeros(num_envs, dtype=bool),
        }

        return obs, info
