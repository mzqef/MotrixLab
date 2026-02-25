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
VBot Section013 分阶段区域收集导航环境 - 金球得分区 + 跳跃庆祝
架构: 与Section011共享 — 69维观测, 分阶段导航, 庆祝FSM

竞赛规则 (Section 3 = 25分):
  Phase APPROACH: 入口平台 → 通过坡道/hfield → 接近金球区
  Phase BALLS:    碰到最近的一个金球即可 (碰到滚球+不摔倒 → +15分)
  Phase CLIMB:    碰到任一金球后 → 到达最终平台
  Phase CELEBRATION: 在最终平台上跳跃庆祝 (10次)

导航目标: 当前阶段最近的未收集区域中心
wp_idx = balls_collected + platform_reached (0-4)
"""

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection013EnvCfg, TerrainScaleHelper

# ============================================================
# 庆祝状态机常量 (与Section011一致)
# ============================================================
CELEB_IDLE = 0        # 未开始庆祝
CELEB_TURNING = 1        # 在平台上, 正在执行庆祝动作
CELEB_SETTLING = 2     # 完成一次动作后等待稳定
CELEB_DONE = 3        # 所有庆祝动作完成

# 机器人躯体尺寸 (与Section011一致, 用于footprint检测)
ROBOT_HALF_X = 0.25  # 前后半长
ROBOT_HALF_Y = 0.15  # 左右半宽


@registry.env("vbot_navigation_section013", "np")
class VBotSection013Env(NpEnv):
    """
    VBot Section03 分阶段区域收集导航 + 跳跃庆祝
    地形: 入口平台 + 高台阶 + 21.8°坡道 + hfield + 3金球 + 最终平台(顶面z=1.494)
    观测: 69维 (与Section011一致, 支持checkpoint迁移)
    """
    _cfg: VBotSection013EnvCfg

    def __init__(self, cfg: VBotSection013EnvCfg, num_envs: int = 1):
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
        # 69维观测 = 54(base) + 3(trunk_acc) + 12(actuator_torques) — 与Section011一致
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

        # 多地形动态action_scale
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
        self._init_foot_contact()

    def _init_foot_contact(self):
        self.foot_sensor_names = ["FR_foot_contact", "FL_foot_contact", "RR_foot_contact", "RL_foot_contact"]
        self.num_foot_check = 4
        self._gait_phase_counter = None
        # 关节扭矩归一化限制
        self.torque_limits = np.array([17, 17, 34, 17, 17, 34, 17, 17, 34, 17, 17, 34], dtype=np.float32)

    def _init_scoring_zones(self, cfg):
        sz = cfg.scoring_zones
        self.ball_centers = np.array(sz.ball_centers, dtype=np.float32)  # (N, 2)
        self.ball_radius = 0.3  # 后备半径 (zone_approach shaping用, footprint检测不依赖此值)
        self.celebration_center = np.array(sz.celebration_center, dtype=np.float32)
        self.celebration_radius = sz.celebration_radius
        self.celebration_min_z = sz.celebration_min_z
        self.has_scoring_zones = True
        self.num_balls = len(sz.ball_centers)
        # 球区间隙中心 (用于gap alignment shaping)
        self._gap_centers_x = np.array([-1.5, 1.5], dtype=np.float32)
        self._ball_zone_enter_y = 30.4
        self._ball_zone_exit_y = 31.9
        # 预计算左/中/右球索引 (用于ordered targeting)
        ball_xs = self.ball_centers[:, 0]
        self._ball_middle_idx = int(np.argmin(np.abs(ball_xs)))
        side_mask = np.ones(self.num_balls, dtype=bool)
        side_mask[self._ball_middle_idx] = False
        side_idxs = np.where(side_mask)[0]
        self._ball_left_idx = int(side_idxs[0] if ball_xs[side_idxs[0]] < ball_xs[side_idxs[1]] else side_idxs[1])
        self._ball_right_idx = int(side_idxs[0] if ball_xs[side_idxs[0]] > ball_xs[side_idxs[1]] else side_idxs[1])
        print(f"[Info] Section013 得分区: {self.num_balls}个金球 (L={self._ball_left_idx}, M={self._ball_middle_idx}, R={self._ball_right_idx})")

    def _init_waypoints(self, cfg):
        """初始化分阶段导航系统
        Phase APPROACH (-1): 入口 → 球区接近
        Phase BALLS (0):     收集3个金球 (任意顺序)
        Phase CLIMB (1):     到达最终平台
        Phase CELEBRATION (2): 跳跃庆祝
        """
        self.PHASE_APPROACH = -1
        self.PHASE_BALLS = 0
        self.PHASE_CLIMB = 1
        self.PHASE_CELEBRATION = 2
        self.NUM_WAYPOINTS = 4  # 3球 + 1平台

        wn = cfg.waypoint_nav
        self.wp_radius = wn.waypoint_radius
        self.wp_final_radius = wn.final_radius
        self.celeb_turn_threshold = getattr(wn, 'celebration_turn_threshold', 1.85)
        self.required_turns = getattr(wn, 'required_turns', 10)
        self.celeb_settle_z = getattr(wn, 'celebration_settle_z', 1.75)
        print(f"[Info] 分阶段导航: {self.num_balls}金球 → 最终平台 → 庆祝({self.required_turns}次)")

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
        has_contact = force_magnitudes > 0.5
        has_high_vel = foot_vel > 2.0
        swing_contact = np.logical_and(has_contact, has_high_vel).astype(np.float32)
        penalty = np.sum(swing_contact * np.square(foot_vel) / 10.0, axis=1)
        return penalty

    # ============================================================
    # 步态质量奖励
    # ============================================================

    def _compute_gait_rewards(self, data: mtx.SceneData) -> dict:
        """~2 feet on ground is ideal for blind terrain traversal."""
        foot_forces = self._get_foot_contact_forces(data)
        force_mag = np.linalg.norm(foot_forces, axis=2)
        in_contact = (force_mag > 0.5).astype(np.float32)
        total_contacts = np.sum(in_contact, axis=1)
        stance_raw = 1.0 - np.abs(total_contacts - 2.0) * 0.33
        stance_reward = np.clip(stance_raw, 0.0, 1.0)
        return {"stance_reward": stance_reward}

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

    @staticmethod
    def _footprint_contains_point(center_2d, robot_xy, robot_heading):
        """检测球心点是否落在机器人矩形足印(XY平面投影)内 (与Section011一致)."""
        dx = center_2d[0] - robot_xy[:, 0]
        dy = center_2d[1] - robot_xy[:, 1]
        cos_h = np.cos(robot_heading)
        sin_h = np.sin(robot_heading)
        local_x = dx * cos_h + dy * sin_h
        local_y = -dx * sin_h + dy * cos_h
        return (np.abs(local_x) <= ROBOT_HALF_X) & (np.abs(local_y) <= ROBOT_HALF_Y)

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
        for start in quat_indices:
            q = dof_pos[:, start:start+4]
            norms = np.linalg.norm(q, axis=1, keepdims=True)
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
    # 航点 & 庆祝状态转换
    # ============================================================

    def _update_waypoint_state(self, info, robot_xy, robot_heading, current_z):
        """
        分阶段区域收集 & 庆祝状态机。

        Phase APPROACH (-1): 入口 → 球区接近 (y >= 30.0)
        Phase BALLS (0):     碰到最近的金球 (任意1个即可, 距离<ball_radius即收集)
        Phase CLIMB (1):     到达最终平台 (z > celebration_min_z)
        Phase CELEBRATION (2): 跳跃庆祝

        wp_idx = balls_count + platform_reached (0-4)
        """
        nav_phase = info["nav_phase"]
        celeb_state = info["celeb_state"]

        n = self._num_envs
        scales = self._cfg.reward_config.scales

        zone_bonus = np.zeros(n, dtype=np.float32)
        ball_bonus_tb = np.zeros(n, dtype=np.float32)
        phase_bonus = np.zeros(n, dtype=np.float32)
        celeb_bonus = np.zeros(n, dtype=np.float32)
        turn_reward = np.zeros(n, dtype=np.float32)

        # --- Phase APPROACH → BALLS: 接近球区 (y >= 30.0) ---
        in_approach = (nav_phase == self.PHASE_APPROACH)
        entered_ball_region = in_approach & (robot_xy[:, 1] >= 30.0)
        if np.any(entered_ball_region):
            nav_phase = np.where(entered_ball_region, self.PHASE_BALLS, nav_phase)
            phase_bonus += np.where(entered_ball_region, scales.get("phase_bonus", 15.0), 0.0)

        # --- Phase BALLS: 金球收集 - footprint覆盖球心点 (与Section011笑脸一致) ---
        balls_reached = info["balls_reached"]
        can_collect = (nav_phase <= self.PHASE_CLIMB)
        if np.any(can_collect) and self.has_scoring_zones:
            for i in range(self.num_balls):
                touched = self._footprint_contains_point(
                    self.ball_centers[i], robot_xy, robot_heading
                )
                first_collect = can_collect & touched & ~balls_reached[:, i]
                balls_reached[:, i] |= (can_collect & touched)
                ball_val = np.where(first_collect, scales.get("waypoint_bonus", 50.0), 0.0)
                zone_bonus += ball_val
                ball_bonus_tb += ball_val
        info["balls_reached"] = balls_reached

        # Phase BALLS → CLIMB: 碰到任一金球即可前进
        any_ball = np.any(balls_reached, axis=1)
        in_phase_balls = (nav_phase == self.PHASE_BALLS)
        balls_to_climb = in_phase_balls & any_ball
        if np.any(balls_to_climb):
            nav_phase = np.where(balls_to_climb, self.PHASE_CLIMB, nav_phase)
            phase_bonus += np.where(balls_to_climb, scales.get("phase_bonus", 15.0), 0.0)

        # --- Phase CLIMB: 到达最终平台 ---
        platform_reached = info.get("platform_reached", np.zeros(n, dtype=bool))
        in_phase_climb = (nav_phase == self.PHASE_CLIMB)
        if np.any(in_phase_climb):
            celeb_xy = self.celebration_center[:2]
            d_celeb = np.linalg.norm(robot_xy - celeb_xy[np.newaxis, :], axis=1)
            arrived = in_phase_climb & (d_celeb < self.wp_final_radius) & (current_z > self.celebration_min_z)
            first_arrive = arrived & ~platform_reached
            platform_reached |= arrived
            zone_bonus += np.where(first_arrive, scales.get("phase_bonus", 15.0), 0.0)
            nav_phase = np.where(arrived, self.PHASE_CELEBRATION, nav_phase)
        info["platform_reached"] = platform_reached

        # --- 更新 wp_idx ---
        balls_count = np.sum(balls_reached, axis=1).astype(np.int32)
        wp_idx = balls_count + platform_reached.astype(np.int32)
        info["wp_idx"] = wp_idx
        info["nav_phase"] = nav_phase

        # --- Phase CELEBRATION: 庆祝动作 ---
        in_celeb = (nav_phase == self.PHASE_CELEBRATION)
        turn_count = info["turn_count"]

        # IDLE -> TURNING
        start_celeb = in_celeb & (celeb_state == CELEB_IDLE)
        if np.any(start_celeb):
            celeb_state = np.where(start_celeb, CELEB_TURNING, celeb_state)

        # TURNING: 奖励向上运动
        turning = in_celeb & (celeb_state == CELEB_TURNING)
        if np.any(turning):
            # 最终平台 standing z ≈ 1.79 (1.494+0.3)
            z_above_standing = np.maximum(current_z - 1.7, 0.0)
            turn_reward += np.where(turning, scales.get("turn_reward", 10.0) * z_above_standing, 0.0)

            turned = turning & (current_z > self.celeb_turn_threshold)
            if np.any(turned):
                turn_count = np.where(turned, turn_count + 1, turn_count)
                celeb_bonus += np.where(turned, scales.get("per_turn_bonus", 60.0), 0.0)
                all_done = turned & (turn_count >= self.required_turns)
                still_turning = turned & (turn_count < self.required_turns)
                celeb_state = np.where(all_done, CELEB_DONE, celeb_state)
                celeb_state = np.where(still_turning, CELEB_SETTLING, celeb_state)
                celeb_bonus += np.where(all_done, scales.get("celebration_bonus", 140.0), 0.0)

        # SETTLING: 等待稳定, 然后重新进入TURNING
        landing = in_celeb & (celeb_state == CELEB_SETTLING)
        if np.any(landing):
            landed = landing & (current_z < self.celeb_settle_z)
            if np.any(landed):
                celeb_state = np.where(landed, CELEB_TURNING, celeb_state)

        info["turn_count"] = turn_count
        info["celeb_state"] = celeb_state

        wp_bonus = zone_bonus + phase_bonus
        info["_ball_bonus_tb"] = ball_bonus_tb
        info["_phase_bonus_tb"] = phase_bonus
        return info, wp_bonus, celeb_bonus, turn_reward

    def _get_current_target(self, info, robot_xy):
        """Target nearest uncollected ball during BALLS phase; celeb center otherwise."""
        nav_phase = info["nav_phase"]
        n = len(nav_phase)
        celeb_xy = self.celebration_center[:2]
        targets = np.tile(celeb_xy, (n, 1))

        if not self.has_scoring_zones:
            return targets

        def _apply_ordered_targets(env_mask, reached, centers,
                                    left_idx, middle_idx, right_idx):
            p = np.where(env_mask)[0]
            if len(p) == 0:
                return
            robot_x = robot_xy[p, 0]
            on_left = robot_x < 0
            first = np.where(on_left, left_idx, right_idx)
            second = np.full(len(p), middle_idx, dtype=np.int32)
            third = np.where(on_left, right_idx, left_idx)
            priority = np.stack([first, second, third], axis=1)
            reached_p = reached[p]
            reached_by_priority = np.array([
                reached_p[np.arange(len(p)), priority[:, r]] for r in range(3)
            ]).T
            uncollected = ~reached_by_priority
            has_any = np.any(uncollected, axis=1)
            first_uncollected_rank = np.argmax(uncollected, axis=1)
            target_zone = priority[np.arange(len(p)), first_uncollected_rank]
            assign = p[has_any]
            targets[assign] = centers[target_zone[has_any]]

        # Phase APPROACH + BALLS: 目标 = 最近金球
        _apply_ordered_targets(
            (nav_phase == self.PHASE_APPROACH) | (nav_phase == self.PHASE_BALLS),
            info["balls_reached"],
            self.ball_centers,
            self._ball_left_idx, self._ball_middle_idx, self._ball_right_idx)

        return targets

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
            root_pos = np.where(nan_mask[:, np.newaxis], np.array([[0.0, 26.0, 1.8, 0.0, 0.0, 0.0]]), root_pos)
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
        info, wp_bonus, celeb_bonus, turn_reward = \
            self._update_waypoint_state(info, robot_xy, robot_heading, current_z)

        # --- 当前导航目标 ---
        target_xy = self._get_current_target(info, robot_xy)
        in_celeb = (info["nav_phase"] >= self.PHASE_CELEBRATION)

        position_error = target_xy - robot_xy
        distance_to_target = np.linalg.norm(position_error, axis=1)

        pose_commands = np.column_stack([target_xy, np.zeros(self._num_envs, dtype=np.float32)])
        info["pose_commands"] = pose_commands

        current_wp_radius = np.where(
            info["nav_phase"] >= self.PHASE_CLIMB,
            self.wp_final_radius,
            self.wp_radius
        )
        reached_wp = distance_to_target < current_wp_radius

        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where((reached_wp | in_celeb)[:, np.newaxis], 0.0, desired_vel_xy)

        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = self._wrap_angle(desired_heading - robot_heading)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_wp | in_celeb, 0.0, desired_yaw_rate)

        velocity_commands = np.column_stack([desired_vel_xy, desired_yaw_rate[:, np.newaxis] if desired_yaw_rate.ndim == 1 else desired_yaw_rate])
        heading_diff = self._wrap_angle(desired_heading - robot_heading)

        # --- 69维观测 (与Section011完全一致的layout) ---
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        last_actions = info["current_actions"]

        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi

        # 盲走越障感知特征
        foot_forces_obs = self._get_foot_contact_forces(data)
        foot_contact = (np.linalg.norm(foot_forces_obs, axis=2) > 0.5).astype(np.float32)
        base_height_norm = np.clip((current_z - 0.5) / 1.2, -1.0, 1.0)[:, np.newaxis]

        # trunk加速度计 + 关节raw扭矩需求
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
            data, info, velocity_commands, base_lin_vel, gyro, projected_gravity,
            joint_vel, distance_to_target, position_error, reached_wp,
            terminated, robot_heading, robot_xy, current_z, speed_xy,
            wp_bonus, celeb_bonus, turn_reward, in_celeb
        )

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        # 庆祝完成后截断
        celeb_done = (info["celeb_state"] == CELEB_DONE)
        self._success_truncate = celeb_done

        # 停滞检测
        stag_cfg_window = getattr(cfg, 'stagnation_window_steps', 1000)
        stag_cfg_dist = getattr(cfg, 'stagnation_min_distance', 0.5)
        stag_cfg_grace = getattr(cfg, 'stagnation_grace_steps', 500)
        ep_steps = info.get("steps", np.zeros(self._num_envs, dtype=np.int32))
        anchor_xy = info["stagnation_anchor_xy"]
        anchor_step = info["stagnation_anchor_step"]
        dist_from_anchor = np.linalg.norm(robot_xy - anchor_xy, axis=1)
        moved_enough = dist_from_anchor >= stag_cfg_dist
        info["stagnation_anchor_xy"] = np.where(moved_enough[:, np.newaxis], robot_xy, anchor_xy)
        info["stagnation_anchor_step"] = np.where(moved_enough, ep_steps, anchor_step)
        steps_since_anchor = ep_steps - info["stagnation_anchor_step"]
        stagnant = (
            (steps_since_anchor > stag_cfg_window)
            & (ep_steps >= stag_cfg_grace)
            & ~in_celeb
        )
        self._stagnation_truncate = stagnant

        # 诊断指标
        current_y = robot_xy[:, 1]
        in_ball_zone = ((current_y >= self._ball_zone_enter_y) & (current_y <= self._ball_zone_exit_y)).astype(np.float32)
        max_y_reached = info.get("max_y_reached", current_y.copy())
        max_y_reached = np.maximum(max_y_reached, current_y)
        info["max_y_reached"] = max_y_reached

        state.info["metrics"] = {
            "distance_to_target": distance_to_target,
            "reached_fraction": reached_wp.astype(np.float32),
            "wp_idx_mean": info["wp_idx"].astype(np.float32),
            "nav_phase_mean": info["nav_phase"].astype(np.float32),
            "celeb_state_mean": info["celeb_state"].astype(np.float32),
            "turn_count_mean": info["turn_count"].astype(np.float32),
            "action_scale_mean": info["current_action_scale"].astype(np.float32).reshape(-1),
            "ball_zone_entry_frac": in_ball_zone,
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

        # === SOFT terminations (grace-protected) ===
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

    def _compute_reward(self, data, info, velocity_commands, base_lin_vel, gyro,
                         projected_gravity, joint_vel, distance_to_target, position_error,
                         reached_wp, terminated, robot_heading, robot_xy, current_z,
                         speed_xy, wp_bonus, celeb_bonus, turn_reward, in_celeb):
        scales = self._cfg.reward_config.scales
        n = self._num_envs

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
        wp_approach = np.clip(wp_delta * scales.get("waypoint_approach", 280.0), -0.5, 2.5)

        wp_facing = scales.get("waypoint_facing", 0.64) * heading_tracking

        # 存活奖励
        gz = np.clip(-projected_gravity[:, 2], 0.0, 1.0)
        upright_factor = np.where(gz > 0.9, 1.0, np.where(gz > 0.7, 0.5, 0.0))
        alive_decay_horizon = scales.get("alive_decay_horizon", 2400.0)
        ep_steps = info.get("steps", np.zeros(n, dtype=np.int32)).astype(np.float32)
        alive_time_decay = np.clip(1.0 - ep_steps / alive_decay_horizon, 0.0, 1.0)
        alive_bonus = scales.get("alive_bonus", 1.0) * upright_factor * alive_time_decay

        # ===== 稳定性惩罚 =====
        orientation_penalty = np.sum(np.square(projected_gravity[:, :2]), axis=1)

        # 坡道方向补偿 — section013: 21.8°坡道 y∈[27.5,29.5]
        current_y_for_slope = robot_xy[:, 1]
        on_ramp = (current_y_for_slope > 27.5) & (current_y_for_slope < 29.5)
        expected_gy = np.where(on_ramp, 0.371, 0.0)  # sin(21.8°) ≈ 0.371
        gy_error = np.abs(projected_gravity[:, 1] - expected_gy)
        slope_compensation = np.where(on_ramp, np.exp(-np.square(gy_error) / 0.05), 0.0)
        slope_orientation_reward = scales.get("slope_orientation", 0.04) * slope_compensation

        lin_vel_z_penalty = np.square(np.clip(base_lin_vel[:, 2], -50.0, 50.0))
        ang_vel_xy_penalty = np.sum(np.square(np.clip(gyro[:, :2], -50.0, 50.0)), axis=1)
        clipped_torques = self._get_actuator_torques(data)
        torque_penalty = np.sum(np.square(np.clip(clipped_torques, -200.0, 200.0)), axis=1)
        _joint_pos = self.get_dof_pos(data)
        dof_pos_penalty = np.sum(np.square(_joint_pos - self.default_angles), axis=1)
        safe_joint_vel = np.clip(joint_vel, -100.0, 100.0)
        dof_vel_penalty = np.sum(np.square(safe_joint_vel), axis=1)
        last_dof_vel = info.get("last_dof_vel", np.zeros_like(joint_vel))
        dof_acc_penalty = np.sum(np.square(safe_joint_vel - np.clip(last_dof_vel, -100.0, 100.0)), axis=1)
        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        # trunk冲击惩罚
        trunk_acc_r = self._get_trunk_acc(data)
        trunk_acc_mag = np.linalg.norm(trunk_acc_r, axis=1)
        impact_excess = np.maximum(trunk_acc_mag - 15.0, 0.0)
        impact_penalty = np.square(impact_excess) / 100.0

        # 扭矩饱和惩罚
        raw_torques = self._raw_torques
        saturation_ratio = np.abs(raw_torques) / self.torque_limits[np.newaxis, :]
        torque_sat_penalty = np.sum(np.maximum(saturation_ratio - 0.9, 0.0) ** 2, axis=1)

        # ===== 高度进步 (watermark) =====
        last_z = info.get("last_z", current_z.copy())
        z_delta = current_z - last_z
        info["last_z"] = current_z.copy()
        max_z_reached = info.get("max_z_reached", current_z.copy())
        z_above_max = np.maximum(current_z - max_z_reached, 0.0)
        max_z_reached = np.maximum(max_z_reached, current_z)
        info["max_z_reached"] = max_z_reached
        height_progress = scales.get("height_progress", 0.0) * z_above_max

        # 高度振荡惩罚
        z_osc = np.abs(z_delta)
        height_osc_penalty = scales.get("height_oscillation", -2.0) * np.maximum(z_osc - 0.015, 0.0)

        # ===== 地形里程碑 =====
        current_y = robot_xy[:, 1]
        milestones_reached = info.get("milestones_reached", np.zeros((n, 2), dtype=bool))
        traversal_total = np.zeros(n, dtype=np.float32)
        # 里程碑1: 通过坡道 (y > 29)
        m1 = (current_y > 29.0) & (current_z > 1.0)
        m1_first = m1 & ~milestones_reached[:, 0]
        milestones_reached[:, 0] |= m1
        traversal_total += np.where(m1_first, 15.0, 0.0)
        # 里程碑2: 通过球区 (y > 31.5)
        m2 = (current_y > 31.5) & (current_z > 0.8)
        m2_first = m2 & ~milestones_reached[:, 1]
        milestones_reached[:, 1] |= m2
        traversal_total += np.where(m2_first, 15.0, 0.0)
        info["milestones_reached"] = milestones_reached

        # ===== Zone吸引力 =====
        zone_approach_reward = np.zeros(n, dtype=np.float32)
        zone_approach_scale = scales.get("zone_approach", 75.0)
        if self.has_scoring_zones and zone_approach_scale > 0:
            balls_reached = info["balls_reached"]
            any_collected = np.any(balls_reached, axis=1)
            pre_celeb = (info["nav_phase"] <= self.PHASE_BALLS) & ~any_collected
            if np.any(pre_celeb):
                for i in range(self.num_balls):
                    d = np.linalg.norm(robot_xy - self.ball_centers[i][np.newaxis, :], axis=1)
                    last_key = f"last_ball_dist_{i}"
                    last_d = info.get(last_key, d.copy())
                    delta = last_d - d
                    in_range = d < 5.0
                    uncollected = ~balls_reached[:, i]
                    active = pre_celeb & in_range & uncollected
                    delta_reward = np.clip(delta * zone_approach_scale * 10.0, -0.3, 2.0)
                    zone_approach_reward += np.where(active, delta_reward, 0.0)
                    info[last_key] = d.copy()

        # ===== 球区连续shaping (稳定接触奖励) =====
        in_ball_zone = (robot_xy[:, 1] > self._ball_zone_enter_y) & (robot_xy[:, 1] < self._ball_zone_exit_y)
        # gap alignment
        dist_to_gaps = np.minimum(
            np.abs(robot_xy[:, 0] - self._gap_centers_x[0]),
            np.abs(robot_xy[:, 0] - self._gap_centers_x[1]),
        )
        ball_gap_alignment = scales.get("ball_gap_alignment", 2.0) * np.where(
            in_ball_zone, np.clip(1.0 - dist_to_gaps / 1.5, 0.0, 1.0), 0.0,
        )
        # 稳定接触奖励/不稳定接触惩罚
        diff = robot_xy[:, np.newaxis, :] - self.ball_centers[np.newaxis, :, :]
        dist_to_balls = np.linalg.norm(diff, axis=2)
        nearest_ball_dist = np.min(dist_to_balls, axis=1)
        ball_contact_proxy = np.where(
            in_ball_zone, np.clip(1.0 - nearest_ball_dist / 1.0, 0.0, 1.0), 0.0,
        )
        in_ball_zone_f = in_ball_zone.astype(np.float32)
        tilt_level = np.linalg.norm(projected_gravity[:, :2], axis=1)
        tilt_stable = np.clip(1.0 - tilt_level / 0.6, 0.0, 1.0)
        ang_speed = np.linalg.norm(gyro, axis=1)
        ang_stable = np.clip(1.0 - ang_speed / 3.0, 0.0, 1.0)
        stable_factor = np.clip(0.7 * tilt_stable + 0.3 * ang_stable, 0.0, 1.0)
        stable_ball_contact_reward = (
            scales.get("ball_contact_reward", 4.0)
            * ball_contact_proxy * stable_factor * in_ball_zone_f
        )
        unstable_ball_contact_penalty = (
            scales.get("ball_unstable_contact_penalty", -8.0)
            * ball_contact_proxy * (1.0 - stable_factor) * in_ball_zone_f
        )

        # ===== 摆动相接触惩罚 =====
        terrain_swing_scale = self._terrain_scale.compute_swing_scale(current_y, scales)
        swing_penalty = (
            scales.get("swing_contact_penalty", -0.003)
            * self._compute_swing_contact_penalty(data, joint_vel)
            * terrain_swing_scale
        )

        # ===== 脚部离地高度奖励 =====
        foot_clearance_scale = scales.get("foot_clearance", 0.0)
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

        # ===== 拖脚惩罚 =====
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

        # ===== 停滞渐进惩罚 =====
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

        # ===== 蹲坐惩罚 =====
        crouch_penalty_scale = scales.get("crouch_penalty", 0.0)
        if crouch_penalty_scale < 0:
            # 估算section013地面高度
            terrain_z_est = np.where(
                current_y < 27.5, 1.294,                                      # 入口平台
                np.where(current_y < 29.5,
                         1.294 + (current_y - 27.5) / 2.0 * 0.0,             # 坡道区 (z变化已含在实际地形)
                np.where(current_y < 31.0, 1.294,                             # hfield + 球区入口
                         1.294)))                                             # 球区+最终平台
            clearance = current_z - terrain_z_est
            min_clearance = 0.20
            crouch_penalty = np.where(clearance < min_clearance, crouch_penalty_scale, 0.0)
        else:
            crouch_penalty = np.zeros(n, dtype=np.float32)

        # ===== 步态质量 =====
        gait = self._compute_gait_rewards(data)
        gait_stance = scales.get("stance_ratio", 0.07) * gait["stance_reward"]

        # ===== 累积奖金 =====
        step_bonus = np.where(terminated, 0.0, wp_bonus + celeb_bonus + traversal_total)
        accumulated_bonus = accumulated_bonus + step_bonus
        info["accumulated_bonus"] = accumulated_bonus

        # ===== 终止惩罚 + 得分清零 =====
        base_termination = scales.get("termination", -150.0)
        score_clear_penalty = np.where(terminated, np.maximum(-0.3 * accumulated_bonus, -100.0), 0.0)
        termination_penalty = np.where(terminated, base_termination, 0.0) + score_clear_penalty

        # ===== 惩罚汇总 =====
        penalties = (
            scales.get("orientation", -0.026) * orientation_penalty
            + scales.get("lin_vel_z", -0.027) * lin_vel_z_penalty
            + scales.get("ang_vel_xy", -0.038) * ang_vel_xy_penalty
            + scales.get("torques", -5e-6) * torque_penalty
            + scales.get("dof_pos", -0.008) * dof_pos_penalty
            + scales.get("dof_vel", -3e-5) * dof_vel_penalty
            + scales.get("dof_acc", -1.5e-7) * dof_acc_penalty
            + scales.get("action_rate", -0.007) * action_rate_penalty
            + scales.get("impact_penalty", -0.1) * impact_penalty
            + scales.get("torque_saturation", -0.012) * torque_sat_penalty
            + termination_penalty
            + swing_penalty
            + height_osc_penalty
            + drag_foot_penalty
            + stagnation_penalty
            + crouch_penalty
            + unstable_ball_contact_penalty
        )

        # ===== 综合奖励 =====
        nav_reward = (
            scales.get("position_tracking", 0.26) * position_tracking
            + wp_approach
            + wp_facing
            + scales.get("forward_velocity", 3.16) * forward_velocity
            + alive_bonus
        )
        nav_reward = np.where(in_celeb, alive_bonus, nav_reward)

        reward = (
            nav_reward
            + wp_bonus
            + celeb_bonus
            + turn_reward
            + height_progress
            + slope_orientation_reward
            + traversal_total
            + zone_approach_reward
            + ball_gap_alignment
            + stable_ball_contact_reward
            + foot_clearance_reward
            + gait_stance
            + penalties
        )

        reward = np.where(terminated, termination_penalty, reward)
        reward = np.where(np.isfinite(reward), reward, -50.0)

        # TensorBoard
        ball_bonus_total = info.get("_ball_bonus_tb", np.zeros(n, dtype=np.float32))
        phase_bonus_total = info.get("_phase_bonus_tb", np.zeros(n, dtype=np.float32))
        info["Reward"] = {
            "position_tracking": scales.get("position_tracking", 0.26) * position_tracking,
            "heading_tracking": wp_facing,
            "forward_velocity": scales.get("forward_velocity", 3.16) * forward_velocity,
            "wp_approach": wp_approach,
            "alive_bonus": alive_bonus,
            "wp_bonus": wp_bonus,
            "celeb_bonus": celeb_bonus,
            "turn_reward": turn_reward,
            "ball_bonus": ball_bonus_total,
            "zone_approach": zone_approach_reward,
            "height_progress": height_progress,
            "height_oscillation": height_osc_penalty,
            "slope_orientation": slope_orientation_reward,
            "traversal_bonus": traversal_total,
            "ball_gap_alignment": ball_gap_alignment,
            "stable_ball_contact": stable_ball_contact_reward,
            "unstable_ball_contact": unstable_ball_contact_penalty,
            "penalties": penalties,
            "termination": termination_penalty,
            "swing_contact_penalty": swing_penalty,
            "foot_clearance": foot_clearance_reward,
            "score_clear_penalty": score_clear_penalty,
            "phase_completion_bonus": phase_bonus_total,
            "gait_stance": gait_stance,
            "impact_penalty": scales.get("impact_penalty", -0.1) * impact_penalty,
            "torque_saturation": scales.get("torque_saturation", -0.012) * torque_sat_penalty,
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

        # 初始朝向 +Y (yaw=π/2)
        init_quat = self._euler_to_quat(0, 0, np.pi / 2)
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

        # 初始目标
        init_balls = np.zeros((num_envs, self.num_balls), dtype=bool)
        init_nav_phase = np.full(num_envs, self.PHASE_APPROACH, dtype=np.int32)
        temp_info = {"nav_phase": init_nav_phase, "balls_reached": init_balls}
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

        heading_diff = self._wrap_angle(desired_heading - robot_heading)

        # 69维观测 (与Section011完全一致)
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
            np.zeros((num_envs, 3), dtype=np.float32),                # 3 trunk_acc
            np.zeros((num_envs, 12), dtype=np.float32),               # 12 actuator_torques
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
            # 庆祝状态机
            "celeb_state": np.full(num_envs, CELEB_IDLE, dtype=np.int32),
            "turn_count": np.zeros(num_envs, dtype=np.int32),
            # 得分区追踪
            "balls_reached": np.zeros((num_envs, self.num_balls), dtype=bool),
            # 竞赛规则: 累积奖金
            "accumulated_bonus": np.zeros(num_envs, dtype=np.float32),
            "oob_terminated": np.zeros(num_envs, dtype=bool),
            # 停滞检测
            "stagnation_anchor_xy": robot_init_xy.copy(),
            "stagnation_anchor_step": np.zeros(num_envs, dtype=np.int32),
            # 高度watermarks
            "max_z_reached": terrain_heights.copy(),
            "min_z_error": np.full(num_envs, 99.0, dtype=np.float32),
        }

        return obs, info
