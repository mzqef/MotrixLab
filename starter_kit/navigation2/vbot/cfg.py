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

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"

@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1

@dataclass
class TerrainZone:
    """Y-gated terrain zone → action_scale + clearance/swing modulation.

    Universal terrain zones covering navigation2 full course (y ≈ -3.5 to 34.33).
    Each section only encounters zones within its Y range.

    Clearance boost: robots inside this zone get foot_clearance multiplied by
    the value from reward_scales[clearance_boost_key]. Robots within pre_zone_margin
    meters before y_min get a fraction (foot_clearance_pre_zone_ratio) of the boost.

    Swing scale: robots inside this zone get swing_contact_penalty multiplied by
    the value from reward_scales[swing_scale_key] (typically < 1.0 to reduce penalty).
    """
    y_min: float
    y_max: float
    action_scale: float
    label: str = ""  # 仅用于调试/日志
    clearance_boost_key: str = ""   # v54: reward_scales key for foot clearance boost (empty = inactive)
    pre_zone_margin: float = 0.0    # v54: meters before y_min where transition zone starts
    post_zone_margin: float = 0.0   # v54: meters after y_max where transition zone continues
    swing_scale_key: str = ""       # v54: reward_scales key for swing contact penalty modifier (empty = inactive)

# 全程地形区域表 (从XML碰撞几何体提取)
# 每个section仅会命中其Y范围内的zone；未命中的zone不影响
DEFAULT_TERRAIN_ZONES: List[TerrainZone] = [
    # === Section 011 (y ≈ -3.5 → 8.83) ===
    TerrainZone(y_min=-1.8,  y_max=1.8,   action_scale=0.40, label="s011_bump",
               clearance_boost_key="foot_clearance_bump_boost", pre_zone_margin=0.7, post_zone_margin=0.0,
               swing_scale_key="swing_contact_bump_scale"),                              # 高度场凹凸区 (matches old hardcoded on_bump range)
    TerrainZone(y_min=2.0,   y_max=6.9,   action_scale=0.40, label="s011_slope"),      # 15°坡道
    # === Section 012 (y ≈ 8.83 → 25.33) ===
    TerrainZone(y_min=12.33, y_max=14.33, action_scale=0.50, label="s012_stairs_up",
               clearance_boost_key="foot_clearance_stair_boost", pre_zone_margin=1.0, post_zone_margin=0.3,
               swing_scale_key="swing_contact_stair_scale"),                             # 楼梯上行
    TerrainZone(y_min=14.33, y_max=21.33, action_scale=0.20, label="s012_bridge_valley"),  # 桥+河谷+平台
    TerrainZone(y_min=21.33, y_max=23.33, action_scale=0.20, label="s012_stairs_down",
               clearance_boost_key="foot_clearance_stair_boost", pre_zone_margin=1.0,
               swing_scale_key="swing_contact_stair_scale"),                             # 楼梯下行
    # === Section 013 (y ≈ 25.33 → 34.33) ===
    TerrainZone(y_min=27.33, y_max=30.83, action_scale=0.40, label="s013_ramp_hfield"), # 21.8°坡+高度场
    TerrainZone(y_min=30.83, y_max=33.83, action_scale=0.40, label="s013_ball_zone"),   # 球障碍+最终平台
]

@dataclass
class ControlConfig:
    # stiffness[N*m/rad] 使用XML中kp参数，仅作记录
    # damping[N*m*s/rad] 使用XML中kv参数，仅作记录
    action_scale: float = 0.25  # 平地navigation默认值
    # torque_limit[N*m] 使用XML forcerange参数
    dynamic_action_scale_enabled: bool = True
    flat_action_scale: float = 0.25  # 未命中任何zone时的fallback
    scale_switch_hold_steps: int = 120  # zone切换后保持N步才允许再次切换
    scale_interp_alpha: float = 0.08    # 指数平滑系数 (0=不变, 1=立即切换)
    terrain_zones: List[TerrainZone] = field(default_factory=lambda: list(DEFAULT_TERRAIN_ZONES))

@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.5]

    # 位置随机化范围 [x_min, y_min, x_max, y_max]
    pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]  # 在ground上随机分散20m x 20m范围

    # the default angles for all joints. key = joint name, value = target angle [rad]
    # 使用locomotion的关节角度配置
    default_joint_angles = {
        "FR_hip_joint": -0.0,     # 右前髋关节
        "FR_thigh_joint": 0.9,    # 右前大腿
        "FR_calf_joint": -1.8,    # 右前小腿
        "FL_hip_joint": 0.0,      # 左前髋关节
        "FL_thigh_joint": 0.9,    # 左前大腿
        "FL_calf_joint": -1.8,    # 左前小腿
        "RR_hip_joint": -0.0,     # 右后髋关节
        "RR_thigh_joint": 0.9,    # 右后大腿
        "RR_calf_joint": -1.8,    # 右后小腿
        "RL_hip_joint": 0.0,      # 左后髋关节
        "RL_thigh_joint": 0.9,    # 左后大腿
        "RL_calf_joint": -1.8,    # 左后小腿
    }

@dataclass
class Commands:
    # 目标位置相对于机器人初始位置的偏移范围 [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max]
    # dx/dy: 相对机器人初始位置的偏移（米）
    # yaw: 目标绝对朝向（弧度），水平方向随机
    pose_command_range = [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14]

@dataclass
class Normalization:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05

@dataclass
class Asset:
    body_name = "base"
    foot_names = ["FR", "FL", "RR", "RL"]
    terminate_after_contacts_on = ["collision_middle_box", "collision_head_box"]
    ground_subtree = "C_"  # 地形根节点，用于subtree接触检测

@dataclass
class Sensor:
    base_linvel = "base_linvel"
    base_gyro = "base_gyro"
    feet = ["FR", "FL", "RR", "RL"]  # 足部接触力传感器名称

# ============================================================
# 全局奖励尺度表 (Section011 v23b-T7 HP-search optimal 为基准)
# 所有section共享此基准值; 各section仅覆盖需要修改的key
# ============================================================
BASE_REWARD_SCALES: dict[str, float] = {
    # ===== v48-T14 AutoML winner (wp_idx_mean=0.484 @15M, best of 15 trials) =====
    # Source: automl_20260220_071134 trial_index=14
    # Key changes from v47: lighter penalties + stronger navigation pull
    # ===== 主动运动奖励 =====
    "forward_velocity": 3.163,                     # T14: +10% (v47=2.875)
    "waypoint_approach": 280.534,                  # T14: 1.68× stronger (v47=166.5)
    "waypoint_facing": 0.637,                      # T14: ~same (v47=0.61)
    "position_tracking": 0.259,                    # T14: lighter (v47=0.384)
    # ===== 存活奖励 =====
    "alive_bonus": 1.013,                          # T14: lighter (v47=1.446)
    # ===== Zone & Waypoint =====
    "zone_approach": 74.727,                       # T14: 2.13× stronger (v47=35.06)
    # ===== 地形适应 =====
    "height_progress": 0.0,                        # v52: not needed — existing forward rewards sufficient (zone_approach, wp_approach, forward_velocity)
    "height_approach": 0.0,                        # v52: not needed — robots already reach ramp with zone+forward rewards
    "height_oscillation": -2.0,                    # unchanged (not in search)
    # ===== 庆祝动作 (v58: X轴行走 + 蹲坐) =====
    "celeb_walk_approach": 200.0,                  # 接近X端点delta奖励
    "celeb_walk_bonus": 30.0,                      # 到达X端点一次性奖励
    "celeb_sit_reward": 5.0,                       # 蹲坐连续奖励 (每步 * z_below)
    "celebration_bonus": 50.0,                     # 庆祝完成大奖 (蹲坐足够久)
    # ===== 稳定性惩罚 =====
    "orientation": -0.026,                         # T14: ~same (v47=-0.027)
    "lin_vel_z": -0.027,                           # T14: 7.2× lighter (v47=-0.195) ← KEY
    "ang_vel_xy": -0.038,                          # T14: lighter (v47=-0.045)
    "torques": -5e-6,                              # unchanged (not in search)
    "dof_pos": -0.008,                             # v52: 轻微姿态惩罚(v51=-0.05太重导致冻结)
    "dof_vel": -3e-5,                              # unchanged (not in search)
    "dof_acc": -1.5e-7,                            # unchanged (not in search)
    "action_rate": -0.007,                         # T14: ~same (v47=-0.008)
    # ===== 传感器驱动惩罚 =====
    "impact_penalty": -0.100,                      # T14: slightly heavier (v47=-0.080)
    "torque_saturation": -0.012,                   # T14: 2.1× lighter (v47=-0.025) ← KEY
    # ===== 摆动相接触惩罚 =====
    "swing_contact_penalty": -0.003,               # T14: 10× lighter (v47=-0.031) ← KEY
    "swing_contact_bump_scale": 0.210,             # T14: lighter (v47=0.356)
    # ===== 终止 =====
    "termination": -150,                           # T14: 1.3× lighter (v47=-200) ← KEY
    "score_clear_factor": 0.0,                     # unchanged
    # ===== 一次性收集奖金 =====
    "waypoint_bonus": 50.046,                      # T14: ~same (v47=50.0)
    "phase_bonus": 13.067,                         # T14: lower (v47=25.0)
    # ===== 抬脚 & 步态 =====
    "stance_ratio": 0.070,                         # T14: +70% (v47=0.041)
    "foot_clearance": 0.219,                       # T14: 1.46× (v47=0.15)
    "foot_clearance_bump_boost": 7.167,            # T14: ~same (v47=8.0)
    "foot_clearance_stair_boost": 3.0,             # v54: centralized (was section012 fallback default)
    "foot_clearance_pre_zone_ratio": 0.5,          # v54→v56: matches old T10 pre-bump ratio (was 0.75, old code used 0.5)
    "swing_contact_stair_scale": 0.5,              # v54: centralized (was section012 fallback default)
    "alive_decay_horizon": 2383.0,                 # T14: 1.59× longer (v47=1500)
    "slope_orientation": 0.0,                      # v52: not needed — robots already climb ramp via forward rewards
    # ===== v49: 拖脚惩罚 + 停滞惩罚 =====
    "drag_foot_penalty": -0.15,                    # v49→v50: 支撑相低速腿惩罚 (每条拖地腿, 统一尺度)
    "stagnation_penalty": -0.5,                    # v49: 停滞渐进惩罚 (从50%窗口开始线性增长)
    # ===== v50: 蹲坐惩罚 =====
    "crouch_penalty": -1.5,                        # v52: 二值惩罚(v51=-5.0太重导致冻结; 配合dof_pos双管齐下)
}

@dataclass
class RewardConfig:
    scales: dict[str, float] = field(default_factory=lambda: dict(BASE_REWARD_SCALES))

@dataclass
class VBotStairsEnvCfg(EnvCfg):
    """Navigation2共享基类（仅供section011/012/013/long_course继承，不注册）"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs.xml"
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 20.0
    max_episode_steps: int = 2000
    sim_dt: float = 0.01
    ctrl_dt: float = 0.01
    reset_yaw_scale: float = 1.0
    max_dof_vel: float = 100.0

    grace_period_steps: int = 100  # 前100步(1秒) 仅保护base_contact和中等倾斜; 严重倾斜/OOB/NaN始终终止
    stagnation_window_steps: int = 600   # 6秒窗口: 若6秒内未移动足够距离则截断
    stagnation_min_distance: float = 0.6  # 6秒内至少走0.6m才算"在动"
    stagnation_grace_steps: int = 300     # 前3秒不检测停滞(给机器人起步时间)

    # 可配置终止参数 (env_overrides可覆盖)
    hard_tilt_deg: float = 70.0           # 硬终止倾斜角度 (>此角度立即终止)
    soft_tilt_deg: float = 50.0           # 软终止倾斜角度 (0=禁用, grace期后终止)
    enable_base_contact_term: bool = True  # 基座接触地面终止
    enable_stagnation_truncate: bool = True  # 停滞检测截断

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)


# ============================================================
# 通用有序航点模型 (可复用于任意section)
# ============================================================

@dataclass
class Waypoint:
    """单个导航航点 — 有序路线中的一个节点。

    kind:
      "virtual"  — 中转/控制点，到达即通过
      "reward"   — 奖励收集点，中心穿越即收集
      "goal"     — 终点目标，到达后进入庆祝

    z_min/z_max: 高度约束 (默认无约束)。
      例如桥面航点 z_min=2.3, 桥下航点 z_max=2.2。

    bonus_key: reward_scales 中的奖金key。
      若key不存在，使用 bonus_default。
    """
    xy: tuple                   # (x, y) 目标位置
    label: str = ""             # 人类可读标签 (调试/日志)
    kind: str = "virtual"       # "reward" | "virtual" | "goal"
    radius: float = 1.2         # 到达检测半径
    z_min: float = -999.0       # z下限 (-999 = 无约束)
    z_max: float = 999.0        # z上限 (999 = 无约束)
    bonus_key: str = ""         # reward_scales key
    bonus_default: float = 0.0  # 默认奖金值


@dataclass
class OrderedRoute:
    """有序导航路线: 严格顺序航点列表 + 庆祝配置。

    机器人必须按 waypoints 顺序依次到达每个航点。
    到达最后一个航点 (kind="goal") 后进入庆祝阶段。
    wp_idx = 已完成航点数量 (0 → len(waypoints))。

    可复用于任意section: 只需在cfg中定义不同的 waypoints 列表。
    """
    waypoints: list = field(default_factory=list)   # List[Waypoint]
    # 庆祝配置
    required_turns: int = 10                        # 庆祝动作次数 (可配置)
    celebration_turn_threshold: float = 1.55        # 动作检测z阈值
    celebration_settle_z: float = 1.50              # 稳定判定z阈值


@registry.envcfg("vbot_navigation_long_course")
@dataclass
class VBotLongCourseEnvCfg(VBotStairsEnvCfg):
    """VBot三段地形完整导航配置（比赛任务）- 使用全程合并地图"""
    # 使用scene_world_full.xml：三段地形碰撞体+视觉体合并
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_world_full.xml"
    max_episode_seconds: float = 180.0  # 全程180秒
    max_episode_steps: int = 18000  # 对应180秒 @ 100Hz

    @dataclass
    class InitState:
        # 起始位置：section01起始（高台中心）
        pos = [0.0, -2.5, 0.50]  # START平台中心, z=0.30 (v51: 0.35→0.30)
        pos_randomization_range = [-2.0, -0.5, 2.0, 0.5]  # X: ±2.0m (5m宽平台), Y: ±0.5m → y∈[-3.0,-2.0]

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }

    @dataclass
    class Commands:
        # 全程范围：航点系统内部管理，此处仅作记录
        pose_command_range = [-3.0, -3.0, -3.14, 3.0, 34.0, 3.14]

    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)


@registry.envcfg("vbot_navigation_section011")
@dataclass
class VBotSection011EnvCfg(VBotStairsEnvCfg):
    """VBot Section01（高台/坡道）导航配置

    XML地形信息 (0126_C_section01.xml):
    - START平台 (Adiban_001)：中心(0, -2.5, -0.25)，尺寸5×1m，顶面z=0
    - 高度场：中心(0, 0, 0)，范围x=±5m, y=±1.5m，高度0~0.277m
    - 地面平台：z=0
    - 15°坡道 (Adiban_003)：中心(0, 4.48, 0.41)
    - 高台 (Adiban_004)：中心(0, 7.83, 1.044)，尺寸(5×2.5×0.25)m，顶面z=1.294
    - 边界墙顶部 z≈2.45

    竞赛得分区 (Section1 = 20分满分):
    - 3个笑脸区 (各+4分=12分): OBJ坐标 (-3,0), (0,0), (3,0)  y∈[-1,1]
    - 3个红包区 (各+2分=6分):  OBJ坐标 (-3,4.4), (0,4.4), (3,4.4) y∈[3.4,5.4]
    - 庆祝动作 (+2分): 在"2026"平台(高台顶部)做出庆祝动作
    """
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section011.xml"
    max_episode_seconds: float = 120.0  # v44: 2分钟 — 停滞检测替代固定时间限制
    max_episode_steps: int = 12000

    @dataclass
    class InitState:
        # 竞赛正确起点：START平台 (Adiban_001), center=(0, -2.5), 顶面z=0
        # 竞赛规则: "初始点位置随机分布在'START'平台区域" y∈[-3.5, -1.5]
        # z=0.30: 略高于自然站立高度(~0.27m)，防止穿地，减少spawning下落冲击
        pos = [0.0, -2.5, 0.50]
        pos_randomization_range = [-2.0, -0.5, 2.0, 0.5]  # X: ±2.0m (5m宽平台), Y: ±0.5m → y∈[-3.0,-2.0]

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 从START平台到高台顶部：起始Y=-2.5 + 偏移10.3 → 目标Y=7.8（≈高台中心y=7.83）
        pose_command_range = [0.0, 10.3, 0.0, 0.0, 10.3, 0.0]

    @dataclass
    class ScoringZones:
        """竞赛得分区定义 (从OBJ顶点数据提取)"""
        # 3个笑脸区(各+4分): 位于height field上，y≈0处
        smiley_centers = [[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0]]  # [x, y]
        smiley_radius = 0.0  # 精确中心接触: 机器人矩形足印必须覆盖中心坐标 (footprint-contains检测)
        smiley_points = 4.0  # 每个笑脸区竞赛得分
        # 3个红包区(各+2分): 位于坡道上，y≈4.4处
        red_packet_centers = [[-3.0, 4.4], [0.0, 4.4], [3.0, 4.4]]  # [x, y]
        red_packet_radius = 0.0  # 精确中心接触: 机器人矩形足印必须覆盖中心坐标 (footprint-contains检测)
        red_packet_points = 2.0  # 每个红包区竞赛得分
        # 庆祝区(+2分): 高台顶部"2026"平台，y≈7.83, z>1.0
        celebration_center = [0.0, 7.83]
        celebration_radius = 1.5  # 高台顶部范围
        celebration_min_z = 1.0   # 必须在高台上
        celebration_points = 2.0

    @dataclass
    class CourseBounds:
        """赛道边界 (超出边界=终止+清零得分)

        从XML碰撞模型提取 (0126_C_section01.xml):
        - 边界墙 x=±5.25 (Abianjie), 内侧可用 x∈[-5.0, 5.0]
        - 后墙 y≈-3.75 → START起始 y≈-3.5
        - 高台末端 y≈8.83
        - 最低地面 z=0 (START平台顶面), 稍有宽容到 z=-0.5 (跌落判定)

        竞赛规则: "超出边界 / 摔倒行为 → 扣除本Section所有得分"
        """
        x_min: float = -5.2     # 左侧边界 (墙内)
        x_max: float = 5.2      # 右侧边界 (墙内)
        y_min: float = -4.0     # 后方边界 (START后墙)
        y_max: float = 9.5      # 前方边界 (高台末端+1m缓冲)
        z_min: float = -0.5     # 跌落判定 (低于地面0.5m)

    @dataclass
    class WaypointNav:
        """多航点导航配置: 3中心航点 + zone吸引力 + X轴行走蹲坐庆祝

        3中心航点 (验证过: 能可靠到达高台):
          WP0: 中心笑脸 (0, 0)    — 前进方向
          WP1: 中心红包 (0, 4.4)  — 继续前进(上坡)
          WP2: 高台     (0, 7.83) — 到达后进入庆祝

        庆祝流程 (v58): 到达高台 → 走向X轴端点 → 蹲坐 → 完成
        """
        # 航点坐标 [x, y] — 前进路线
        waypoints = [[0.0, 0.0], [0.0, 4.4], [0.0, 7.83]]
        # 航点到达半径
        waypoint_radius = 1.0  # 笑脸/红包zone半径较大，走到附近即可
        final_radius = 0.5     # 高台目标更精确
        # 庆祝参数 (v58: X轴行走 + 蹲坐)
        celeb_x_waypoint = [4.0, 7.83]   # X轴端点目标 (平台右侧, y=庆祝区中心y)
        celeb_walk_radius = 1.0           # X端点到达半径
        celeb_sit_z = 1.35                # 蹲坐z阈值 (平台顶z=1.294, 站立≈1.56, 蹲坐<1.35)
        celeb_sit_steps = 30              # 蹲坐保持步数 (30步=0.3秒)


    scoring_zones: ScoringZones = field(default_factory=ScoringZones)
    waypoint_nav: WaypointNav = field(default_factory=WaypointNav)
    course_bounds: CourseBounds = field(default_factory=CourseBounds)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)


@registry.envcfg("vbot_navigation_section012")
@dataclass
class VBotSection012EnvCfg(VBotStairsEnvCfg):
    """VBot Section02（楼梯/桥梁/障碍物）导航配置 — 桥优先多目标版

    竞赛规则 (Section 2 = 60分):
      +10: 通过波浪地形到达楼梯
      +5:  从左楼梯到达吊桥 / 从右楼梯到达河床
      +10: 经过吊桥途径拜年红包到达楼梯口
      +5:  从楼梯口下来到达丙午大吉平台
      +5:  庆祝动作
      +5:  经过河床到达楼梯 (右路线)
      +3×5=15: 河床石头上贺礼红包
      +5×2=10: 桥底下拜年红包

    桥优先策略 (固定主线):
      Phase 0: 通过波浪地形到达左楼梯底 (虚拟WP)
      Phase 1: 爬左楼梯到达桥入口 (虚拟WP)
      Phase 2: 过桥 — 3个虚拟导航点引导过桥 (entry→mid→exit)
      Phase 3: 收集桥上拜年红包 (+10分, 已经过桥时自然收集)
      Phase 4: 下左楼梯到达底部 (虚拟WP)
      Phase 5: 收集桥下拜年红包 (2个, 各+5, 过桥后激活)
      Phase 6: 到达丙午大吉平台
      Phase 7: 庆祝动作

    XML地形 (0131_C_section02_hotfix1.xml):
    - 入口平台：中心(0, 10.33, 1.294)
    - 左楼梯 (x=-3)：10级, ΔZ≈0.15/级, y=12.43→14.23, z=1.369→2.794
    - 拱桥 (x≈-3)：y=15.31→20.33, z≈2.51→2.71, 宽≈2.64m
    - 右楼梯 (x=2)：10级, ΔZ≈0.10/级, z=1.319→2.294
    - 5个球形障碍 (R=0.75)：右侧河床
    - 终点平台：(0, 24.33, 1.294)
    """
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section012.xml"
    max_episode_seconds: float = 60.0  # Section02复杂地形，需要更多时间
    max_episode_steps: int = 6000
    grace_period_steps: int = 100  # 前100步(1秒) 仅保护base_contact和中等倾斜
    @dataclass
    class InitState:
        # 起始位置：section02入口平台（来自section01高台，z≈1.294，机器人0.5m）
        pos = [0.0, 9.5, 1.8]
        pos_randomization_range = [-0.3, -0.3, 0.3, 0.3]  # 小范围随机±0.3m

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 不再使用固定offset目标; 目标由有序路线状态机动态指定
        pose_command_range = [0.0, 14.5, 0.0, 0.0, 14.5, 0.0]

    @dataclass
    class Section012Route(OrderedRoute):
        """右侧优先全收集路线: 石头红包 → 桥下红包 → 远端上桥 → 桥上红包 → 原路返回 → 终点 → 庆祝

        路线逻辑:
          1) 从入口右转，沿右侧河床收集5个石头贺礼红包 (固定顺序)
          2) 穿越河谷到左侧桥下，收集2个桥下拜年红包
          3) 从远端 (出口侧) 爬左楼梯上桥
          4) 过桥收集桥上拜年红包
          5) 原路返回到远端桥头，下楼梯
          6) 到达丙午大吉终点平台
          7) 庆祝跳跃 (~10次)

        航点坐标来自XML碰撞/可视体 (0131_C_section02_hotfix1.xml)。
        """
        waypoints: list = field(default_factory=lambda: [
            # === 阶段1: 右侧路线 — 收集石头贺礼红包 (5个, 各+3竞赛分) ===
            Waypoint(xy=(2.0, 12.0),    label="right_approach",     kind="virtual",  radius=1.5,
                     bonus_key="transit_bonus", bonus_default=10.0),
            Waypoint(xy=(0.36, 15.84),  label="stone_1_near_left",  kind="reward",   radius=1.2,
                     bonus_key="stone_bonus", bonus_default=10.0),
            Waypoint(xy=(3.5, 15.84),   label="stone_2_near_right", kind="reward",   radius=1.2,
                     bonus_key="stone_bonus", bonus_default=10.0),
            Waypoint(xy=(2.0, 17.83),   label="stone_3_center",     kind="reward",   radius=1.2,
                     bonus_key="stone_bonus", bonus_default=10.0),
            Waypoint(xy=(0.36, 19.72),  label="stone_4_far_left",   kind="reward",   radius=1.2,
                     bonus_key="stone_bonus", bonus_default=10.0),
            Waypoint(xy=(3.5, 19.72),   label="stone_5_far_right",  kind="reward",   radius=1.2,
                     bonus_key="stone_bonus", bonus_default=10.0),
            # === 阶段2: 桥下红包收集 (2个, 各+5竞赛分) ===
            Waypoint(xy=(-3.0, 19.5),   label="under_bridge_far",   kind="reward",   radius=1.5,
                     z_max=2.2, bonus_key="under_bridge_bonus", bonus_default=15.0),
            Waypoint(xy=(-3.0, 16.0),   label="under_bridge_near",  kind="reward",   radius=1.5,
                     z_max=2.2, bonus_key="under_bridge_bonus", bonus_default=15.0),
            # === 阶段3: 远端上桥 → 桥上红包 → 原路返回 ===
            Waypoint(xy=(-3.0, 22.5),   label="bridge_climb_base",  kind="virtual",  radius=1.5,
                     bonus_key="transit_bonus", bonus_default=10.0),
            Waypoint(xy=(-3.0, 20.0),   label="bridge_far_entry",   kind="virtual",  radius=1.5,
                     z_min=2.3, bonus_key="bridge_entry_bonus", bonus_default=20.0),
            Waypoint(xy=(-3.0, 17.83),  label="bridge_hongbao",     kind="reward",   radius=2.0,
                     z_min=2.3, bonus_key="bridge_hongbao_bonus", bonus_default=30.0),
            Waypoint(xy=(-3.0, 20.0),   label="bridge_turnaround",  kind="virtual",  radius=1.5,
                     z_min=2.3, bonus_key="transit_bonus", bonus_default=5.0),
            Waypoint(xy=(-3.0, 22.5),   label="bridge_descent",     kind="virtual",  radius=1.5,
                     bonus_key="transit_bonus", bonus_default=10.0),
            # === 阶段4: 终点 ===
            Waypoint(xy=(0.0, 24.33),   label="exit_platform",      kind="goal",     radius=0.8,
                     bonus_key="exit_bonus", bonus_default=30.0),
        ])
        required_turns: int = 10
        celebration_turn_threshold: float = 1.55
        celebration_settle_z: float = 1.50
    @dataclass
    class CourseBounds:
        """赛道边界 (超出=终止+清零得分)

        来自XML: 边界墙 x=±5.25, 入口平台y=8.83, 出口平台y=25.33
        """
        x_min: float = -5.2
        x_max: float = 5.2
        y_min: float = 8.5       # 入口平台前方
        y_max: float = 25.5      # 出口平台后方+缓冲
        z_min: float = 0.5       # 跌落到河谷底部以下判定 (河谷最低z≈1.19)

    ordered_route: Section012Route = field(default_factory=Section012Route)
    course_bounds: CourseBounds = field(default_factory=CourseBounds)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)


@registry.envcfg("vbot_navigation_section013")
@dataclass
class VBotSection013EnvCfg(VBotStairsEnvCfg):
    """VBot Section03（滚球/坡道/最终平台）分阶段区域收集导航配置

    XML地形信息 (0126_C_section03.xml):
    - 入口平台：z=1.294
    - 0.75m高隔离墙 (Cdiban_006)：y=27.58
    - 21.8°坡道 (Cdiban_002)
    - 高度场：中心(0, 29.33, 1.343)
    - 3个金球障碍 (R=0.75)：y=31.23，x=-3/0/3，球心z=0.844
    - 最终平台 (Cdiban_004)：中心(0, 32.33, 0.994)，顶面z=1.494
    - 终点墙：y=34.33

    竞赛得分区 (Section 3 = 25分):
    - 滚球通过 10~15分: 通过3金球区域(稳定接触+15, 无接触+10)
    - 随机地形 5分
    - 最终庆祝 5分

    架构: 与Section011共享分阶段导航+庆祝FSM, 69维观测
    """
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section013.xml"
    max_episode_seconds: float = 120.0  # 与section011一致: 停滞检测替代固定时间限制
    max_episode_steps: int = 12000
    @dataclass
    class InitState:
        # 起始位置：section03入口，地面z≈1.294，机器人高度0.5m以上
        pos = [0.0, 26.0, 1.8]
        pos_randomization_range = [-2.0, -0.5, 2.0, 0.5]  # X: ±2.0m (10m宽平台), Y: ±0.5m

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 固定目标：最终平台中心
        # 起始(0, 26.0) + 偏移(0, 6.33) → 目标(0, 32.33)
        pose_command_range = [0.0, 6.33, 0.0, 0.0, 6.33, 0.0]

    @dataclass
    class ScoringZones:
        """竞赛得分区定义 (Section 3 = 25分)

        3个金球得分区 (稳定通过+15分, 无接触通过+10分):
          球心坐标: (-3, 31.23), (0, 31.23), (3, 31.23), R=0.75
          通过间隙中心: x ≈ -1.5, 1.5
        庆祝区 (+5分): 最终平台上做庆祝动作
        """
        # 3个金球得分区: 机器人躯体覆盖球心点即视为接触 (footprint检测, 与Section011笑脸一致)
        ball_centers = [[-3.0, 31.23], [0.0, 31.23], [3.0, 31.23]]
        ball_points = 5.0  # 每个球区竞赛得分
        # 庆祝区 (+5分): 最终平台
        celebration_center = [0.0, 32.33]
        celebration_radius = 1.5
        celebration_min_z = 1.2  # 最终平台顶面z=1.494, 站立约+0.3
        celebration_points = 5.0

    @dataclass
    class CourseBounds:
        """赛道边界 (超出=终止+清零得分)

        从XML碰撞模型提取:
        - 边界墙 x=±5.25
        - 入口平台 y≈25.33
        - 终点墙 y≈34.33
        - 最低地面 z≈0.8 (坡道底部), 跌落判定z=-0.0
        """
        x_min: float = -5.2
        x_max: float = 5.2
        y_min: float = 24.5   # 入口平台前方
        y_max: float = 34.5   # 终点墙后方
        z_min: float = 0.0    # 坡道底部以下判定

    @dataclass
    class WaypointNav:
        """分阶段导航配置: 金球收集 + 最终平台 + 跳跃庆祝

        Phase APPROACH: 入口 → 坡道/hfield区域
        Phase BALLS: 收集3个金球得分区 (任意顺序)
        Phase CLIMB: 到达最终平台
        Phase CELEBRATION: 庆祝动作 (10次)
        """
        waypoint_radius = 1.2  # 金球区到达半径
        final_radius = 0.8     # 最终平台精确到达
        celebration_turn_threshold = 1.85  # 最终平台z=1.494, 站立+0.3=1.79, 跳+0.06
        required_turns = 10
        celebration_settle_z = 1.75  # 稳定判定

    scoring_zones: ScoringZones = field(default_factory=ScoringZones)
    waypoint_nav: WaypointNav = field(default_factory=WaypointNav)
    course_bounds: CourseBounds = field(default_factory=CourseBounds)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)


# ============================================================
# TerrainScaleHelper — 多zone动态action_scale (hold + 指数平滑)
# ============================================================

class TerrainScaleHelper:
    """Vectorized multi-zone dynamic action scale with hold + exponential smoothing."""

    def __init__(self, cc: ControlConfig):
        self.enabled = bool(cc.dynamic_action_scale_enabled)
        self.flat_scale = float(cc.flat_action_scale)
        self.hold_steps = int(cc.scale_switch_hold_steps)
        self.interp_alpha = float(np.clip(cc.scale_interp_alpha, 0.0, 1.0))
        self.fallback_scale = float(cc.action_scale)

        # 预编译zone数组用于向量化查询
        zones = cc.terrain_zones
        if zones:
            self._zone_y_min = np.array([z.y_min for z in zones], dtype=np.float32)
            self._zone_y_max = np.array([z.y_max for z in zones], dtype=np.float32)
            self._zone_scale = np.array([z.action_scale for z in zones], dtype=np.float32)
            self._num_zones = len(zones)
            # v54: pre-compile clearance boost and swing scale zone metadata
            self._zone_clearance_keys = [z.clearance_boost_key for z in zones]
            self._zone_pre_margins = np.array([z.pre_zone_margin for z in zones], dtype=np.float32)
            self._zone_post_margins = np.array([z.post_zone_margin for z in zones], dtype=np.float32)
            self._zone_swing_keys = [z.swing_scale_key for z in zones]
        else:
            self._zone_y_min = np.empty(0, dtype=np.float32)
            self._zone_y_max = np.empty(0, dtype=np.float32)
            self._zone_scale = np.empty(0, dtype=np.float32)
            self._num_zones = 0
            self._zone_clearance_keys = []
            self._zone_pre_margins = np.empty(0, dtype=np.float32)
            self._zone_post_margins = np.empty(0, dtype=np.float32)
            self._zone_swing_keys = []

    def _lookup_scale(self, probe_y: np.ndarray) -> np.ndarray:
        """Vectorized first-match zone lookup by Y position.

        Args:
            probe_y: (n,) robot Y positions.

        Returns:
            (n,) desired action scales. Falls back to flat_scale if no zone matches.
        """
        n = probe_y.shape[0]
        result = np.full(n, self.flat_scale, dtype=np.float32)

        if self._num_zones == 0:
            return result

        # (n, Z) boolean mask: which zones each env falls into
        in_zone = (probe_y[:, None] >= self._zone_y_min) & (probe_y[:, None] < self._zone_y_max)

        # 第一个匹配zone的索引 (zones按Y排列, 不重叠)
        any_match = np.any(in_zone, axis=1)
        if np.any(any_match):
            first_zone_idx = np.argmax(in_zone[any_match], axis=1)
            result[any_match] = self._zone_scale[first_zone_idx]

        return result

    def update(self, info: dict, probe_y: np.ndarray, num_envs: int) -> np.ndarray:
        """Update and return current action scale for all envs.

        Args:
            info: Environment info dict (persistent across steps). Stores internal
                  state keys: _ts_current, _ts_target, _ts_last_switch.
            probe_y: (n,) robot Y world positions.
            num_envs: Number of environments.

        Returns:
            (n,) current action scale array, exponentially smoothed.
        """
        # 初始化持久化buffer
        if "_ts_current" not in info:
            init = self.flat_scale if self.enabled else self.fallback_scale
            info["_ts_current"] = np.full(num_envs, init, dtype=np.float32)
            info["_ts_target"] = np.full(num_envs, init, dtype=np.float32)
            info["_ts_last_switch"] = np.zeros(num_envs, dtype=np.int32)

        current = info["_ts_current"]

        if not self.enabled:
            current[:] = self.fallback_scale
            return current

        desired = self._lookup_scale(probe_y)

        steps = info.get("steps", np.zeros(num_envs, dtype=np.int32))
        hold_ok = (steps - info["_ts_last_switch"]) >= self.hold_steps
        need_switch = hold_ok & (np.abs(info["_ts_target"] - desired) > 1e-6)
        if np.any(need_switch):
            info["_ts_target"][need_switch] = desired[need_switch]
            info["_ts_last_switch"][need_switch] = steps[need_switch]

        alpha = self.interp_alpha
        current[:] = (1.0 - alpha) * current + alpha * info["_ts_target"]

        # 也存入 info 以便外部观测 (TensorBoard, debug logging等)
        info["current_action_scale"] = current
        return current

    def compute_clearance_boost(self, probe_y: np.ndarray, scales: dict) -> np.ndarray:
        """Compute foot clearance boost multiplier based on terrain zones.

        For each zone with a clearance_boost_key:
        - Robots inside the zone get the full boost value from scales[key].
        - Robots within pre_zone_margin before the zone get boost * pre_zone_ratio.
        - All other robots get 1.0 (no boost).

        Multiple zones stack via max (highest boost wins).

        Args:
            probe_y: (n,) robot Y positions.
            scales: reward_scales dict.

        Returns:
            (n,) clearance multiplier array (>= 1.0).
        """
        n = probe_y.shape[0]
        result = np.ones(n, dtype=np.float32)
        if self._num_zones == 0:
            return result

        pre_zone_ratio = float(scales.get("foot_clearance_pre_zone_ratio", 0.5))

        for i in range(self._num_zones):
            key = self._zone_clearance_keys[i]
            if not key:
                continue
            boost_val = float(scales.get(key, 1.0))
            if boost_val <= 1.0:
                continue
            # v56: margins overridable via scales["{key}_pre_margin"] / ["{key}_post_margin"]
            # If not in scales, falls back to TerrainZone config value
            margin = float(scales.get(f"{key}_pre_margin", self._zone_pre_margins[i]))
            post_margin = float(scales.get(f"{key}_post_margin", self._zone_post_margins[i]))
            y_min = float(self._zone_y_min[i])
            y_max = float(self._zone_y_max[i])

            # Inside zone: full boost
            in_zone = (probe_y >= y_min) & (probe_y < y_max)
            result = np.where(in_zone, np.maximum(result, boost_val), result)

            # Pre-zone transition: partial boost
            if margin > 0:
                pre_y_min = y_min - margin
                in_pre = (probe_y >= pre_y_min) & (probe_y < y_min)
                pre_boost = boost_val * pre_zone_ratio
                result = np.where(in_pre, np.maximum(result, pre_boost), result)
                
            # Post-zone transition: full boost (to cover back legs)
            if post_margin > 0:
                post_y_max = y_max + post_margin
                in_post = (probe_y >= y_max) & (probe_y < post_y_max)
                result = np.where(in_post, np.maximum(result, boost_val), result)

        return result

    def compute_swing_scale(self, probe_y: np.ndarray, scales: dict) -> np.ndarray:
        """Compute swing contact penalty scale based on terrain zones.

        For each zone with a swing_scale_key, robots inside get the scale value
        from scales[key] (typically < 1.0 to reduce penalty on rough terrain).
        Others get 1.0.

        Multiple zones stack via min (lightest penalty wins).

        Args:
            probe_y: (n,) robot Y positions.
            scales: reward_scales dict.

        Returns:
            (n,) swing penalty multiplier array.
        """
        n = probe_y.shape[0]
        result = np.ones(n, dtype=np.float32)
        if self._num_zones == 0:
            return result

        for i in range(self._num_zones):
            key = self._zone_swing_keys[i]
            if not key:
                continue
            swing_val = float(scales.get(key, 1.0))
            y_min = float(self._zone_y_min[i])
            y_max = float(self._zone_y_max[i])
            
            # v56: swing scale applies to main zone ONLY (matches old T10 on_bump behavior)
            # Old code: on_bump = (y > -1.8) & (y < 1.8) — no pre/post zone extension
            in_zone = (probe_y >= y_min) & (probe_y < y_max)
            result = np.where(in_zone, np.minimum(result, swing_val), result)

        return result
