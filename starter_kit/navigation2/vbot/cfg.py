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
    """Y-gated terrain zone → action_scale mapping.

    Universal terrain zones covering navigation2 full course (y ≈ -3.5 to 34.33).
    Each section only encounters zones within its Y range.
    """
    y_min: float
    y_max: float
    action_scale: float
    label: str = ""  # 仅用于调试/日志

# 全程地形区域表 (从XML碰撞几何体提取)
# 每个section仅会命中其Y范围内的zone；未命中的zone不影响
DEFAULT_TERRAIN_ZONES: List[TerrainZone] = [
    # === Section 011 (y ≈ -3.5 → 8.83) ===
    TerrainZone(y_min=-1.5,  y_max=1.5,   action_scale=0.40, label="s011_bump"),      # 高度场凹凸区
    TerrainZone(y_min=2.0,   y_max=6.9,   action_scale=0.40, label="s011_slope"),      # 15°坡道
    # === Section 012 (y ≈ 8.83 → 25.33) ===
    TerrainZone(y_min=12.33, y_max=14.33, action_scale=0.50, label="s012_stairs_up"),   # 楼梯上行
    TerrainZone(y_min=14.33, y_max=21.33, action_scale=0.20, label="s012_bridge_valley"),  # 桥+河谷+平台
    TerrainZone(y_min=21.33, y_max=23.33, action_scale=0.20, label="s012_stairs_down"), # 楼梯下行
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
    "height_progress": 26.965,                     # T14: ~same (v47=28.30)
    "height_approach": 5.0,                        # unchanged (not in search)
    "height_oscillation": -2.0,                    # unchanged (not in search)
    # ===== 跳跃 & 庆祝 =====
    "jump_reward": 10.093,                         # T14: ~same
    "per_jump_bonus": 59.641,                      # T14: 2.4× (v47=25.0)
    "celebration_bonus": 141.242,                  # T14: 1.77× (v47=80.0)
    # ===== 稳定性惩罚 =====
    "orientation": -0.026,                         # T14: ~same (v47=-0.027)
    "lin_vel_z": -0.027,                           # T14: 7.2× lighter (v47=-0.195) ← KEY
    "ang_vel_xy": -0.038,                          # T14: lighter (v47=-0.045)
    "torques": -5e-6,                              # unchanged (not in search)
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
    "alive_decay_horizon": 2383.0,                 # T14: 1.59× longer (v47=1500)
    "slope_orientation": 0.0,                      # unchanged (disabled)
    # ===== v49: 拖脚惩罚 + 停滞惩罚 =====
    "drag_foot_penalty": -0.02,                    # v49: 支撑相低速腿惩罚 (每条拖地腿, bump区×2)
    "stagnation_penalty": -0.5,                    # v49: 停滞渐进惩罚 (从50%窗口开始线性增长)
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
    reset_yaw_scale: float = 0.1
    max_dof_vel: float = 100.0

    grace_period_steps: int = 100  # 前100步(1秒) 仅保护base_contact和中等倾斜; 严重倾斜/OOB/NaN始终终止
    stagnation_window_steps: int = 1000   # 10秒窗口: 若10秒内未移动足够距离则截断
    stagnation_min_distance: float = 0.5  # 10秒内至少走0.5m才算"在动"
    stagnation_grace_steps: int = 500     # 前5秒不检测停滞(给机器人起步时间)

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)


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
        pos = [0.0, -2.5, 0.35]  # START平台中心, z=0.35
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
        pos = [0.0, -2.5, 0.35]  # START平台中心, z=0.35
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
        """多航点导航配置: 3中心航点 + zone吸引力 + 庆祝

        3中心航点 (验证过: 能可靠到达高台):
          WP0: 中心笑脸 (0, 0)    — 前进方向
          WP1: 中心红包 (0, 4.4)  — 继续前进(上坡)
          WP2: 高台     (0, 7.83) — 到达 + 庆祝旋转

        侧面zone通过zone_approach奖励吸引收集(代码中实现):
          机器人在2.5m内感受到未收集zone的引力
          配合passive zone detection (1.2m), 引导机器人顺路收集
        """
        # 航点坐标 [x, y] — 前进路线
        waypoints = [[0.0, 0.0], [0.0, 4.4], [0.0, 7.83]]
        # 航点到达半径
        waypoint_radius = 1.0  # 笑脸/红包zone半径较大，走到附近即可
        final_radius = 0.5     # 高台目标更精确
        # 庆祝旋转参数
        celebration_jump_threshold = 1.55  # v16b: 实测站立z≈1.52, 小跳+0.03m即可
        # v27: 多次跳跃庆祝
        required_jumps = 3                # 需要跳3次才算完成庆祝
        celebration_landing_z = 1.50      # 落地判定: z < 1.50 = 已着地, 可以再跳


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
        # 不再使用固定offset目标; 目标由状态机动态指定
        # 保留结构兼容性: 起始(0,9.5) + 偏移(0,14.5) → 仅在 Phase 6 fallback 时使用
        pose_command_range = [0.0, 14.5, 0.0, 0.0, 14.5, 0.0]
    @dataclass
    class ScoringZones:
        """竞赛得分区定义 (从XML碰撞/可视体提取坐标)"""
        # 桥上拜年红包 (+10分: 过桥途径收集)
        # 红包在桥面中央, 通过过桥行为自然收集
        bridge_hongbao_center = [-3.0, 17.83]  # 桥面中心位置
        bridge_hongbao_radius = 2.0  # 宽松判定 (桥面长5.2m, 中间2m即算过)
        bridge_hongbao_min_z = 2.3   # 必须在桥面上 (桥面z≈2.51~2.71)
        bridge_hongbao_points = 10.0
        # 桥底下拜年红包 (2个, 各+5分)
        # 位于桥面下方, 河谷底部附近, 过桥后才激活
        under_bridge_centers = [[-3.0, 16.0], [-3.0, 19.5]]  # 桥入口下方 + 桥出口下方
        under_bridge_radius = 1.5  # 宽松判定
        under_bridge_max_z = 2.2   # 必须在桥面下 (桥面z≈2.5, 桥下z<2.2)
        under_bridge_points = 5.0
        # 河床石头贺礼红包 (5个, 各+3分, 位于右侧通道球形障碍顶部)
        stone_hongbao_centers = [
            [3.5, 15.84],   # C_Bpo_sphere_001 top
            [0.36, 15.84],  # C_Bpo_sphere_002 top
            [2.0, 17.83],   # C_Bpo_sphere_003 top (center, lower)
            [3.5, 19.72],   # C_Bpo_sphere_004 top
            [0.36, 19.72],  # C_Bpo_sphere_005 top
        ]
        stone_hongbao_radius = 1.0  # 球体R=0.75, 判定半径略大于球顶
        stone_hongbao_points = 3.0
        # 庆祝区 (终点平台, +5分)
        celebration_center = [0.0, 24.33]
        celebration_radius = 1.5
        celebration_min_z = 1.0
        celebration_points = 5.0
    @dataclass
    class BridgeNav:
        """桥优先导航虚拟航点定义

        桥面路径: x≈-3.0, y=15.31→20.33, z≈2.51→2.71
        虚拟航点强制机器人走桥面路线, 防止走桥下近路或绕行右侧
        """
        # 虚拟导航点序列 [x, y] — 桥优先固定主线
        # Phase 0: 波浪地形→左楼梯底
        wave_to_stair = [-3.0, 12.3]    # 左楼梯底部入口 (y=12.43处)
        # Phase 1: 左楼梯顶→桥入口
        stair_top = [-3.0, 14.5]        # 左楼梯顶部 (y=14.23, z≈2.79)
        stair_top_min_z = 2.3           # 必须爬上楼梯 (z>2.3 = 到达顶部)
        # Phase 2: 过桥虚拟导航点 (3个)
        bridge_entry = [-3.0, 15.8]     # 桥面入口
        bridge_mid = [-3.0, 17.83]      # 桥面中心 (最低点z≈2.51)
        bridge_exit = [-3.0, 20.0]      # 桥面出口
        bridge_min_z = 2.3              # 必须在桥面上
        # Phase 4: 下楼梯
        stair_down_bottom = [-3.0, 23.2]  # 左楼梯底部 (下到地面)
        # Phase 6: 终点平台
        exit_platform = [0.0, 24.33]     # 丙午大吉平台中心
        # 航点到达半径
        waypoint_radius = 1.2  # 虚拟导航点半径 (宽松)
        bridge_wp_radius = 1.5  # 桥面导航点半径 (桥宽2.64m, 放宽)
        final_radius = 0.8     # 终点平台精确到达
        # 庆祝: 跳跃
        celebration_jump_threshold = 1.55  # 与section011一致
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

    scoring_zones: ScoringZones = field(default_factory=ScoringZones)
    bridge_nav: BridgeNav = field(default_factory=BridgeNav)
    course_bounds: CourseBounds = field(default_factory=CourseBounds)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)


@registry.envcfg("vbot_navigation_section013")
@dataclass
class VBotSection013EnvCfg(VBotStairsEnvCfg):
    """VBot Section03（滚球/坡道/最终平台）导航配置

    XML地形信息 (0126_C_section03.xml):
    - 高度场：中心(0, 29.33, 1.343)
    - 入口平台：z=1.294
    - 0.75m高隔离墙 (Cdiban_006)：y=27.58
    - 21.8°坡道 (Cdiban_002)
    - 3个金球障碍 (R=0.75)：y=31.23，x=-3/0/3，球心z=0.844
    - 最终平台 (Cdiban_004)：中心(0, 32.33, 0.994)，顶面z=1.494
    - 终点墙：y=34.33
    """
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section013.xml"
    max_episode_seconds: float = 50.0  # Section03: 50s/5000steps (生产配置, Run5 proven)
    max_episode_steps: int = 5000
    @dataclass
    class InitState:
        # 起始位置：section03入口，地面z≈1.294，机器人高度0.5m以上
        pos = [0.0, 26.0, 1.8]
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # ±0.5m随机（竞赛会随机起始点）

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
        # 起始(0, 26.0) + 偏移(0, 6.33) → 目标(0, 32.33)（≈最终平台中心，顶面z=1.494）
        pose_command_range = [0.0, 6.33, 0.0, 0.0, 6.33, 0.0]


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
        else:
            self._zone_y_min = np.empty(0, dtype=np.float32)
            self._zone_y_max = np.empty(0, dtype=np.float32)
            self._zone_scale = np.empty(0, dtype=np.float32)
            self._num_zones = 0

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
