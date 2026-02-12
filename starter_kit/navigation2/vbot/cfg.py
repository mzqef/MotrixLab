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
class ControlConfig:
    # stiffness[N*m/rad] 使用XML中kp参数，仅作记录
    # damping[N*m*s/rad] 使用XML中kv参数，仅作记录
    action_scale = 0.25  # 平地navigation使用0.25
    # torque_limit[N*m] 使用XML forcerange参数

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

@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # ===== 导航任务核心奖励 =====
            "position_tracking": 2.0,      # 位置误差奖励（提高10倍）
            "fine_position_tracking": 2.0,  # 精细位置奖励（提高10倍）
            "heading_tracking": 1.0,        # 朝向跟踪奖励（新增）
            "forward_velocity": 0.5,        # 前进速度奖励（鼓励朝目标移动）
            
            # ===== Locomotion稳定性奖励（保持但降低权重） =====
            "orientation": -0.05,           # 姿态稳定（降低权重）
            "lin_vel_z": -0.5,              # 垂直速度惩罚
            "ang_vel_xy": -0.05,            # XY轴角速度惩罚
            "torques": -1e-5,               # 扭矩惩罚
            "dof_vel": -5e-5,               # 关节速度惩罚
            "dof_acc": -2.5e-7,             # 关节加速度惩罚
            "action_rate": -0.01,           # 动作变化率惩罚
            
            # ===== 终止惩罚 =====
            "termination": -200.0,          # 终止惩罚
        }
    )

@registry.envcfg("vbot_navigation_flat")
@dataclass
class VBotEnvCfg(EnvCfg):
    model_file: str = model_file
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 10
    max_episode_steps: int = 1000
    sim_dt: float = 0.01    # 仿真步长 10ms = 100Hz
    ctrl_dt: float = 0.01
    reset_yaw_scale: float = 0.1
    max_dof_vel: float = 100.0  # 最大关节速度阈值，训练初期给予更大容忍度

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)


@registry.envcfg("vbot_navigation_stairs")
@dataclass
class VBotStairsEnvCfg(VBotEnvCfg):
    """VBot在楼梯地形上的导航配置，继承flat配置"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs.xml"
    max_episode_seconds: float = 20.0  # 增加到20秒，给更多时间学习转向
    max_episode_steps: int = 2000
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25  # 楼梯navigation使用0.2，足够转向但比平地更谨慎
    
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("VBotStairsMultiTarget-v0")
@dataclass
class VBotStairsMultiTargetEnvCfg(VBotStairsEnvCfg):
    """VBot楼梯多目标导航配置，继承单目标配置"""
    max_episode_seconds: float = 60.0  # 多目标需要更长时间
    max_episode_steps: int = 6000


@registry.envcfg("vbot_navigation_stairs_obstacles")
@dataclass
class VBotStairsObstaclesEnvCfg(VBotStairsEnvCfg):
    """VBot楼梯地形带障碍球的导航配置"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs_obstacles.xml"
    max_episode_seconds: float = 20.0
    max_episode_steps: int = 2000

@registry.envcfg("vbot_navigation_section01")
@dataclass
class VBotSection01EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section01.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, -2.4, 0.5]  # 中心位置
        
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # X±0.5m, Y±0.5m随机
        
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
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section02")
@dataclass
class VBotSection02EnvCfg(VBotStairsEnvCfg):
    """VBot Section02单独训练配置 - 中间楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section02.xml"
    max_episode_seconds: float = 60.0  # Section02较复杂，需要更多时间
    max_episode_steps: int = 6000
    
    @dataclass
    class InitState:
        # 起始位置：section02的起始位置（继承自locomotion）
        # pos = [-2.5, 8.5, 1.8]
        # pos = [-2.5, 8.5, 1.8]
        pos = [-2.5, 12.0, 1.8]  # Y坐标对应section02的起点，高度1.8m
        # pos = [-2.5, 15.0, 3.3]  # Y坐标对应section02的起点，高度1.8m
        # pos = [-2.5, 21.0, 3.3]  # Y坐标对应section02的起点，高度1.8m
        # pos = [-2.5, 24.6, 1.8]  # Y坐标对应section02的起点，高度1.8m
        # pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        pos_randomization_range = [-0., -0., 0., 0.]  # 小范围随机±0.5m
        
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
        # 目标范围：覆盖section02区域（10-20米）
        pose_command_range = [-3.0, 16.0, 3.14, -3.0, 26.0, 3.14]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section03")
@dataclass
class VBotSection03EnvCfg(VBotStairsEnvCfg):
    """VBot Section03单独训练配置 - 终点楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section03.xml"
    max_episode_seconds: float = 50.0  # 拉长一倍：从25秒增加到50秒
    max_episode_steps: int = 5000  # 拉长一倍：从2500步增加到5000步
    
    @dataclass
    class InitState:
        # 起始位置：section03的起始位置（继承自locomotion）
        pos = [0.0, 26.0, 1.8]  # Y坐标对应section03的起点，高度1.8m
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        
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
        # 目标范围：覆盖section03区域（20-32米）
        pose_command_range = [-3.0, 20.0, -3.14, 3.0, 32.0, 3.14]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_long_course")
@dataclass
class VBotLongCourseEnvCfg(VBotStairsEnvCfg):
    """VBot三段地形完整导航配置（比赛任务）- 使用全程合并地图"""
    # 使用scene_world_full.xml：三段地形碰撞体+视觉体合并
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_world_full.xml"
    max_episode_seconds: float = 90.0  # 全程90秒
    max_episode_steps: int = 9000  # 对应90秒 @ 100Hz
    
    @dataclass
    class InitState:
        # 起始位置：section01起始（高台中心）
        pos = [0.0, -2.4, 0.5]  # 与section01一致
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        
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
    
    @dataclass
    class ControlConfig:
        action_scale = 0.5  # 0.5: 地形需要更大关节范围（抬脚过障碍, 高度场0.277m）
    
    
    @dataclass
    class RewardConfig:
        scales: dict = field(default_factory=lambda: {
            # ===== 导航任务核心奖励 =====
            "position_tracking": 1.5,
            "fine_position_tracking": 5.0,
            "heading_tracking": 0.8,
            "forward_velocity": 1.5,
            "distance_progress": 2.0,
            "alive_bonus": 0.5,
            "approach_scale": 8.0,
            # ===== 全程特有 =====
            "waypoint_bonus": 30.0,      # 每到达一个中间航点的奖励
            "arrival_bonus": 100.0,      # 到达最终目标的大奖
            "stop_scale": 2.0,
            "zero_ang_bonus": 6.0,
            # ===== 惩罚 =====
            "orientation": -0.05,
            "lin_vel_z": -0.3,
            "ang_vel_xy": -0.03,
            "torques": -1e-5,
            "dof_vel": -5e-5,
            "dof_acc": -2.5e-7,
            "action_rate": -0.01,
            "termination": -100.0,
        })
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)

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
    max_episode_seconds: float = 40.0  # 40秒: START到高台10.3m, 需更多时间
    max_episode_steps: int = 4000
    grace_period_steps: int = 500  # 前500步(5秒)不判终止，让agent充分学会在bumps区域站立+行走
    @dataclass
    class InitState:
        # 竞赛正确起点：START平台 (Adiban_001), center=(0, -2.5), 顶面z=0
        # 竞赛规则: "初始点位置随机分布在'START'平台区域" y∈[-3.5, -1.5]
        pos = [0.0, -2.5, 0.5]  # START平台中心 (竞赛正确位置)
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
    class ControlConfig:
        action_scale = 0.5  # 0.5: 地形需要更大关节范围（抬脚过障碍, 高度场0.277m）
    @dataclass
    class ScoringZones:
        """竞赛得分区定义 (从OBJ顶点数据提取)"""
        # 3个笑脸区(各+4分): 位于height field上，y≈0处
        smiley_centers = [[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0]]  # [x, y]
        smiley_radius = 1.2  # 检测半径（OBJ范围~2m，取1.2m宽松判定）
        smiley_points = 4.0  # 每个笑脸区竞赛得分
        # 3个红包区(各+2分): 位于坡道上，y≈4.4处
        red_packet_centers = [[-3.0, 4.4], [0.0, 4.4], [3.0, 4.4]]  # [x, y]
        red_packet_radius = 1.2  # 检测半径
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
        celebration_spin_angle = 3.14159  # 每次旋转180°
        celebration_spin_tolerance = 0.3   # 角度容差(rad) ≈ 17°
        celebration_spin_speed_limit = 0.3  # 旋转时平移速度上限
        celebration_hold_steps = 30  # 旋转完成后保持静止的步数
    @dataclass
    class RewardConfig:
        scales: dict = field(default_factory=lambda: {
            # ===== v9: LIVING FIRST — 存活>速度, bump区行走训练 =====
            # 主动运动奖励 (降低: 不鼓励鲁莽冲刺)
            "forward_velocity": 1.5,         # 朝目标方向速度 (v7:5.0→v9:1.5, 防止冲刺撞死)
            "waypoint_approach": 100.0,      # 朝当前航点靠近step-delta (v7:200→v9:100)
            "waypoint_facing": 0.15,         # 面朝当前航点(极低被动信号)
            # 存活奖励 (LIVING FIRST原则: 活着>>冲刺)
            "position_tracking": 0.05,       # 微弱梯度信号
            "alive_bonus": 0.5,              # 0.5×4000=2000 (存活4000步=2000奖励, 远超冲刺)
            # 一次性大奖
            "waypoint_bonus": 100.0,         # 到达每个航点(3×100=300)
            "smiley_bonus": 40.0,            # 通过笑脸区(3×40=120)
            "red_packet_bonus": 20.0,        # 通过红包区(3×20=60)
            "celebration_bonus": 100.0,      # 庆祝旋转完成
            # Zone吸引力
            "zone_approach": 0.0,            # 先禁用: 学会直线导航后再开启
            # 地形适应
            "height_progress": 12.0,         # 爬坡z-高度进步
            "traversal_bonus": 30.0,         # 地形里程碑(×2=60)
            # 抬脚奖励 (过bumps/坡道必须抬脚)
            "foot_clearance": 0.02,          # 摆动相抬脚高度奖励 (微弱信号, 防止主导)
            # 庆祝旋转引导
            "spin_progress": 4.0,
            "spin_hold": 6.0,
            # ===== 禁用旧信号 (代码中未使用 / 反作用) =====
            "fine_position_tracking": 0.0,
            "heading_tracking": 0.0,
            "distance_progress": 0.0,
            "approach_scale": 0.0,
            "arrival_bonus": 0.0,
            "stop_scale": 0.0,
            "zero_ang_bonus": 0.0,
            "near_target_speed": 0.0,        # 禁用: 阻碍机器人前进
            "departure_penalty": 0.0,
            # ===== 摆动相接触惩罚 =====
            "swing_contact_penalty": -0.05,  # 降低: 抬脚幅度大时轻微接触正常
            # ===== 稳定性惩罚 (坡道宽松 + 大action_scale适应) =====
            "orientation": -0.015,           # 更宽松: 坡道自然倾斜
            "lin_vel_z": -0.06,              # 更宽松: 爬坡需要垂直速度
            "ang_vel_xy": -0.01,
            "torques": -5e-6,               # 降低: 更大力矩正常
            "dof_vel": -3e-5,               # 降低: 更大关节速度正常
            "dof_acc": -1.5e-7,
            "action_rate": -0.005,           # 降低: action_scale=0.5导致更大动作差
            "termination": -100.0,           # 基础终止惩罚 (加重: 死亡代价高, OOB/摔倒还会额外扣除累积奖金)
        })
    scoring_zones: ScoringZones = field(default_factory=ScoringZones)
    waypoint_nav: WaypointNav = field(default_factory=WaypointNav)
    course_bounds: CourseBounds = field(default_factory=CourseBounds)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)

@registry.envcfg("vbot_navigation_section012")
@dataclass
class VBotSection012EnvCfg(VBotStairsEnvCfg):
    """VBot Section02（楼梯/桥梁/障碍物）导航配置

    XML地形信息 (0126_C_section02.xml):
    - 高度场：中心(0, 10.33, 1.294)
    - 左侧楼梯 (x=-3)：10级，ΔZ≈0.15/级，y=12.43→14.23，z=1.369→2.794
    - 右侧楼梯 (x=2)：10级，ΔZ≈0.10/级，z=1.319→2.294
    - 拱桥 (x≈-3)：y=15.31→20.33，z≈2.51→2.86
    - 5个球形障碍 (R=0.75)：右侧通道
    - 8个锥形障碍、2个logo障碍
    - 终点平台：y≈24.33，z≈1.294
    """
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section012.xml"
    max_episode_seconds: float = 60.0  # Section02复杂地形，需要更多时间
    max_episode_steps: int = 6000
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
        # 固定目标：section02出口（终点平台中心）
        # 起始(0, 9.5) + 偏移(0, 14.5) → 目标(0, 24.0)（≈终点平台y≈24.33，z≈1.294）
        pose_command_range = [0.0, 14.5, 0.0, 0.0, 14.5, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    @dataclass
    class RewardConfig:
        scales: dict = field(default_factory=lambda: {
            # ===== 导航任务核心奖励 =====
            "position_tracking": 1.5,
            "fine_position_tracking": 5.0,
            "heading_tracking": 0.8,
            "forward_velocity": 1.5,
            "distance_progress": 2.0,
            "alive_bonus": 0.3,
            "approach_scale": 8.0,
            # ===== 到达奖励 =====
            "arrival_bonus": 80.0,    # Section02值60分，大奖鼓励
            "stop_scale": 1.5,
            "zero_ang_bonus": 6.0,
            # ===== 惩罚 =====
            "orientation": -0.05,
            "lin_vel_z": -0.3,
            "ang_vel_xy": -0.03,
            "torques": -1e-5,
            "dof_vel": -5e-5,
            "dof_acc": -2.5e-7,
            "action_rate": -0.01,
            "termination": -200.0,
        })
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)

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
    max_episode_seconds: float = 50.0  # Section03需要较多时间
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
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    @dataclass
    class RewardConfig:
        scales: dict = field(default_factory=lambda: {
            # ===== 导航任务核心奖励 =====
            "position_tracking": 1.5,
            "fine_position_tracking": 5.0,
            "heading_tracking": 0.8,
            "forward_velocity": 1.5,
            "distance_progress": 2.0,
            "alive_bonus": 0.3,
            "approach_scale": 8.0,
            # ===== 到达奖励 =====
            "arrival_bonus": 60.0,
            "stop_scale": 1.5,
            "zero_ang_bonus": 6.0,
            # ===== 惩罚 =====
            "orientation": -0.05,
            "lin_vel_z": -0.3,
            "ang_vel_xy": -0.03,
            "torques": -1e-5,
            "dof_vel": -5e-5,
            "dof_acc": -2.5e-7,
            "action_rate": -0.01,
            "termination": -200.0,
        })
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)

