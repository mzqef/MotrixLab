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
    - 高度场：中心(0, 0, 0)，范围x=±5m, y=±1.5m，高度0~0.277m
    - 地面平台：z=0
    - 15°坡道 (Adiban_003)：中心(0, 4.48, 0.41)
    - 高台 (Adiban_004)：中心(0, 7.83, 1.044)，尺寸(5×2.5×0.25)m，顶面z=1.294
    - 边界墙顶部 z≈2.45
    """
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section011.xml"
    max_episode_seconds: float = 40.0
    max_episode_steps: int = 4000
    @dataclass
    class InitState:
        # 起始位置：section01起点，地面z=0，机器人高度0.5m
        pos = [0.0, -2.4, 0.5]
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
        # 固定目标：高台顶部中心
        # 起始Y=-2.4 + 偏移10.2 → 目标Y=7.8（≈高台中心y=7.83，顶面z=1.294）
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    @dataclass
    class RewardConfig:
        scales: dict = field(default_factory=lambda: {
            # ===== 导航任务核心奖励 =====
            "position_tracking": 2.0,
            "fine_position_tracking": 2.0,
            "heading_tracking": 1.0,
            "forward_velocity": 1.5,        # 增强前进激励
            "distance_progress": 2.0,
            "alive_bonus": 1.0,              # 强存活激励
            "approach_scale": 8.0,
            # ===== 到达奖励 =====
            "arrival_bonus": 50.0,
            "stop_scale": 2.0,
            "zero_ang_bonus": 6.0,
            # ===== 惩罚 =====
            "orientation": -0.05,
            "lin_vel_z": -0.5,
            "ang_vel_xy": -0.05,
            "torques": -1e-5,
            "dof_vel": -5e-5,
            "dof_acc": -2.5e-7,
            "action_rate": -0.01,
            "termination": -50.0,            # 降低终止惩罚，避免梯度崩溃
        })
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

