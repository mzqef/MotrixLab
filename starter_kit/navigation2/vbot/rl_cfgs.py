# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
# ==============================================================================
"""
Nav2 专用 RL 配置文件 —— 独立于 motrix_rl/cfgs.py
所有 navigation2 的 PPO 超参数在此定义，不影响 navigation1 训练。
"""

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.skrl.cfg import PPOCfg


class nav2:
    """Navigation2 专用 RL 配置"""

    @rlcfg("vbot_navigation_section001")
    @dataclass
    class VBotSection001PPOConfig(PPOCfg):
        """VBot Section001 导航 PPO 配置（竞赛地图1 - 圆形平台）"""
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 100_000_000
        check_point_interval: int = 1000

        learning_rate: float = 3e-4
        rollouts: int = 24
        learning_epochs: int = 8
        mini_batches: int = 32
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.005

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        share_policy_value_features: bool = False
        # (256,128,64) 经验证对VBot收敛稳定
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("vbot_navigation_section011")
    @dataclass
    class VBotSection011PPOConfig(PPOCfg):
        """VBot Section011 导航 PPO 配置（Nav2 section01 平地+坡道+高台）"""
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 100_000_000
        check_point_interval: int = 1000

        learning_rate: float = 3e-4
        rollouts: int = 24
        learning_epochs: int = 8
        mini_batches: int = 32
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.005

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        share_policy_value_features: bool = False
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("vbot_navigation_section012")
    @dataclass
    class VBotSection012PPOConfig(PPOCfg):
        """VBot Section012 导航 PPO 配置（楼梯/桥梁/障碍物，60分赛段）"""
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 200_000_000
        check_point_interval: int = 1000

        learning_rate: float = 2e-4
        rollouts: int = 32
        learning_epochs: int = 8
        mini_batches: int = 32
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.008

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        share_policy_value_features: bool = False
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("vbot_navigation_section013")
    @dataclass
    class VBotSection013PPOConfig(PPOCfg):
        """VBot Section013 导航 PPO 配置（高度场/滚球，25分赛段）"""
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 150_000_000
        check_point_interval: int = 1000

        learning_rate: float = 2.5e-4
        rollouts: int = 28
        learning_epochs: int = 8
        mini_batches: int = 32
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.006

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        share_policy_value_features: bool = False
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("vbot_navigation_long_course")
    @dataclass
    class VBotLongCoursePPOConfig(PPOCfg):
        """VBot全程导航 PPO 配置（三段地形合并，90秒，竞赛提交用）"""
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 4
        max_env_steps: int = 300_000_000
        check_point_interval: int = 1000

        learning_rate: float = 2e-4
        rollouts: int = 48
        learning_epochs: int = 8
        mini_batches: int = 32
        discount_factor: float = 0.995
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.01

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        share_policy_value_features: bool = False
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
