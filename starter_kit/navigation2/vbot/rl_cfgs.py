# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
# ==============================================================================
"""
Navigation2 专用 RL 配置文件 —— 独立于 motrix_rl/cfgs.py
所有 navigation2 的 PPO 超参数在此定义，不影响 navigation1 训练。
"""

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.skrl.cfg import PPOCfg


class navigation2:
    """Navigation2 专用 RL 配置"""

    # NOTE: section001 config is defined in navigation1/vbot/rl_cfgs.py
    # Do NOT duplicate it here — duplicate @rlcfg registration overrides nav1's
    # correct config (value_net 512,256,128) with wrong sizes (256,128,64)

    @rlcfg("vbot_navigation_section011")
    @dataclass
    class VBotSection011PPOConfig(PPOCfg):
        """VBot Section011 导航 PPO 配置（Navigation2 section01 平地+坡道+高台）
        
        Stage 15: Even Higher GAE Lambda (λ=0.99)
        - Stage 14: γ=0.999, λ=0.98 → wp_idx=1.956 (new best)
        - λ=0.98→0.99: GAE horizon extends to ~460 steps (1% mark)
        - γλ = 0.999×0.99 = 0.989, 50% horizon at 63 steps
        - Risk: near Monte-Carlo returns, high variance
        - Warm-start: Stage 14 best agent_3500.pt (wp_idx=1.956)
        """
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 15_000_000
        check_point_interval: int = 500

        learning_rate: float = 5e-5
        lr_scheduler_type: str | None = None
        rollouts: int = 24
        learning_epochs: int = 8
        mini_batches: int = 32
        discount_factor: float = 0.999  # Stage 13 proven
        lambda_param: float = 0.99      # Stage 15: push GAE further (was 0.98)
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.01

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        share_policy_value_features: bool = False
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)   # v10 proven
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)   # v10 actual (asymmetric value net)

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
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

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
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

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
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
