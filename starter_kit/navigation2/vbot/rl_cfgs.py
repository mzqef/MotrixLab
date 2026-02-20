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
        """
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 100_000_000    # v48-T14: full deployment
        check_point_interval: int = 500

        learning_rate: float = 4.513e-4     # v48-T14: 4.5× higher (v47=1e-4), kl_adaptive self-regulates
        lr_scheduler_type: str | None = "kl_adaptive"  # KL-adaptive: self-regulates on KL excess
        rollouts: int = 24
        learning_epochs: int = 6            # v23b-T7: fewer epochs (was 10)
        mini_batches: int = 16              # v23b-T7: same
        discount_factor: float = 0.999      # Stage 13 proven (keep — curriculum tested)
        lambda_param: float = 0.99          # Stage 15 proven (keep — curriculum tested)
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.00775  # v48-T14: 1.8× higher (v47=0.00432)

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        share_policy_value_features: bool = False
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)   # match value net size
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)   # v21 proven: larger value net

    @rlcfg("vbot_navigation_section012")
    @dataclass
    class VBotSection012PPOConfig(PPOCfg):
        """VBot Section012 导航 PPO 配置（楼梯/桥梁/障碍物，60分赛段）
        与section011对齐以支持69维obs warm-start:
        """
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 50_000_000
        check_point_interval: int = 500

        learning_rate: float = 0.0003568963889028796          # warm-start: same as v29 proven
        lr_scheduler_type: str | None = None # constant LR — no decay for warm-start
        rollouts: int = 24                   # section011 universally converged
        learning_epochs: int = 6             # section011 universally converged (was 8)
        mini_batches: int = 16               # section011 universally converged (was 32)
        discount_factor: float = 0.999       # Stage 13 proven: full-episode planning
        lambda_param: float = 0.99           # Stage 15 proven: high GAE horizon
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.004318625492723052     # warm-start on new terrain needs exploration

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        share_policy_value_features: bool = False
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)   # section011 proven
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)   # section011 proven

    @rlcfg("vbot_navigation_section013")
    @dataclass
    class VBotSection013PPOConfig(PPOCfg):
        """VBot Section013 导航 PPO 配置（高度场/滚球，25分赛段）
        """
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 30_000_000       # Run8: fresh 30M from section001
        check_point_interval: int = 500

        learning_rate: float = 0.0003568963889028796            # Run5/10 proven: warm-start LR
        lr_scheduler_type: str | None = None   # constant LR (Run5/10 proven)
        rollouts: int = 24                     # AutoML收敛值
        learning_epochs: int = 6               # AutoML收敛值 (was 8)
        mini_batches: int = 16                 # AutoML收敛值 (was 32)
        discount_factor: float = 0.999         # Stage 13证明: 长视野 (was 0.99)
        lambda_param: float = 0.99             # Stage 15证明: 高GAE (was 0.95)
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.004318625492723052       # warm-start需要更多探索 (was 0.006)

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

        learning_rate: float = 0.0003568963889028796
        rollouts: int = 48
        learning_epochs: int = 8
        mini_batches: int = 32
        discount_factor: float = 0.995
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.004318625492723052

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        share_policy_value_features: bool = False
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
