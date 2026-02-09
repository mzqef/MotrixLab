# Navigation1 专用 RL 配置文件
"""
Nav1 专用 RL 配置 —— 竞赛 Stage 1 (圆形平台导航)
"""

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.skrl.cfg import PPOCfg


class nav1:
    """Navigation1 专用 RL 配置"""

    @rlcfg("vbot_navigation_section001")
    @dataclass
    class VBotSection001PPOConfig(PPOCfg):
        """VBot Section001 导航 PPO 配置（竞赛地图1 - 圆形平台 外围→中心）"""
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 100_000_000
        check_point_interval: int = 1000

        learning_rate: float = 5e-4  # Curriculum: 较高初始LR
        lr_scheduler_type: str | None = "linear"  # Exp6: 线性退火替代KL自适应——KL scheduler全部失败
        learning_rate_scheduler_kl_threshold: float = 0.012  # 仅 lr_scheduler_type="kl_adaptive" 时生效
        rollouts: int = 32  # Phase5: 48→32, HP-opt sweet spot; 更快收敛
        learning_epochs: int = 5  # Phase5: 8→5, HP-opt best=4; 减少stale rollout过拟合
        mini_batches: int = 16  # Phase5: 32→16, HP-opt best=16; 更大effective batch
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.01  # Phase5: 0.008→0.01, 更多探索逃离局部最优

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        share_policy_value_features: bool = False
        # (256,128,64) 经验证对VBot收敛稳定
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
