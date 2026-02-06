#!/usr/bin/env python3
"""
AutoML Training Pipeline for VBot Quadruped Navigation.

Automatically orchestrates:
- Curriculum learning (stage progression)
- Hyperparameter optimization (Bayesian/random/grid)
- Reward engineering (evolutionary/grid)
- Training execution and evaluation
- Feedback loop for continuous improvement

Usage:
    uv run starter_kit_schedule/scripts/automl.py --mode full --budget-hours 48
    uv run starter_kit_schedule/scripts/automl.py --resume
    uv run starter_kit_schedule/scripts/automl.py --status
"""

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "starter_kit_log"
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HPConfig:
    """Hyperparameter configuration."""
    learning_rate: float = 3e-4
    entropy_loss_scale: float = 1e-3
    policy_hidden_layer_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    value_hidden_layer_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    rollouts: int = 24
    learning_epochs: int = 5
    mini_batches: int = 32
    discount_factor: float = 0.99
    lambda_param: float = 0.95
    ratio_clip: float = 0.2
    grad_norm_clip: float = 1.0


@dataclass
class RewardConfig:
    """Reward weights configuration — matches starter_kit/navigation*/vbot/cfg.py scales."""
    # Navigation core rewards
    position_tracking: float = 2.0
    fine_position_tracking: float = 2.0
    heading_tracking: float = 1.0
    forward_velocity: float = 0.5
    # Navigation-specific rewards (approach/arrival/stop)
    approach_scale: float = 4.0
    arrival_bonus: float = 10.0
    stop_scale: float = 2.0
    zero_ang_bonus: float = 6.0
    # Stability penalties
    orientation: float = -0.05
    lin_vel_z: float = -0.5
    ang_vel_xy: float = -0.05
    torques: float = -1e-5
    dof_vel: float = -5e-5
    dof_acc: float = -2.5e-7
    action_rate: float = -0.01
    # Termination
    termination: float = -200.0


@dataclass
class EvalMetrics:
    """Evaluation metrics."""
    episode_reward_mean: float = 0.0
    episode_reward_std: float = 0.0
    episode_length_mean: float = 0.0
    success_rate: float = 0.0
    termination_rate: float = 0.0

    def compute_score(self) -> float:
        """Compute weighted AutoML score."""
        score = (
            0.4 * min(self.episode_reward_mean / 50.0, 1.0) +  # Normalize to [0, 1]
            0.3 * self.success_rate +
            0.2 * (1.0 - min(self.episode_length_mean / 1000.0, 1.0)) +
            0.1 * (1.0 - self.termination_rate)
        )
        return score


@dataclass
class AutoMLConfig:
    """AutoML configuration."""
    mode: str = "full"  # full | stage | hp-search | reward-search | eval
    budget_hours: float = 48.0
    target_reward: float = 35.0
    max_iterations: int = 100
    environment: str = "vbot_navigation_section001"

    # Curriculum settings
    auto_promote: bool = True
    promotion_patience: int = 3
    promotion_threshold: float = 30.0
    stages: List[str] = field(default_factory=lambda: [
        "stage1_flat",
        "stage2a_waves",
        "stage2b_stairs",
        "stage2c_obstacles",
    ])

    # HP search settings
    hp_method: str = "bayesian"  # bayesian | random | grid
    hp_trials_per_stage: int = 20
    hp_warmup_trials: int = 5
    hp_eval_steps: int = 5_000_000

    # Reward search settings
    reward_method: str = "evolutionary"  # evolutionary | grid | bayesian
    reward_population_size: int = 10
    reward_generations: int = 5
    reward_eval_steps: int = 2_000_000

    # Evaluation settings
    eval_episodes: int = 100

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_improvement: float = 0.5

    # Training settings
    full_train_steps: int = 50_000_000
    checkpoint_interval: int = 500
    num_envs: int = 2048
    seed: int = 42


@dataclass
class AutoMLState:
    """AutoML execution state (for resume)."""
    automl_id: str = ""
    status: str = "initialized"  # initialized | running | paused | completed | failed
    mode: str = "full"

    # Progress tracking
    current_phase: str = "curriculum"  # curriculum | hp_search | reward_search | training | eval
    current_stage: str = "stage1_flat"
    current_iteration: int = 0

    # Budget tracking
    start_time: str = ""
    elapsed_hours: float = 0.0
    budget_hours: float = 48.0

    # Best results per stage
    best_results: Dict[str, Dict] = field(default_factory=dict)

    # Search history
    hp_search_history: List[Dict] = field(default_factory=list)
    reward_search_history: List[Dict] = field(default_factory=list)

    # Curriculum progress
    curriculum_progress: Dict[str, str] = field(default_factory=dict)

    # Current configs
    current_hp_config: Dict = field(default_factory=dict)
    current_reward_config: Dict = field(default_factory=dict)


# =============================================================================
# Search Space Definitions
# =============================================================================

HP_SEARCH_SPACE = {
    "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
    "entropy_loss_scale": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
    "policy_hidden_layer_sizes": {
        "type": "categorical",
        "choices": [[128, 64], [256, 128, 64], [512, 256, 128], [256, 256, 256]],
    },
    "rollouts": {"type": "choice", "values": [16, 24, 32, 48]},
    "learning_epochs": {"type": "choice", "values": [4, 5, 6, 8]},
    "mini_batches": {"type": "choice", "values": [16, 32, 64]},
}

REWARD_SEARCH_SPACE = {
    "position_tracking": {"type": "uniform", "low": 0.5, "high": 5.0},
    "fine_position_tracking": {"type": "uniform", "low": 0.5, "high": 4.0},
    "heading_tracking": {"type": "uniform", "low": 0.1, "high": 2.0},
    "forward_velocity": {"type": "uniform", "low": 0.0, "high": 2.0},
    "approach_scale": {"type": "uniform", "low": 2.0, "high": 10.0},
    "arrival_bonus": {"type": "uniform", "low": 5.0, "high": 25.0},
    "stop_scale": {"type": "uniform", "low": 1.0, "high": 5.0},
    "zero_ang_bonus": {"type": "uniform", "low": 2.0, "high": 12.0},
    "orientation": {"type": "uniform", "low": -0.3, "high": -0.01},
    "lin_vel_z": {"type": "uniform", "low": -2.0, "high": -0.1},
    "action_rate": {"type": "uniform", "low": -0.05, "high": -0.001},
    "termination": {"type": "choice", "values": [-500, -300, -200, -100, -50]},
}


# =============================================================================
# Sampling Functions
# =============================================================================

def sample_from_space(space: Dict[str, Any]) -> Any:
    """Sample a value from search space definition. Returns native Python types."""
    if space["type"] == "loguniform":
        return float(np.exp(np.random.uniform(np.log(space["low"]), np.log(space["high"]))))
    elif space["type"] == "uniform":
        return float(np.random.uniform(space["low"], space["high"]))
    elif space["type"] == "choice":
        val = np.random.choice(space["values"])
        return int(val) if isinstance(val, (np.integer, int)) else float(val)
    elif space["type"] == "categorical":
        return random.choice(space["choices"])
    else:
        raise ValueError(f"Unknown space type: {space['type']}")


def sample_hp_config() -> HPConfig:
    """Sample a random HP configuration."""
    return HPConfig(
        learning_rate=sample_from_space(HP_SEARCH_SPACE["learning_rate"]),
        entropy_loss_scale=sample_from_space(HP_SEARCH_SPACE["entropy_loss_scale"]),
        policy_hidden_layer_sizes=sample_from_space(HP_SEARCH_SPACE["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=sample_from_space(HP_SEARCH_SPACE["policy_hidden_layer_sizes"]),
        rollouts=sample_from_space(HP_SEARCH_SPACE["rollouts"]),
        learning_epochs=sample_from_space(HP_SEARCH_SPACE["learning_epochs"]),
        mini_batches=sample_from_space(HP_SEARCH_SPACE["mini_batches"]),
    )


def sample_reward_config() -> RewardConfig:
    """Sample a random reward configuration from REWARD_SEARCH_SPACE."""
    kwargs = {}
    for key, space in REWARD_SEARCH_SPACE.items():
        kwargs[key] = sample_from_space(space)
    return RewardConfig(**kwargs)


# =============================================================================
# Evolutionary Reward Search
# =============================================================================

def crossover_rewards(parent1: RewardConfig, parent2: RewardConfig) -> RewardConfig:
    """Crossover two reward configs."""
    child_dict = {}
    for key in asdict(parent1).keys():
        if random.random() < 0.5:
            child_dict[key] = getattr(parent1, key)
        else:
            child_dict[key] = getattr(parent2, key)
    return RewardConfig(**child_dict)


def mutate_reward(config: RewardConfig, mutation_rate: float = 0.2) -> RewardConfig:
    """Mutate a reward config."""
    config_dict = asdict(config)
    for key, space in REWARD_SEARCH_SPACE.items():
        if random.random() < mutation_rate:
            if space["type"] in ["uniform", "loguniform"]:
                current = config_dict[key]
                # Perturb by ±20%
                perturbation = current * random.uniform(-0.2, 0.2)
                new_val = current + perturbation
                # Clip to bounds
                new_val = max(space["low"], min(space["high"], new_val))
                config_dict[key] = new_val
            elif space["type"] == "choice":
                config_dict[key] = random.choice(space["values"])
    return RewardConfig(**config_dict)


# =============================================================================
# AutoML Pipeline
# =============================================================================

class AutoMLPipeline:
    """Main AutoML orchestration class."""

    def __init__(self, config: AutoMLConfig, state: Optional[AutoMLState] = None):
        self.config = config
        self.state = state or AutoMLState(
            automl_id=f"automl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode=config.mode,
            budget_hours=config.budget_hours,
        )

        # Initialize directories
        self.log_dir = PROJECT_ROOT / "starter_kit_log" / "automl" / self.state.automl_id
        self.schedule_dir = PROJECT_ROOT / "starter_kit_schedule"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize curriculum progress
        if not self.state.curriculum_progress:
            for stage in config.stages:
                self.state.curriculum_progress[stage] = "pending"

    def save_state(self):
        """Save current state for resume."""
        state_path = self.schedule_dir / "progress" / "automl_state.yaml"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            yaml.dump(asdict(self.state), f, default_flow_style=False)
        logger.info(f"State saved to {state_path}")

    @classmethod
    def load_state(cls, config: AutoMLConfig) -> "AutoMLPipeline":
        """Load state from checkpoint."""
        state_path = PROJECT_ROOT / "starter_kit_schedule" / "progress" / "automl_state.yaml"
        if not state_path.exists():
            raise FileNotFoundError(f"No AutoML state found at {state_path}")

        with open(state_path) as f:
            state_dict = yaml.safe_load(f)

        state = AutoMLState(**state_dict)
        return cls(config, state)

    def check_budget(self) -> bool:
        """Check if budget is exhausted."""
        return self.state.elapsed_hours < self.config.budget_hours

    def update_elapsed_time(self):
        """Update elapsed time tracking."""
        if self.state.start_time:
            start = datetime.fromisoformat(self.state.start_time)
            self.state.elapsed_hours = (datetime.now() - start).total_seconds() / 3600

    def run(self):
        """Run the AutoML pipeline."""
        logger.info(f"Starting AutoML pipeline: {self.state.automl_id}")
        logger.info(f"Mode: {self.config.mode}, Budget: {self.config.budget_hours}h")

        self.state.status = "running"
        self.state.start_time = datetime.now().isoformat()
        self.save_state()

        try:
            if self.config.mode == "full":
                self._run_full_pipeline()
            elif self.config.mode == "stage":
                self._run_stage_optimization(self.config.environment)
            elif self.config.mode == "hp-search":
                self._run_hp_search_only()
            elif self.config.mode == "reward-search":
                self._run_reward_search_only()
            elif self.config.mode == "eval":
                self._run_evaluation_only()

            self.state.status = "completed"
            logger.info("AutoML pipeline completed successfully!")

        except KeyboardInterrupt:
            logger.warning("AutoML interrupted by user")
            self.state.status = "paused"
        except Exception as e:
            logger.error(f"AutoML failed: {e}")
            self.state.status = "failed"
            raise
        finally:
            self.save_state()
            self._generate_report()

    def _run_full_pipeline(self):
        """Run complete curriculum with HP + reward optimization."""
        for stage in self.config.stages:
            if not self.check_budget():
                logger.warning("Budget exhausted, stopping")
                break

            if self.state.curriculum_progress.get(stage) == "completed":
                logger.info(f"Stage {stage} already completed, skipping")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Starting stage: {stage}")
            logger.info(f"{'='*60}")

            self.state.current_stage = stage
            self.state.curriculum_progress[stage] = "in_progress"
            self.save_state()

            # Run stage optimization
            best_metrics = self._run_stage_optimization(stage)

            # Check for promotion
            if best_metrics.episode_reward_mean >= self.config.promotion_threshold:
                logger.info(f"Stage {stage} completed! Reward: {best_metrics.episode_reward_mean:.2f}")
                self.state.curriculum_progress[stage] = "completed"
            else:
                logger.warning(f"Stage {stage} did not reach promotion threshold")
                if not self.config.auto_promote:
                    break

            self.update_elapsed_time()
            self.save_state()

    def _run_stage_optimization(self, stage: str) -> EvalMetrics:
        """Optimize a single curriculum stage."""
        logger.info(f"Optimizing stage: {stage}")

        # Phase 1: HP Search
        self.state.current_phase = "hp_search"
        self.save_state()
        best_hp = self._run_hp_search(stage)

        # Phase 2: Reward Search
        self.state.current_phase = "reward_search"
        self.save_state()
        best_reward = self._run_reward_search(stage, best_hp)

        # Phase 3: Full Training
        self.state.current_phase = "training"
        self.save_state()
        metrics = self._run_full_training(stage, best_hp, best_reward)

        # Save best results
        self.state.best_results[stage] = {
            "hp_config": asdict(best_hp),
            "reward_config": asdict(best_reward),
            "metrics": asdict(metrics),
            "checkpoint": str(self.log_dir / "stages" / stage / "best_checkpoint"),
        }

        return metrics

    def _run_hp_search(self, stage: str) -> HPConfig:
        """Run hyperparameter search for a stage."""
        logger.info(f"Running HP search for {stage}: {self.config.hp_trials_per_stage} trials")

        best_hp = None
        best_score = -float("inf")

        for trial in range(self.config.hp_trials_per_stage):
            if not self.check_budget():
                break

            self.state.current_iteration = trial

            # Sample or use Bayesian acquisition
            if trial < self.config.hp_warmup_trials or self.config.hp_method == "random":
                hp_config = sample_hp_config()
            else:
                # Simple Bayesian: use best + perturbation
                hp_config = self._bayesian_hp_suggest(best_hp)

            logger.info(f"HP Trial {trial+1}/{self.config.hp_trials_per_stage}: lr={hp_config.learning_rate:.2e}")

            # Train and evaluate
            metrics = self._train_and_eval(
                stage, hp_config, RewardConfig(),
                max_steps=self.config.hp_eval_steps
            )

            score = metrics.compute_score()

            # Record history
            self.state.hp_search_history.append({
                "trial": trial,
                "config": asdict(hp_config),
                "metrics": asdict(metrics),
                "score": score,
            })

            if score > best_score:
                best_score = score
                best_hp = hp_config
                logger.info(f"New best HP! Score: {score:.4f}, Reward: {metrics.episode_reward_mean:.2f}")

            self.update_elapsed_time()
            self.save_state()

        logger.info(f"HP search complete. Best lr: {best_hp.learning_rate:.2e}")
        return best_hp or HPConfig()

    def _bayesian_hp_suggest(self, current_best: Optional[HPConfig]) -> HPConfig:
        """Simple Bayesian suggestion (perturb best config)."""
        if current_best is None:
            return sample_hp_config()

        # Perturb the best config
        config_dict = asdict(current_best)

        # Perturb learning rate in log space
        lr = config_dict["learning_rate"]
        lr = lr * np.exp(np.random.normal(0, 0.3))
        lr = max(1e-5, min(1e-3, lr))
        config_dict["learning_rate"] = lr

        # Perturb entropy
        ent = config_dict["entropy_loss_scale"]
        ent = ent * np.exp(np.random.normal(0, 0.3))
        ent = max(1e-4, min(1e-2, ent))
        config_dict["entropy_loss_scale"] = ent

        # Occasionally try different architecture
        if random.random() < 0.2:
            config_dict["policy_hidden_layer_sizes"] = sample_from_space(
                HP_SEARCH_SPACE["policy_hidden_layer_sizes"]
            )
            config_dict["value_hidden_layer_sizes"] = config_dict["policy_hidden_layer_sizes"]

        return HPConfig(**config_dict)

    def _run_reward_search(self, stage: str, hp_config: HPConfig) -> RewardConfig:
        """Run evolutionary reward search."""
        logger.info(f"Running reward search for {stage}: {self.config.reward_generations} generations")

        # Initialize population
        population = [sample_reward_config() for _ in range(self.config.reward_population_size)]

        best_reward = None
        best_score = -float("inf")

        for gen in range(self.config.reward_generations):
            if not self.check_budget():
                break

            logger.info(f"Reward Generation {gen+1}/{self.config.reward_generations}")

            # Evaluate population
            scores = []
            for i, reward_config in enumerate(population):
                metrics = self._train_and_eval(
                    stage, hp_config, reward_config,
                    max_steps=self.config.reward_eval_steps
                )
                score = metrics.compute_score()
                scores.append((score, reward_config, metrics))

                if score > best_score:
                    best_score = score
                    best_reward = reward_config

            # Record history
            self.state.reward_search_history.append({
                "generation": gen,
                "best_config": asdict(best_reward),
                "best_score": best_score,
            })

            # Selection (top 50%)
            scores.sort(reverse=True, key=lambda x: x[0])
            survivors = [s[1] for s in scores[:len(scores)//2]]

            # Create next generation
            new_population = [best_reward]  # Elite
            while len(new_population) < self.config.reward_population_size:
                parent1, parent2 = random.sample(survivors, 2)
                child = crossover_rewards(parent1, parent2)
                child = mutate_reward(child)
                new_population.append(child)

            population = new_population

            self.update_elapsed_time()
            self.save_state()

        logger.info(f"Reward search complete. Best termination: {best_reward.termination}")
        return best_reward or RewardConfig()

    def _run_full_training(
        self, stage: str, hp_config: HPConfig, reward_config: RewardConfig
    ) -> EvalMetrics:
        """Run full training with best configs."""
        logger.info(f"Running full training for {stage}")
        logger.info(f"HP: lr={hp_config.learning_rate:.2e}, entropy={hp_config.entropy_loss_scale:.2e}")
        logger.info(f"Reward: pos={reward_config.position_tracking:.2f}, term={reward_config.termination:.0f}")

        metrics = self._train_and_eval(
            stage, hp_config, reward_config,
            max_steps=self.config.full_train_steps
        )

        logger.info(f"Full training complete. Reward: {metrics.episode_reward_mean:.2f}")
        return metrics

    def _train_and_eval(
        self,
        stage: str,
        hp_config: HPConfig,
        reward_config: RewardConfig,
        max_steps: int
    ) -> EvalMetrics:
        """
        Train and evaluate a configuration via subprocess.

        Generates a config JSON, launches train_one.py as a subprocess,
        parses the run directory from stdout, evaluates TensorBoard logs,
        and writes experiment summaries for analyze.py compatibility.
        """
        import subprocess as sp

        iteration_tag = (
            f"{self.state.automl_id}_{stage}_i{self.state.current_iteration}_{int(time.time())}"
        )

        # Convert HPConfig → rl_overrides dict
        rl_overrides = {
            "learning_rate": hp_config.learning_rate,
            "entropy_loss_scale": hp_config.entropy_loss_scale,
            "policy_hidden_layer_sizes": list(hp_config.policy_hidden_layer_sizes),
            "value_hidden_layer_sizes": list(hp_config.value_hidden_layer_sizes),
            "rollouts": hp_config.rollouts,
            "learning_epochs": hp_config.learning_epochs,
            "mini_batches": hp_config.mini_batches,
            "discount_factor": hp_config.discount_factor,
            "max_env_steps": max_steps,
            "seed": self.config.seed,
        }

        # Convert RewardConfig → reward_scales dict
        reward_scales = asdict(reward_config)

        # Resolve starter_kit dir so train_one.py can import the env module
        if "navigation2" in self.config.environment or "stairs" in self.config.environment:
            starter_kit_dir = str(PROJECT_ROOT / "starter_kit" / "navigation2")
        else:
            starter_kit_dir = str(PROJECT_ROOT / "starter_kit" / "navigation1")

        config_data = {
            "run_tag": iteration_tag,
            "env_name": self.config.environment,
            "starter_kit_dir": starter_kit_dir,
            "reward_scales": reward_scales,
            "rl_overrides": rl_overrides,
        }

        # Write config JSON
        config_dir = self.log_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{iteration_tag}.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2, cls=_NumpyEncoder)

        # Determine train script path (co-located in starter_kit_schedule/scripts)
        train_script = Path(__file__).resolve().parent / "train_one.py"
        cmd = ["uv", "run", str(train_script), "--config", str(config_path)]

        logger.info(f"Launching training: {iteration_tag}")
        logger.info(f"  HP: lr={hp_config.learning_rate:.2e}, rollouts={hp_config.rollouts}")
        logger.info(f"  Reward: pos={reward_config.position_tracking:.2f}, "
                     f"approach={reward_config.approach_scale:.1f}, term={reward_config.termination:.0f}")
        logger.info(f"  Steps: {max_steps:,}")

        start = time.time()
        try:
            # Adaptive timeout: ~1 second per 3500 env steps, minimum 30 minutes
            timeout = max(1800, int(max_steps / 3500))
            result = sp.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=timeout,
            )
        except sp.TimeoutExpired:
            logger.warning(f"Training timed out after {time.time() - start:.0f}s")
            return EvalMetrics()

        elapsed = time.time() - start

        # Parse run directory from subprocess stdout
        run_dir = None
        for line in result.stdout.split("\n"):
            if line.startswith("PIPELINE_RUN_DIR="):
                run_dir = line.split("=", 1)[1].strip()

        if result.returncode != 0 or not run_dir:
            logger.error(f"Training failed (rc={result.returncode})")
            if result.stderr:
                for errline in result.stderr.strip().split("\n")[-5:]:
                    logger.error(f"  stderr: {errline}")
            return EvalMetrics()

        # Evaluate via TensorBoard logs
        from evaluate import evaluate_run  # noqa: E402  — co-located in starter_kit_schedule/scripts/

        eval_result = evaluate_run(run_dir)

        if eval_result.get("status") != "ok":
            logger.warning(f"Evaluation returned status={eval_result.get('status')}")
            return EvalMetrics()

        metrics = EvalMetrics(
            episode_reward_mean=eval_result.get("final_reward", 0.0),
            episode_reward_std=eval_result.get("stability", 0.0),
            success_rate=eval_result.get("final_reached_fraction", 0.0),
            termination_rate=0.0,
        )

        logger.info(
            f"  Result: reward={metrics.episode_reward_mean:.2f}, "
            f"reached={metrics.success_rate:.2%}, elapsed={elapsed:.0f}s"
        )

        # Write experiment summary for analyze.py compatibility
        exp_dir = LOG_DIR / "experiments" / iteration_tag
        exp_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "experiment_id": iteration_tag,
            "config_id": iteration_tag,
            "campaign_id": self.state.automl_id,
            "config": {
                "environment": self.config.environment,
                "algorithm": "PPO",
                "hyperparameters": rl_overrides,
            },
            "execution": {
                "started_at": datetime.fromtimestamp(start).isoformat() + "Z",
                "completed_at": datetime.now().isoformat() + "Z",
                "duration_hours": elapsed / 3600,
                "status": "completed",
            },
            "results": {
                "final_metrics": {
                    "episode_reward_mean": metrics.episode_reward_mean,
                    "success_rate": metrics.success_rate,
                },
                "reward_scales": reward_scales,
                "run_dir": run_dir,
            },
        }
        with open(exp_dir / "summary.yaml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        # Update master index
        index_path = LOG_DIR / "index.yaml"
        if index_path.exists():
            with open(index_path) as f:
                index = yaml.safe_load(f) or {"campaigns": [], "experiments": []}
        else:
            index = {"campaigns": [], "experiments": []}
        index["experiments"].append({
            "experiment_id": iteration_tag,
            "campaign_id": self.state.automl_id,
            "environment": self.config.environment,
            "status": "completed",
            "completed_at": datetime.now().isoformat() + "Z",
        })
        with open(index_path, "w") as f:
            yaml.dump(index, f, default_flow_style=False)

        return metrics

    def _run_hp_search_only(self):
        """Run HP search only mode."""
        stage = self.state.current_stage or self.config.stages[0]
        best_hp = self._run_hp_search(stage)
        self.state.current_hp_config = asdict(best_hp)

    def _run_reward_search_only(self):
        """Run reward search only mode."""
        stage = self.state.current_stage or self.config.stages[0]
        hp_config = HPConfig(**self.state.current_hp_config) if self.state.current_hp_config else HPConfig()
        best_reward = self._run_reward_search(stage, hp_config)
        self.state.current_reward_config = asdict(best_reward)

    def _run_evaluation_only(self):
        """Run evaluation only mode."""
        logger.info("Evaluation-only mode not yet implemented")

    def _generate_report(self):
        """Generate AutoML report."""
        report_path = self.log_dir / "report.md"

        report = f"""# AutoML Report: {self.state.automl_id}

## Summary
- **Status**: {self.state.status}
- **Mode**: {self.config.mode}
- **Budget**: {self.state.elapsed_hours:.1f} / {self.config.budget_hours} hours
- **Iterations**: {self.state.current_iteration}

## Curriculum Progress
"""
        for stage, status in self.state.curriculum_progress.items():
            emoji = "✅" if status == "completed" else "🔄" if status == "in_progress" else "⏳"
            report += f"- {emoji} {stage}: {status}\n"

        report += "\n## Best Results\n"
        for stage, results in self.state.best_results.items():
            report += f"\n### {stage}\n"
            metrics = results.get("metrics", {})
            report += f"- Reward: {metrics.get('episode_reward_mean', 'N/A'):.2f}\n"
            report += f"- Success Rate: {metrics.get('success_rate', 'N/A'):.2%}\n"
            hp = results.get("hp_config", {})
            report += f"- Learning Rate: {hp.get('learning_rate', 'N/A'):.2e}\n"

        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Report generated: {report_path}")

    def print_status(self):
        """Print current AutoML status."""
        print(f"\n{'='*60}")
        print(f"AutoML Status: {self.state.automl_id}")
        print(f"{'='*60}")
        print(f"Status: {self.state.status}")
        print(f"Mode: {self.config.mode}")
        print(f"Phase: {self.state.current_phase}")
        print(f"Stage: {self.state.current_stage}")
        print(f"Iteration: {self.state.current_iteration}")
        print(f"Budget: {self.state.elapsed_hours:.1f} / {self.config.budget_hours} hours")
        print("\nCurriculum Progress:")
        for stage, status in self.state.curriculum_progress.items():
            emoji = "✅" if status == "completed" else "🔄" if status == "in_progress" else "⏳"
            print(f"  {emoji} {stage}: {status}")

        if self.state.best_results:
            print("\nBest Results:")
            for stage, results in self.state.best_results.items():
                metrics = results.get("metrics", {})
                print(f"  {stage}: reward={metrics.get('episode_reward_mean', 'N/A'):.2f}")
        print()


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AutoML Training Pipeline")

    # Mode selection
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "stage", "hp-search", "reward-search", "eval"],
                       help="AutoML mode")

    # Configuration
    parser.add_argument("--config", type=str, help="Path to AutoML config YAML")
    parser.add_argument("--budget-hours", type=float, default=48.0, help="Time budget in hours")
    parser.add_argument("--target-reward", type=float, default=35.0, help="Target reward to achieve")
    parser.add_argument("--env", type=str, default="vbot_navigation_section001", help="Environment name")
    parser.add_argument("--stage", type=str, help="Specific stage for stage mode")

    # Control commands
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--pause", action="store_true", help="Pause AutoML")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--export-best", type=str, help="Export best config to file")
    parser.add_argument("--dashboard", action="store_true", help="Launch live dashboard")

    # Search settings
    parser.add_argument("--hp-trials", type=int, default=20, help="HP search trials per stage")
    parser.add_argument("--reward-generations", type=int, default=5, help="Reward search generations")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build config
    config = AutoMLConfig(
        mode=args.mode,
        budget_hours=args.budget_hours,
        target_reward=args.target_reward,
        environment=args.env,
        hp_trials_per_stage=args.hp_trials,
        reward_generations=args.reward_generations,
    )

    # Load from YAML if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)
            # Update config from YAML
            for key, value in yaml_config.get("automl", {}).items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Handle control commands
    if args.status:
        try:
            pipeline = AutoMLPipeline.load_state(config)
            pipeline.print_status()
        except FileNotFoundError:
            print("No AutoML run in progress")
        return

    if args.resume:
        try:
            pipeline = AutoMLPipeline.load_state(config)
            logger.info(f"Resuming AutoML: {pipeline.state.automl_id}")
            pipeline.run()
        except FileNotFoundError:
            logger.error("No AutoML state to resume from")
        return

    if args.report:
        try:
            pipeline = AutoMLPipeline.load_state(config)
            pipeline._generate_report()
        except FileNotFoundError:
            print("No AutoML run found")
        return

    if args.export_best:
        try:
            pipeline = AutoMLPipeline.load_state(config)
            best = pipeline.state.best_results
            with open(args.export_best, "w") as f:
                yaml.dump(best, f, default_flow_style=False)
            print(f"Best config exported to {args.export_best}")
        except FileNotFoundError:
            print("No AutoML run found")
        return

    # Start new AutoML run
    pipeline = AutoMLPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
