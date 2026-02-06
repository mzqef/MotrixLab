# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""
Initialize a new training campaign with hyperparameter search.

Usage:
    uv run starter_kit_schedule/scripts/init_campaign.py \
        --name "My Campaign" \
        --env anymal_c_navigation_flat \
        --search-method grid
"""

import os
import sys
import yaml
import itertools
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

from absl import app, flags

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FLAGS = flags.FLAGS

flags.DEFINE_string("name", None, "Campaign name", required=True)
flags.DEFINE_string("env", "anymal_c_navigation_flat", "Environment name")
flags.DEFINE_string("search_method", "grid", "Search method: grid, random, bayesian")
flags.DEFINE_integer("max_trials", None, "Maximum number of trials (for random/bayesian)")
flags.DEFINE_string("plan", None, "Path to existing plan file to use")
flags.DEFINE_integer("max_env_steps", 100_000_000, "Maximum environment steps per trial")
flags.DEFINE_integer("checkpoint_interval", 1000, "Checkpoint save interval")

SCHEDULE_DIR = PROJECT_ROOT / "starter_kit_schedule"
LOG_DIR = PROJECT_ROOT / "starter_kit_log"


@dataclass
class SearchSpace:
    """Default hyperparameter search space for PPO."""
    
    # Fixed parameters
    fixed: dict = field(default_factory=lambda: {
        "seed": 42,
        "num_envs": 2048,
        "discount_factor": 0.99,
        "lambda_param": 0.95,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
    })
    
    # Searchable parameters
    searchable: dict = field(default_factory=lambda: {
        "learning_rate": {
            "type": "choice",
            "values": [1e-4, 3e-4, 1e-3]
        },
        "policy_hidden_layer_sizes": {
            "type": "choice",
            "values": [[256, 128, 64], [512, 256, 128], [256, 256, 256]]
        },
        "value_hidden_layer_sizes": {
            "type": "choice",
            "values": [[256, 128, 64], [512, 256, 128], [256, 256, 256]]
        },
        "rollouts": {
            "type": "choice",
            "values": [24, 32, 48]
        },
        "learning_epochs": {
            "type": "choice",
            "values": [4, 5, 6]
        },
        "mini_batches": {
            "type": "choice",
            "values": [16, 32, 64]
        },
    })


def generate_grid_configs(search_space: SearchSpace) -> list[dict]:
    """Generate all configurations for grid search."""
    searchable = search_space.searchable
    fixed = search_space.fixed
    
    # Get all parameter names and their values
    param_names = list(searchable.keys())
    param_values = [searchable[p]["values"] for p in param_names]
    
    # Generate all combinations
    configs = []
    for combo in itertools.product(*param_values):
        config = fixed.copy()
        for name, value in zip(param_names, combo):
            config[name] = value
        configs.append(config)
    
    return configs


def generate_random_configs(search_space: SearchSpace, n_trials: int) -> list[dict]:
    """Generate random configurations for random search."""
    import random
    
    searchable = search_space.searchable
    fixed = search_space.fixed
    
    configs = []
    for _ in range(n_trials):
        config = fixed.copy()
        for param, spec in searchable.items():
            if spec["type"] == "choice":
                config[param] = random.choice(spec["values"])
            elif spec["type"] == "loguniform":
                import math
                log_low = math.log(spec["low"])
                log_high = math.log(spec["high"])
                config[param] = math.exp(random.uniform(log_low, log_high))
            elif spec["type"] == "uniform":
                config[param] = random.uniform(spec["low"], spec["high"])
        configs.append(config)
    
    return configs


def create_campaign(name: str, env: str, search_method: str, 
                    max_trials: int | None, max_env_steps: int,
                    checkpoint_interval: int) -> str:
    """Create a new training campaign."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_id = f"campaign_{timestamp}"
    safe_name = name.replace(" ", "_").lower()
    
    # Create search space
    search_space = SearchSpace()
    
    # Generate configurations based on search method
    if search_method == "grid":
        configs = generate_grid_configs(search_space)
        if max_trials:
            configs = configs[:max_trials]
    elif search_method == "random":
        n_trials = max_trials or 20
        configs = generate_random_configs(search_space, n_trials)
    elif search_method == "bayesian":
        # For Bayesian, we generate initial configs and add more during search
        n_initial = min(5, max_trials or 5)
        configs = generate_random_configs(search_space, n_initial)
    else:
        raise ValueError(f"Unknown search method: {search_method}")
    
    # Create plan
    plan = {
        "plan_id": campaign_id,
        "name": name,
        "description": f"Training campaign for {env} using {search_method} search",
        "created_at": datetime.now().isoformat() + "Z",
        "status": "queued",
        
        "environment": {
            "name": env,
            "stage": 1 if "navigation1" in env or "flat" in env else 2,
            "starter_kit_path": f"starter_kit/navigation1/anymal_c" if "flat" in env else "starter_kit/navigation2/anymal_c"
        },
        
        "phases": [
            {
                "phase_id": "main",
                "name": "Main Training",
                "description": "Full training phase",
                "max_env_steps": max_env_steps,
                "checkpoint_interval": checkpoint_interval,
                "early_stopping": {
                    "metric": "episode_reward_mean",
                    "patience": 50,
                    "min_delta": 0.001
                }
            }
        ],
        
        "search": {
            "method": search_method,
            "max_trials": len(configs),
            "parallel_trials": 1,
            "space": asdict(search_space)
        },
        
        "resources": {
            "max_runtime_hours": 48,
            "checkpoint_storage_gb": 50
        },
        
        "evaluation": {
            "primary_metric": "episode_reward_mean",
            "secondary_metrics": ["episode_length_mean", "success_rate"],
            "evaluation_episodes": 100
        }
    }
    
    # Save plan
    plan_path = SCHEDULE_DIR / "plans" / f"{campaign_id}.yaml"
    with open(plan_path, "w") as f:
        yaml.dump(plan, f, default_flow_style=False, allow_unicode=True)
    
    # Also save as active plan
    active_plan_path = SCHEDULE_DIR / "plans" / "active_plan.yaml"
    with open(active_plan_path, "w") as f:
        yaml.dump(plan, f, default_flow_style=False, allow_unicode=True)
    
    # Save individual configs
    configs_dir = SCHEDULE_DIR / "configs" / "generated"
    for i, config in enumerate(configs):
        config_id = f"config_{i+1:03d}"
        config_data = {
            "config_id": config_id,
            "campaign_id": campaign_id,
            "environment": env,
            "hyperparameters": config,
            "max_env_steps": max_env_steps,
            "checkpoint_interval": checkpoint_interval
        }
        config_path = configs_dir / f"{config_id}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    # Initialize progress tracking
    queue = {
        "campaign_id": campaign_id,
        "status": "queued",
        "created_at": datetime.now().isoformat() + "Z",
        "configs_pending": [f"config_{i+1:03d}" for i in range(len(configs))],
        "configs_running": [],
        "configs_completed": [],
        "configs_failed": []
    }
    queue_path = SCHEDULE_DIR / "progress" / "queue.yaml"
    with open(queue_path, "w") as f:
        yaml.dump(queue, f, default_flow_style=False, allow_unicode=True)
    
    # Initialize checkpoint registry
    checkpoint_registry = {
        "checkpoints": [],
        "best_by_env": {env: None}
    }
    registry_path = SCHEDULE_DIR / "checkpoints" / "checkpoint_registry.yaml"
    with open(registry_path, "w") as f:
        yaml.dump(checkpoint_registry, f, default_flow_style=False, allow_unicode=True)
    
    # Create campaign log directory
    campaign_log_dir = LOG_DIR / "campaigns" / campaign_id
    campaign_log_dir.mkdir(parents=True, exist_ok=True)
    
    campaign_config = {
        "campaign_id": campaign_id,
        "name": name,
        "environment": env,
        "search_method": search_method,
        "total_configs": len(configs),
        "created_at": datetime.now().isoformat() + "Z",
        "status": "queued"
    }
    with open(campaign_log_dir / "campaign_config.yaml", "w") as f:
        yaml.dump(campaign_config, f, default_flow_style=False, allow_unicode=True)
    
    # Initialize master index
    index_path = LOG_DIR / "index.yaml"
    if index_path.exists():
        with open(index_path) as f:
            index = yaml.safe_load(f) or {"campaigns": [], "experiments": []}
    else:
        index = {"campaigns": [], "experiments": []}
    
    index["campaigns"].append({
        "campaign_id": campaign_id,
        "name": name,
        "created_at": datetime.now().isoformat() + "Z",
        "status": "queued"
    })
    
    with open(index_path, "w") as f:
        yaml.dump(index, f, default_flow_style=False, allow_unicode=True)
    
    return campaign_id, len(configs)


def main(argv):
    if FLAGS.plan:
        # Load existing plan
        with open(FLAGS.plan) as f:
            plan = yaml.safe_load(f)
        print(f"Loaded plan from {FLAGS.plan}")
        # TODO: Initialize from existing plan
        return
    
    campaign_id, n_configs = create_campaign(
        name=FLAGS.name,
        env=FLAGS.env,
        search_method=FLAGS.search_method,
        max_trials=FLAGS.max_trials,
        max_env_steps=FLAGS.max_env_steps,
        checkpoint_interval=FLAGS.checkpoint_interval
    )
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Campaign Initialized                       ║
╠══════════════════════════════════════════════════════════════╣
║  Campaign ID: {campaign_id:<45} ║
║  Name: {FLAGS.name:<52} ║
║  Environment: {FLAGS.env:<45} ║
║  Search Method: {FLAGS.search_method:<43} ║
║  Total Configurations: {n_configs:<36} ║
╠══════════════════════════════════════════════════════════════╣
║  Plan saved to:                                               ║
║    starter_kit_schedule/plans/{campaign_id}.yaml              
║                                                               ║
║  To start training, run:                                      ║
║    uv run starter_kit_schedule/scripts/run_search.py          ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    app.run(main)
