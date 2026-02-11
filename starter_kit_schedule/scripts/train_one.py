"""
Single experiment runner for the training pipeline.

Runs one training experiment with specified reward scales and PPO hyperparameters.
Invoked as a subprocess by the AutoML orchestrator (automl.py).

Usage:
    uv run starter_kit_schedule/scripts/train_one.py --config experiment.json
"""

import argparse
import datetime
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Known env-name → starter_kit directory mappings
_STARTER_KIT_DIRS = {
    "navigation1": os.path.join(PROJECT_ROOT, "starter_kit", "navigation1"),
    "navigation2": os.path.join(PROJECT_ROOT, "starter_kit", "navigation2"),
}


def _resolve_starter_kit_dir(config):
    """Resolve the starter_kit directory for env registration.

    Priority:
        1. Explicit ``starter_kit_dir`` in the config JSON.
        2. Heuristic based on env_name (navigation1/navigation2).
    """
    explicit = config.get("starter_kit_dir")
    if explicit and os.path.isdir(explicit):
        return explicit

    env_name = config.get("env_name", "")
    if "navigation2" in env_name or "stairs" in env_name:
        return _STARTER_KIT_DIRS["navigation2"]
    # Default to navigation1
    return _STARTER_KIT_DIRS["navigation1"]


def main():
    parser = argparse.ArgumentParser(description="Run a single pipeline training experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    run_tag = config.get("run_tag", f"exp_{int(time.time())}")
    reward_scales = config.get("reward_scales", {})
    rl_overrides = config.get("rl_overrides", {})
    env_name = config.get("env_name", "vbot_navigation_section001")

    # Convert list values to tuples for PPO config fields that expect tuples
    tuple_fields = {"policy_hidden_layer_sizes", "value_hidden_layer_sizes"}
    for key in tuple_fields:
        if key in rl_overrides and isinstance(rl_overrides[key], list):
            rl_overrides[key] = tuple(rl_overrides[key])

    # Setup sys.path so the starter_kit's env registration module can be imported
    starter_kit_dir = _resolve_starter_kit_dir(config)
    if starter_kit_dir not in sys.path:
        sys.path.insert(0, starter_kit_dir)
    import vbot  # noqa: F401 — triggers env registration

    # Monkey-patch registry.make to inject reward scale overrides
    from motrix_envs import registry as env_registry

    _original_make = env_registry.make

    def _patched_make(name, *args, **kwargs):
        env = _original_make(name, *args, **kwargs)
        if name == env_name and reward_scales:
            # env._cfg.reward_config.scales is the same dict as env.reward_scales (alias),
            # so in-place update modifies both references
            env._cfg.reward_config.scales.update(reward_scales)
        return env

    env_registry.make = _patched_make

    # List existing run dirs BEFORE training to identify the new one later
    runs_dir = os.path.join("runs", env_name)
    existing_dirs = set()
    if os.path.exists(runs_dir):
        existing_dirs = set(os.listdir(runs_dir))

    # Record start time — the SKRL agent creates its directory ~seconds after train() starts.
    # We use this to match the correct directory by its timestamp-based name, avoiding
    # contamination from concurrently-launched manual training runs.
    pre_train_dt = datetime.datetime.now()

    # Import and run trainer
    from motrix_rl.skrl.torch.train import ppo

    print(f"[Pipeline] Starting experiment: {run_tag}")
    print(f"[Pipeline] Reward overrides: {json.dumps(reward_scales, indent=2)}")
    print(f"[Pipeline] RL overrides: {json.dumps({k: str(v) for k, v in rl_overrides.items()}, indent=2)}")

    start_time = time.time()
    trainer = ppo.Trainer(env_name, cfg_override=rl_overrides)

    # Warm-start from checkpoint if specified (loads policy/value weights only, not optimizer state)
    checkpoint_path = config.get("checkpoint")
    trainer.train(checkpoint=checkpoint_path)
    elapsed = time.time() - start_time

    # Find the new run directory created during training.
    # ROBUST: Match by directory name timestamp (YY-MM-DD_HH-MM-SS-FFFFFF_PPO)
    # to pre_train_dt, avoiding contamination from concurrent manual runs.
    run_dir = None
    if os.path.exists(runs_dir):
        current_dirs = set(os.listdir(runs_dir))
        new_dirs = current_dirs - existing_dirs
        if new_dirs:
            best_dir = None
            best_diff = float("inf")
            for d in new_dirs:
                try:
                    # Parse timestamp from dir name: "26-02-10_00-37-48-487151_PPO"
                    ts_str = d.rsplit("_PPO", 1)[0]
                    dt = datetime.datetime.strptime(ts_str, "%y-%m-%d_%H-%M-%S-%f")
                    diff = abs((dt - pre_train_dt).total_seconds())
                    if diff < best_diff:
                        best_diff = diff
                        best_dir = d
                except (ValueError, IndexError):
                    continue
            if best_dir and best_diff < 120:
                run_dir = os.path.join(runs_dir, best_dir)
                print(f"[Pipeline] Matched run dir by timestamp (delta={best_diff:.1f}s): {best_dir}")
            elif new_dirs:
                # Fallback: if no timestamp match, use newest by mtime (legacy behavior)
                run_dir = os.path.join(
                    runs_dir,
                    max(new_dirs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
                )
                print(f"[Pipeline] WARNING: Timestamp match failed, using mtime fallback: {os.path.basename(run_dir)}")

    # Save experiment metadata
    if run_dir:
        metadata = {
            "run_tag": run_tag,
            "env_name": env_name,
            "reward_scales": reward_scales,
            "rl_overrides": {k: str(v) for k, v in rl_overrides.items()},
            "run_dir": run_dir,
            "elapsed_seconds": elapsed,
            "start_time": start_time,
        }
        meta_path = os.path.join(run_dir, "experiment_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # Print the run directory for the pipeline to parse
    print(f"PIPELINE_RUN_DIR={run_dir}")
    print(f"PIPELINE_ELAPSED={elapsed:.1f}")
    print(f"[Pipeline] Experiment {run_tag} finished in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
