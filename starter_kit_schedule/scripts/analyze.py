# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""
Analyze training results and generate comparison reports.

Usage:
    uv run starter_kit_schedule/scripts/analyze.py
    uv run starter_kit_schedule/scripts/analyze.py --campaign <campaign_id>
    uv run starter_kit_schedule/scripts/analyze.py --export-best best_config.yaml
"""

import os
import sys
import yaml
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from absl import app, flags

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FLAGS = flags.FLAGS

flags.DEFINE_string("campaign", None, "Campaign ID to analyze (default: active)")
flags.DEFINE_string("export_best", None, "Export best configuration to file")
flags.DEFINE_string("metric", "episode_reward_mean", "Primary metric for comparison")
flags.DEFINE_bool("ascending", False, "Sort ascending (default: descending)")

SCHEDULE_DIR = PROJECT_ROOT / "starter_kit_schedule"
LOG_DIR = PROJECT_ROOT / "starter_kit_log"


def load_yaml_safe(path: Path) -> dict | None:
    """Safely load a YAML file."""
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return None


def load_all_experiments(campaign_id: str | None = None) -> list[dict]:
    """Load all experiment summaries, optionally filtered by campaign."""
    experiments = []
    exp_dir = LOG_DIR / "experiments"
    
    if not exp_dir.exists():
        return experiments
    
    for exp_path in exp_dir.iterdir():
        if exp_path.is_dir():
            summary = load_yaml_safe(exp_path / "summary.yaml")
            if summary:
                if campaign_id and summary.get("campaign_id") != campaign_id:
                    continue
                
                # Try to load metrics
                metrics_file = exp_path / "metrics.jsonl"
                if metrics_file.exists():
                    try:
                        with open(metrics_file) as f:
                            metrics = [json.loads(line) for line in f if line.strip()]
                        summary["metrics_history"] = metrics
                    except:
                        summary["metrics_history"] = []
                
                experiments.append(summary)
    
    return experiments


def extract_final_metrics(experiment: dict) -> dict:
    """Extract final metrics from experiment."""
    # Try from summary first
    final_metrics = experiment.get("results", {}).get("final_metrics", {})
    
    # If empty, try from metrics history
    if not final_metrics and experiment.get("metrics_history"):
        # Get last metric entry
        for entry in reversed(experiment["metrics_history"]):
            if isinstance(entry, dict):
                final_metrics = {k: v for k, v in entry.items() 
                               if k not in ["timestamp", "raw_line"]}
                break
    
    return final_metrics


def analyze_hyperparameter_importance(experiments: list[dict], metric: str) -> dict:
    """Analyze which hyperparameters have the most impact on performance."""
    if len(experiments) < 2:
        return {}
    
    # Group experiments by hyperparameter values
    hp_impact = defaultdict(list)
    
    for exp in experiments:
        hp = exp.get("config", {}).get("hyperparameters", {})
        metrics = extract_final_metrics(exp)
        metric_value = metrics.get(metric)
        
        if metric_value is None:
            continue
        
        for param, value in hp.items():
            # Convert lists/tuples to string for grouping
            if isinstance(value, (list, tuple)):
                value = str(value)
            hp_impact[(param, value)].append(metric_value)
    
    # Calculate mean performance for each hyperparameter value
    hp_means = {}
    for (param, value), values in hp_impact.items():
        if param not in hp_means:
            hp_means[param] = {}
        hp_means[param][value] = sum(values) / len(values)
    
    # Calculate variance (importance) for each hyperparameter
    importance = {}
    for param, value_means in hp_means.items():
        if len(value_means) > 1:
            mean_of_means = sum(value_means.values()) / len(value_means)
            variance = sum((v - mean_of_means) ** 2 for v in value_means.values()) / len(value_means)
            importance[param] = {
                "variance": variance,
                "best_value": max(value_means.items(), key=lambda x: x[1])[0],
                "value_means": value_means
            }
    
    return importance


def generate_comparison_report(experiments: list[dict], metric: str, ascending: bool = False) -> dict:
    """Generate a comparison report across experiments."""
    
    # Sort by metric
    def get_metric_value(exp):
        metrics = extract_final_metrics(exp)
        return metrics.get(metric, float('-inf') if not ascending else float('inf'))
    
    sorted_experiments = sorted(experiments, key=get_metric_value, reverse=not ascending)
    
    # Build report
    report = {
        "generated_at": datetime.now().isoformat() + "Z",
        "total_experiments": len(experiments),
        "primary_metric": metric,
        "sort_order": "ascending" if ascending else "descending",
        "rankings": [],
        "best_config": None,
        "hyperparameter_importance": {},
        "recommendations": []
    }
    
    # Add rankings
    for rank, exp in enumerate(sorted_experiments[:20], 1):  # Top 20
        metrics = extract_final_metrics(exp)
        report["rankings"].append({
            "rank": rank,
            "experiment_id": exp.get("experiment_id"),
            "config_id": exp.get("config_id"),
            metric: metrics.get(metric),
            "status": exp.get("execution", {}).get("status"),
            "duration_hours": exp.get("execution", {}).get("duration_hours")
        })
    
    # Best config
    if sorted_experiments:
        best = sorted_experiments[0]
        report["best_config"] = {
            "experiment_id": best.get("experiment_id"),
            "config_id": best.get("config_id"),
            "hyperparameters": best.get("config", {}).get("hyperparameters", {}),
            "metrics": extract_final_metrics(best)
        }
    
    # Hyperparameter importance
    report["hyperparameter_importance"] = analyze_hyperparameter_importance(experiments, metric)
    
    # Generate recommendations
    importance = report["hyperparameter_importance"]
    if importance:
        for param, data in sorted(importance.items(), key=lambda x: -x[1]["variance"])[:5]:
            report["recommendations"].append(
                f"Consider {param}={data['best_value']} (highest avg {metric})"
            )
    
    return report


def print_report(report: dict):
    """Print the comparison report."""
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Experiment Analysis Report                 ║
╠══════════════════════════════════════════════════════════════╣
║  Total Experiments: {report['total_experiments']:<39} ║
║  Primary Metric: {report['primary_metric']:<42} ║
║  Generated: {report['generated_at'][:19]:<47} ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Rankings
    print("📊 Top Configurations:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Config':<15} {report['primary_metric']:<20} {'Duration':<12} {'Status'}")
    print("-" * 70)
    
    for r in report["rankings"][:10]:
        metric_val = r.get(report['primary_metric'])
        metric_str = f"{metric_val:.2f}" if isinstance(metric_val, (int, float)) else "N/A"
        duration = r.get('duration_hours', 0)
        duration_str = f"{duration:.1f}h" if duration else "N/A"
        print(f"{r['rank']:<6} {r.get('config_id', 'N/A'):<15} {metric_str:<20} {duration_str:<12} {r.get('status', 'N/A')}")
    
    print()
    
    # Best config
    if report["best_config"]:
        print("🏆 Best Configuration:")
        print("-" * 70)
        best = report["best_config"]
        print(f"  Experiment: {best.get('experiment_id')}")
        print(f"  Config: {best.get('config_id')}")
        print(f"  Hyperparameters:")
        for k, v in best.get("hyperparameters", {}).items():
            print(f"    {k}: {v}")
        print()
    
    # Hyperparameter importance
    if report["hyperparameter_importance"]:
        print("📈 Hyperparameter Importance (by variance):")
        print("-" * 70)
        sorted_importance = sorted(
            report["hyperparameter_importance"].items(),
            key=lambda x: -x[1]["variance"]
        )
        for param, data in sorted_importance[:5]:
            print(f"  {param}:")
            print(f"    Best value: {data['best_value']}")
            print(f"    Variance: {data['variance']:.4f}")
        print()
    
    # Recommendations
    if report["recommendations"]:
        print("💡 Recommendations:")
        print("-" * 70)
        for rec in report["recommendations"]:
            print(f"  • {rec}")
        print()


def main(argv):
    # Determine campaign
    campaign_id = FLAGS.campaign
    if not campaign_id:
        # Try to get from active plan
        queue = load_yaml_safe(SCHEDULE_DIR / "progress" / "queue.yaml")
        if queue:
            campaign_id = queue.get("campaign_id")
    
    print(f"Analyzing campaign: {campaign_id or 'all'}")
    
    # Load experiments
    experiments = load_all_experiments(campaign_id)
    
    if not experiments:
        print("No experiments found to analyze.")
        print("Run training first: uv run starter_kit_schedule/scripts/run_search.py")
        return
    
    # Generate report
    report = generate_comparison_report(
        experiments, 
        FLAGS.metric, 
        FLAGS.ascending
    )
    
    # Print report
    print_report(report)
    
    # Save report
    report_dir = LOG_DIR / "analysis" / "comparison_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"report_{timestamp}.yaml"
    
    with open(report_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Report saved to: {report_path}")
    
    # Export best config if requested
    if FLAGS.export_best and report["best_config"]:
        best_config = {
            "environment": report["best_config"].get("hyperparameters", {}).get("environment", ""),
            "hyperparameters": report["best_config"]["hyperparameters"],
            "source_experiment": report["best_config"]["experiment_id"],
            "metrics": report["best_config"]["metrics"]
        }
        
        export_path = Path(FLAGS.export_best)
        with open(export_path, "w") as f:
            yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Best config exported to: {export_path}")
        
        # Also save to best_configs directory
        best_dir = LOG_DIR / "analysis" / "best_configs"
        best_dir.mkdir(parents=True, exist_ok=True)
        best_path = best_dir / f"best_{campaign_id or 'all'}_{timestamp}.yaml"
        with open(best_path, "w") as f:
            yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    app.run(main)
