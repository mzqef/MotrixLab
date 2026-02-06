"""
MotrixLab Navigation1 — Automated Training Pipeline

Thin entry point that configures and invokes the AutoML pipeline
(starter_kit_schedule/scripts/automl.py) for VBot flat-ground navigation.

The heavy lifting — HP search (Bayesian/random), evolutionary reward search,
curriculum progression, state persistence, and reporting — is all handled
by AutoMLPipeline.  This file only provides:
  1. navigation1-specific defaults (env name, stages)
  2. sys.path setup so automl.py can import train_one.py and evaluate.py

Usage:
    uv run starter_kit/navigation1/pipeline/run.py                       # hp-search, 48h
    uv run starter_kit/navigation1/pipeline/run.py --mode reward-search  # reward optimization
    uv run starter_kit/navigation1/pipeline/run.py --mode full           # full pipeline
    uv run starter_kit/navigation1/pipeline/run.py --budget-hours 12     # budget cap
    uv run starter_kit/navigation1/pipeline/run.py --hp-trials 10        # quick HP sweep
    uv run starter_kit/navigation1/pipeline/run.py --resume              # resume interrupted
    uv run starter_kit/navigation1/pipeline/run.py --status              # check progress

Results:  starter_kit_log/<automl_id>/  (self-contained: configs/, experiments/, index.yaml, report.md)
Analysis: uv run starter_kit_schedule/scripts/analyze.py
Status:   uv run starter_kit_schedule/scripts/status.py --watch
"""

import sys
from pathlib import Path

# ---- path setup ----
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "starter_kit_schedule" / "scripts"))

# Default --env to vbot_navigation_section001 if not provided
if "--env" not in sys.argv:
    sys.argv.extend(["--env", "vbot_navigation_section001"])

from automl import main  # noqa: E402

if __name__ == "__main__":
    main()
