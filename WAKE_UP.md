# WAKE_UP — Current Training State
**Generated**: 2026-02-07T05:55:10.526798

## AutoML Status
- **ID**: automl_20260207_034031
- **Status**: running
- **Phase**: training
- **Iteration**: 14.0
- **Elapsed**: 2.1 / 8.0 hours

## Active Training Runs

- `26-02-07_05-47-42-371699_PPO` (modified 10s ago)
- `26-02-07_05-39-29-476815_PPO` (modified 461s ago)

## Suggested Next Actions
1. Check TensorBoard: `uv run tensorboard --logdir runs/vbot_navigation_section001`
2. Evaluate best: `uv run scripts/play.py --env vbot_navigation_section001`
3. Resume AutoML: `uv run starter_kit_schedule/scripts/automl.py --resume`
4. Check reward curves and adjust weights if plateauing
