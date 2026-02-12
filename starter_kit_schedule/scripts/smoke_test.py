#!/usr/bin/env python3
"""
Smoke Test & Reward Budget Auditor
====================================
Quick validation that an environment config loads, steps, and has a sane reward budget.

Replaces: _smoke_test.py, _smoke_v7.py, _test_long_course.py

Usage:
    # Smoke test: create env, step 11 times, check for NaN/Inf
    uv run starter_kit_schedule/scripts/smoke_test.py --env vbot_navigation_section011

    # Reward budget audit: compute standing-still vs completing vs sprint-die budgets
    uv run starter_kit_schedule/scripts/smoke_test.py --env vbot_navigation_section011 --budget

    # Both
    uv run starter_kit_schedule/scripts/smoke_test.py --env vbot_navigation_section011 --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]

# Environment import table
ENV_IMPORTS = {
    "vbot_navigation_section001": "starter_kit.navigation1.vbot",
    "vbot_navigation_section011": "starter_kit.navigation2.vbot",
    "vbot_navigation_section012": "starter_kit.navigation2.vbot",
    "vbot_navigation_section013": "starter_kit.navigation2.vbot",
    "vbot_navigation_long_course": "starter_kit.navigation2.vbot",
}


def _register_env(env_name: str):
    """Register environment by importing its module."""
    if env_name in ENV_IMPORTS:
        mod_path = ENV_IMPORTS[env_name]
        parts = mod_path.split(".")
        # Add parent to sys.path for starter_kit imports
        parent_path = str(WORKSPACE / parts[0])
        parent_parent = str(WORKSPACE)
        for p in [parent_parent, parent_path]:
            if p not in sys.path:
                sys.path.insert(0, p)
        import importlib
        importlib.import_module(mod_path)


def _safe_close(env):
    """Close env if it has a close method (NpEnv doesn't)."""
    if hasattr(env, 'close'):
        env.close()


# ============================================================
# Smoke test: create, reset, step, check finite
# ============================================================

def smoke_test(env_name: str, num_envs: int = 4, num_steps: int = 11):
    """Create env, reset, step N times, check all outputs finite."""
    from motrix_envs.registry import make

    _register_env(env_name)
    env = make(env_name, num_envs=num_envs)
    print(f"=== Smoke Test: {env_name} ===")
    print(f"  obs_space: {env.observation_space.shape}")
    print(f"  act_space: {env.action_space.shape}")

    # Initialize state (NpEnv pattern: init_state → step)
    env.init_state()
    print(f"  init_state() OK")

    # First step with zeros (this triggers reset internally)
    actions = np.zeros(env.action_space.shape, dtype=np.float32)
    state = env.step(actions)
    obs2 = state.obs
    reward = state.reward
    assert np.all(np.isfinite(obs2)), "Step[0]: obs contains NaN/Inf!"
    assert np.all(np.isfinite(reward)), "Step[0]: reward contains NaN/Inf!"
    print(f"  Step[0] OK: reward={reward}")

    # Random steps
    for i in range(1, num_steps):
        actions = np.random.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        state = env.step(actions)
        if not np.all(np.isfinite(state.reward)):
            print(f"  ⚠️ Step[{i}]: reward NaN/Inf detected!")
            _safe_close(env)
            return False
        if not np.all(np.isfinite(state.obs)):
            print(f"  ⚠️ Step[{i}]: obs NaN/Inf detected!")
            _safe_close(env)
            return False

    _safe_close(env)
    print(f"  All {num_steps} steps OK. Final reward={state.reward}")
    print("  ✅ PASSED")
    return True


# ============================================================
# Reward budget audit
# ============================================================

def reward_budget_audit(env_name: str):
    """Compute reward budgets for standing still vs completing vs sprint-die."""
    from motrix_envs.registry import make

    _register_env(env_name)
    env = make(env_name, num_envs=1)
    cfg = env._cfg

    # Get reward scales
    reward_cfg = getattr(cfg, 'reward_config', None)
    if reward_cfg is None:
        print("  No reward_config found in env cfg")
        _safe_close(env)
        return
    scales = reward_cfg.scales
    max_steps = getattr(cfg, 'max_episode_steps', 1000)
    grace_steps = getattr(cfg, 'grace_period_steps', 0)
    term_penalty = abs(scales.get('termination', -50.0))

    print(f"\n=== Reward Budget Audit: {env_name} ===")
    print(f"  max_episode_steps: {max_steps}")
    print(f"  grace_period: {grace_steps}")
    print(f"  termination_penalty: {scales.get('termination', 'N/A')}")
    print()

    # Active reward scales
    print("  Active reward scales:")
    for k, v in sorted(scales.items()):
        if v != 0:
            print(f"    {k}: {v}")

    # --- Budget: Standing still ---
    alive_bonus = scales.get('alive_bonus', 0)
    alive_budget = alive_bonus * max_steps
    pos_tracking = scales.get('position_tracking', 0)
    pos_budget = pos_tracking * 0.607 * max_steps  # exp(-1) ≈ 0.37 average at typical dist
    wp_facing = scales.get('waypoint_facing', 0)
    facing_budget = wp_facing * 0.5 * max_steps  # facing partially correct on average
    standing = alive_budget + pos_budget + facing_budget

    # --- Budget: Sprint and die (250 steps typical) ---
    die_steps = 250
    fwd_vel = scales.get('forward_velocity', 0)
    sprint_fwd = fwd_vel * 0.8 * die_steps  # max clipped velocity
    sprint_alive = alive_bonus * die_steps
    sprint_penalty = scales.get('termination', -50.0)
    sprint_total = sprint_fwd + sprint_alive + sprint_penalty

    # --- Budget: Complete task ---
    wp_bonus = scales.get('waypoint_bonus', 0)
    smiley = scales.get('smiley_bonus', 0)
    red_pkt = scales.get('red_packet_bonus', 0)
    celeb = scales.get('celebration_bonus', 0)
    traversal = scales.get('traversal_bonus', 0)

    # Count waypoints if available
    wp_nav = getattr(cfg, 'waypoint_nav', None) or getattr(cfg, 'WaypointNav', lambda: None)()
    num_wp = len(getattr(wp_nav, 'waypoints', [])) if wp_nav else 3
    num_smileys = len(getattr(getattr(cfg, 'scoring_zones', None) or getattr(cfg, 'ScoringZones', lambda: None)(), 'smiley_centers', [])) if hasattr(cfg, 'scoring_zones') or hasattr(cfg, 'ScoringZones') else 3
    num_red = len(getattr(getattr(cfg, 'scoring_zones', None) or getattr(cfg, 'ScoringZones', lambda: None)(), 'red_packet_centers', [])) if hasattr(cfg, 'scoring_zones') or hasattr(cfg, 'ScoringZones') else 3

    goal_bonuses = wp_bonus * num_wp + smiley * num_smileys + red_pkt * num_red + celeb + traversal * 2
    completing = standing + goal_bonuses  # alive + goal bonuses

    print(f"\n  {'='*50}")
    print(f"  STANDING STILL ({max_steps} steps):")
    print(f"    alive:   {alive_budget:.0f} ({alive_bonus} × {max_steps})")
    print(f"    pos:     {pos_budget:.0f}")
    print(f"    facing:  {facing_budget:.0f}")
    print(f"    TOTAL:   {standing:.0f}")

    print(f"\n  SPRINT & DIE ({die_steps} steps):")
    print(f"    fwd_vel: {sprint_fwd:.0f} ({fwd_vel} × 0.8 × {die_steps})")
    print(f"    alive:   {sprint_alive:.0f}")
    print(f"    penalty: {sprint_penalty:.0f}")
    print(f"    TOTAL:   {sprint_total:.0f}")

    print(f"\n  COMPLETE TASK ({max_steps} steps):")
    print(f"    survive: {standing:.0f}")
    print(f"    bonuses: {goal_bonuses:.0f} (wp={wp_bonus}×{num_wp} smiley={smiley}×{num_smileys} red={red_pkt}×{num_red} celeb={celeb} trav={traversal}×2)")
    print(f"    TOTAL:   {completing:.0f}")

    print(f"\n  {'='*50}")
    print(f"  RATIOS:")
    print(f"    survive / sprint-die: {standing / max(sprint_total, 1):.1f}:1  {'✅' if standing > sprint_total * 2 else '⚠️ TOO LOW'}")
    print(f"    complete / standing:  {completing / max(standing, 1):.1f}:1  {'✅' if completing > standing * 1.2 else '⚠️ TOO LOW'}")
    print(f"    complete / sprint:    {completing / max(sprint_total, 1):.1f}:1")

    if standing <= sprint_total * 2:
        print(f"\n  ⚠️ WARNING: Surviving is NOT sufficiently better than sprint-die!")
        print(f"     Robot may learn to sprint recklessly. Increase alive_bonus or decrease forward_velocity.")

    if completing <= standing * 1.1:
        print(f"\n  ⚠️ WARNING: Completing is barely better than standing still!")
        print(f"     Robot may learn to just survive without navigating. Increase goal bonuses.")

    _safe_close(env)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Smoke Test & Reward Budget Auditor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run starter_kit_schedule/scripts/smoke_test.py --env vbot_navigation_section011
  uv run starter_kit_schedule/scripts/smoke_test.py --env vbot_navigation_section011 --budget
  uv run starter_kit_schedule/scripts/smoke_test.py --env vbot_navigation_section011 --all
        """
    )
    parser.add_argument("--env", required=True, help="Environment name")
    parser.add_argument("--budget", action="store_true", help="Run reward budget audit")
    parser.add_argument("--all", action="store_true", help="Run both smoke test and budget audit")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of envs for smoke test")
    parser.add_argument("--num-steps", type=int, default=11, help="Number of steps for smoke test")
    args = parser.parse_args()

    if args.all or (not args.budget):
        ok = smoke_test(args.env, args.num_envs, args.num_steps)
        if not ok:
            sys.exit(1)

    if args.budget or args.all:
        reward_budget_audit(args.env)


if __name__ == "__main__":
    main()
