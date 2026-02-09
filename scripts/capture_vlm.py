#!/usr/bin/env python
"""
capture_vlm.py — Play a trained policy, capture frames, and send them to a VLM
(GitHub Copilot CLI with gpt-4.1) for visual behavior analysis and bug detection.

Usage:
    # Basic: play best policy, capture 20 frames, analyze with VLM
    uv run scripts/capture_vlm.py --env vbot_navigation_section001

    # Specify policy, capture interval, and number of frames
    uv run scripts/capture_vlm.py --env vbot_navigation_section001 \
        --policy runs/vbot_navigation_section001/.../best_agent.pt \
        --capture-every 30 --max-frames 30

    # Capture only (no VLM analysis)
    uv run scripts/capture_vlm.py --env vbot_navigation_section001 --no-vlm

    # Custom VLM prompt
    uv run scripts/capture_vlm.py --env vbot_navigation_section001 \
        --vlm-prompt "Focus on leg coordination and gait symmetry"

    # Use a specific Copilot CLI model
    uv run scripts/capture_vlm.py --env vbot_navigation_section001 --vlm-model gpt-4.1
"""

import ctypes
import ctypes.wintypes
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from absl import app, flags

# ── Environment registration (same pattern as play.py) ─────────────────────
_NAV2_ENVS = {
    "vbot_navigation_section011",
    "vbot_navigation_section012",
    "vbot_navigation_section013",
    "vbot_navigation_long_course",
}
_env_name_for_import = None
for _i, _arg in enumerate(sys.argv):
    if _arg == "--env" and _i + 1 < len(sys.argv):
        _env_name_for_import = sys.argv[_i + 1]
        break

if _env_name_for_import in _NAV2_ENVS:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter_kit"))
    import navigation2  # noqa: F401
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter_kit" / "navigation1"))
    import vbot  # noqa: F401

from motrix_envs import registry as env_registry  # noqa: E402
from motrix_envs.np.renderer import NpRenderer  # noqa: E402
from motrix_rl import registry  # noqa: E402
from motrix_rl.skrl import get_log_dir  # noqa: E402

logger = logging.getLogger(__name__)

# ── Flags ──────────────────────────────────────────────────────────────────
_ENV = flags.DEFINE_string("env", "vbot_navigation_section001", "Environment name")
_POLICY = flags.DEFINE_string("policy", None, "Policy checkpoint path (auto-discovers best if omitted)")
_TRAIN_BACKEND = flags.DEFINE_string("train-backend", "torch", "Backend for policy inference (torch/jax)")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 1, "Number of parallel envs for playback")
_SEED = flags.DEFINE_integer("seed", 42, "Random seed")

# Capture settings
_CAPTURE_EVERY = flags.DEFINE_integer("capture-every", 15, "Capture a frame every N simulation steps")
_MAX_FRAMES = flags.DEFINE_integer("max-frames", 20, "Maximum number of frames to capture")
_WARMUP_STEPS = flags.DEFINE_integer("warmup-steps", 30, "Steps before starting capture (let robot initialize)")
_OUTPUT_DIR = flags.DEFINE_string("output-dir", None, "Output directory for frames (auto-generated if omitted)")
_CAPTURE_DELAY = flags.DEFINE_float("capture-delay", 0.15, "Delay in seconds after render before capturing frame")
_WINDOW_SIZE = flags.DEFINE_string("window-size", "1920x1080", "Render window size WxH (e.g. 1920x1080, 2560x1440, max)")

# VLM settings
_NO_VLM = flags.DEFINE_bool("no-vlm", False, "Skip VLM analysis, only capture frames")
_VLM_MODEL = flags.DEFINE_string("vlm-model", "gpt-4.1", "Copilot CLI model to use for VLM analysis")
_VLM_PROMPT = flags.DEFINE_string("vlm-prompt", None, "Custom prompt for VLM analysis (appended to default)")
_VLM_BATCH_SIZE = flags.DEFINE_integer("vlm-batch-size", 10, "Max frames per VLM invocation (to avoid context limits)")


def find_best_policy(env_name: str) -> str:
    """Auto-discover the best policy checkpoint for the given env."""
    env_dir = Path(get_log_dir(env_name))
    if not env_dir.exists():
        raise FileNotFoundError(f"No training results for '{env_name}' in {env_dir}")

    training_runs = [d for d in env_dir.iterdir() if d.is_dir()]
    if not training_runs:
        raise FileNotFoundError(f"No training runs for '{env_name}'")

    latest_run = max(training_runs, key=lambda x: x.stat().st_mtime)
    ckpt_dir = latest_run / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints in {latest_run}")

    best_files = list(ckpt_dir.glob("best_agent.*"))
    if best_files:
        return str(best_files[0])

    checkpoint_files = list(ckpt_dir.glob("agent_*.pt")) + list(ckpt_dir.glob("agent_*.pickle"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No policy files in {ckpt_dir}")

    def extract_ts(f):
        parts = Path(f).stem.split("_")
        try:
            return int(parts[1]) if len(parts) >= 2 else 0
        except ValueError:
            return 0

    return str(max(checkpoint_files, key=extract_ts))


# ── Win32 DPI awareness (must be set early) ────────────────────────────────
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # fallback
    except Exception:
        pass

# ── Win32 window management ────────────────────────────────────────────────
_MOTRIX_WINDOW_TITLE = "MotrixRender"
_render_hwnd: int = 0  # cached handle


def _find_motrix_window(max_retries: int = 10, retry_delay: float = 0.5) -> int:
    """Find the MotrixRender window handle, waiting for it to appear."""
    global _render_hwnd
    if _render_hwnd and ctypes.windll.user32.IsWindow(_render_hwnd):
        return _render_hwnd

    user32 = ctypes.windll.user32
    for attempt in range(max_retries):
        hwnd = user32.FindWindowW(None, _MOTRIX_WINDOW_TITLE)
        if hwnd:
            _render_hwnd = hwnd
            return hwnd
        time.sleep(retry_delay)
    return 0


def _close_render_window() -> None:
    """Close the MotrixRender window via Win32 WM_CLOSE message."""
    global _render_hwnd
    hwnd = _find_motrix_window(max_retries=1, retry_delay=0)
    if hwnd:
        WM_CLOSE = 0x0010
        ctypes.windll.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        _render_hwnd = 0
        time.sleep(0.3)
        logger.info("MotrixRender window closed")
    else:
        logger.warning("No MotrixRender window found to close")


def setup_render_window(width: int = 1920, height: int = 1080) -> bool:
    """
    Find the MotrixRender window, resize it, and bring it to the foreground.

    Call this once after the renderer is created and the first frame is rendered.
    Returns True if setup succeeded.
    """
    hwnd = _find_motrix_window()
    if not hwnd:
        logger.warning("Could not find MotrixRender window")
        return False

    user32 = ctypes.windll.user32

    # Restore if minimized (SW_RESTORE = 9)
    user32.ShowWindow(hwnd, 9)
    time.sleep(0.1)

    # Get screen dimensions for centering
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    # Clamp to screen size
    width = min(width, screen_w)
    height = min(height, screen_h)

    # Center on screen
    x = max(0, (screen_w - width) // 2)
    y = max(0, (screen_h - height) // 2)

    # Resize and position: MoveWindow(hwnd, x, y, width, height, repaint)
    user32.MoveWindow(hwnd, x, y, width, height, True)

    # Bring to foreground
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)  # Wait for repaint

    # Verify
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    actual_w = rect.right - rect.left
    actual_h = rect.bottom - rect.top
    logger.info(f"MotrixRender window resized to {actual_w}x{actual_h} at ({rect.left}, {rect.top})")
    return True


def capture_window_screenshot(output_path: str, delay: float = 0.0) -> bool:
    """
    Capture only the MotrixRender window using PIL ImageGrab.

    Falls back to full-screen grab if the window cannot be found.
    Returns True if capture succeeded.
    """
    if delay > 0:
        time.sleep(delay)

    try:
        from PIL import ImageGrab

        hwnd = _find_motrix_window(max_retries=1, retry_delay=0)
        if hwnd:
            user32 = ctypes.windll.user32
            # Bring to front (non-blocking — avoids stealing focus repeatedly)
            user32.SetForegroundWindow(hwnd)

            # Get the window client area bounding box
            rect = ctypes.wintypes.RECT()
            # Use GetClientRect + ClientToScreen for content area (no title bar / borders)
            user32.GetClientRect(hwnd, ctypes.byref(rect))

            # Convert client coords to screen coords
            point = ctypes.wintypes.POINT(rect.left, rect.top)
            user32.ClientToScreen(hwnd, ctypes.byref(point))
            left, top = point.x, point.y

            point2 = ctypes.wintypes.POINT(rect.right, rect.bottom)
            user32.ClientToScreen(hwnd, ctypes.byref(point2))
            right, bottom = point2.x, point2.y

            img = ImageGrab.grab(bbox=(left, top, right, bottom))
        else:
            logger.warning("MotrixRender window not found, falling back to full-screen capture")
            img = ImageGrab.grab()

        img.save(output_path)
        return True
    except Exception as e:
        logger.warning(f"Screenshot failed: {e}")
        return False


def run_vlm_analysis(frames_dir: str, env_name: str, model: str, custom_prompt: str = None, batch_size: int = 10) -> str:
    """
    Send captured frames to GitHub Copilot CLI (gpt-4.1) for VLM analysis.

    Returns the VLM response text.
    """
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frames:
        return "No frames found for analysis."

    # Ensure absolute path for Copilot CLI --add-dir
    abs_frames_dir = str(Path(frames_dir).resolve())

    # Build the analysis prompt
    default_prompt = (
        f"You are analyzing simulation frames from a quadruped robot (VBot) policy evaluation "
        f"in the '{env_name}' environment. The frames are captured in chronological order.\n\n"
        "For each frame (or sequence of frames), analyze:\n"
        "1. **Robot Pose & Stability**: Is the robot upright? Body tilt angle? Center of mass over support polygon?\n"
        "2. **Gait Quality**: Leg coordination, stride symmetry, foot placement pattern. Is it a trot, walk, or irregular gait?\n"
        "3. **Navigation Progress**: Is the robot moving toward the target? Is it stuck, circling, or drifting?\n"
        "4. **Failure Modes / Bugs**: Any falls, flips, leg tangling, jittering, or unnatural poses?\n"
        "5. **Terrain Interaction**: How does the robot handle the ground? Any sliding, penetration, or bouncing?\n\n"
        "After analyzing individual frames, provide:\n"
        "- **Overall Behavior Summary**: What is the learned policy doing?\n"
        "- **Detected Issues**: List any bugs or problematic behaviors with frame numbers\n"
        "- **Reward Engineering Suggestions**: Based on visual issues, what reward/penalty adjustments would help?\n"
        "- **Training Recommendations**: Should training continue, or does the policy need fundamental changes?\n"
    )

    if custom_prompt:
        default_prompt += f"\n**Additional Focus**: {custom_prompt}\n"

    # Process frames in batches to stay within context limits
    all_responses = []
    for batch_start in range(0, len(frames), batch_size):
        batch_frames = frames[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(frames) + batch_size - 1) // batch_size

        if total_batches > 1:
            batch_prompt = f"[Batch {batch_num}/{total_batches}] " + default_prompt
        else:
            batch_prompt = default_prompt

        # Build copilot CLI command
        cmd = [
            "copilot",
            "--model", model,
            "--allow-all",
            "--add-dir", abs_frames_dir,
        ]

        # Add frame file references to the prompt with absolute paths
        frame_refs = "\n".join(
            f"- Frame {i + batch_start}: {f.resolve()} (step {f.stem.split('_')[1] if '_' in f.stem else '?'})"
            for i, f in enumerate(batch_frames)
        )
        full_prompt = f"{batch_prompt}\n\nFrames to analyze:\n{frame_refs}\n\nLook at each PNG file listed above and describe what you see."

        cmd.extend(["-p", full_prompt, "-s"])

        logger.info(f"Sending batch {batch_num}/{total_batches} ({len(batch_frames)} frames) to {model}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout per batch
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            if result.returncode == 0:
                all_responses.append(result.stdout.strip())
            else:
                err_msg = f"Copilot CLI error (batch {batch_num}): {result.stderr.strip()}"
                logger.error(err_msg)
                all_responses.append(err_msg)
        except FileNotFoundError:
            msg = "ERROR: 'copilot' CLI not found. Install GitHub Copilot CLI: https://docs.github.com/en/copilot/using-github-copilot/using-github-copilot-in-the-command-line"
            logger.error(msg)
            return msg
        except subprocess.TimeoutExpired:
            msg = f"VLM analysis timed out for batch {batch_num}"
            logger.warning(msg)
            all_responses.append(msg)

    return "\n\n---\n\n".join(all_responses)


def main(argv):
    env_name = _ENV.value
    backend = _TRAIN_BACKEND.value

    # ── Resolve policy path ────────────────────────────────────────────────
    if _POLICY.present:
        policy_path = _POLICY.value
    else:
        try:
            policy_path = find_best_policy(env_name)
            logger.info(f"Auto-discovered policy: {policy_path}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Train a model first or specify --policy")
            return

    # ── Setup output directory ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if _OUTPUT_DIR.present:
        output_dir = Path(_OUTPUT_DIR.value)
    else:
        output_dir = Path(f"starter_kit_log/vlm_captures/{env_name}/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"VLM Policy Analysis — {env_name}")
    print(f"{'=' * 60}")
    print(f"Policy:        {policy_path}")
    print(f"Output:        {output_dir}")
    print(f"Capture every: {_CAPTURE_EVERY.value} steps")
    print(f"Max frames:    {_MAX_FRAMES.value}")
    print(f"Warmup steps:  {_WARMUP_STEPS.value}")
    print(f"VLM model:     {_VLM_MODEL.value}")
    print(f"VLM analysis:  {'OFF' if _NO_VLM.value else 'ON'}")
    print(f"Window size:   {_WINDOW_SIZE.value}")
    print(f"{'=' * 60}")

    # ── Create environment and renderer ─────────────────────────────────────
    num_envs = _NUM_ENVS.value
    env = env_registry.make(env_name, num_envs=num_envs)
    renderer = NpRenderer(env)

    # Initial render to create the window
    import numpy as np
    action_dim = env.action_space.shape[-1]
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)
    env.step(actions)
    renderer.render()
    time.sleep(0.3)

    # ── Resize render window ────────────────────────────────────────────────
    win_size_str = _WINDOW_SIZE.value
    if win_size_str.lower() == "max":
        user32 = ctypes.windll.user32
        win_w = user32.GetSystemMetrics(0)
        win_h = user32.GetSystemMetrics(1)
    else:
        try:
            win_w, win_h = (int(x) for x in win_size_str.lower().split("x"))
        except ValueError:
            print(f"WARNING: Invalid --window-size '{win_size_str}', using 1920x1080")
            win_w, win_h = 1920, 1080

    if setup_render_window(win_w, win_h):
        print(f"Render window: {win_w}x{win_h}")
    else:
        print("WARNING: Could not resize render window; captures may show the robot very small")

    # ── Load policy ─────────────────────────────────────────────────────────
    if backend == "torch":
        import torch
        from skrl import config as skrl_config

        skrl_config.torch.backend = "torch"

        rlcfg = registry.default_rl_cfg(env_name, "skrl", backend="torch")
        from motrix_rl.skrl.torch import wrap_env
        from motrix_rl.skrl.torch.train.ppo import Trainer as TorchTrainer

        trainer_obj = TorchTrainer(env_name, enable_render=True, cfg_override={"play_num_envs": num_envs, "seed": _SEED.value})
        # We need to build the agent separately for inference
        from motrix_rl.skrl.torch import wrap_env as torch_wrap_env
        from motrix_rl.skrl.torch.train.ppo import _get_cfg

        skrl_env = torch_wrap_env(env, enable_render=False)  # no double renderer
        models = trainer_obj._make_model(skrl_env, rlcfg)
        ppo_cfg = _get_cfg(rlcfg, skrl_env)
        agent = trainer_obj._make_agent(models, skrl_env, ppo_cfg)
        agent.load(policy_path)
        use_torch = True
    else:
        # JAX backend
        from skrl import config as skrl_config

        skrl_config.jax.backend = "jax"

        rlcfg = registry.default_rl_cfg(env_name, "skrl", backend="jax")
        from motrix_rl.skrl.jax import wrap_env as jax_wrap_env
        from motrix_rl.skrl.jax.train.ppo import Trainer as JaxTrainer, _get_cfg

        trainer_obj = JaxTrainer(env_name, enable_render=True, cfg_override={"play_num_envs": num_envs, "seed": _SEED.value})
        skrl_env = jax_wrap_env(env, enable_render=False)
        models = trainer_obj._make_model(skrl_env, rlcfg)
        ppo_cfg = _get_cfg(rlcfg, skrl_env)
        agent = trainer_obj._make_agent(models, skrl_env, ppo_cfg)
        agent.load(policy_path)
        use_torch = False

    # ── Play + Capture loop ─────────────────────────────────────────────────
    print("\nStarting policy playback with frame capture...")
    print(f"Warming up for {_WARMUP_STEPS.value} steps...")

    if use_torch:
        import torch
        ctx = torch.inference_mode()
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    captured_frames = []
    step_count = 0
    fps = 30  # Slower FPS for capture quality

    with ctx:
        obs, _ = skrl_env.reset()

        while len(captured_frames) < _MAX_FRAMES.value:
            t0 = time.time()

            # Agent acts
            outputs = agent.act(obs, timestep=0, timesteps=0)
            if use_torch:
                actions = outputs[-1].get("mean_actions", outputs[0])
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])

            obs, reward, terminated, truncated, info = skrl_env.step(actions)

            # Render the frame
            renderer.render()

            step_count += 1

            # Capture frame after warmup, at the specified interval
            if step_count > _WARMUP_STEPS.value and (step_count - _WARMUP_STEPS.value) % _CAPTURE_EVERY.value == 0:
                frame_path = str(output_dir / f"frame_{step_count:05d}.png")
                success = capture_window_screenshot(frame_path, delay=_CAPTURE_DELAY.value)
                if success:
                    captured_frames.append(frame_path)
                    n = len(captured_frames)
                    print(f"  [{n}/{_MAX_FRAMES.value}] Captured frame at step {step_count} → {Path(frame_path).name}")

            # Throttle to target FPS
            elapsed = time.time() - t0
            if elapsed < 1.0 / fps:
                time.sleep(1.0 / fps - elapsed)

    print(f"\nCapture complete: {len(captured_frames)} frames saved to {output_dir}")

    # ── Close the simulation window ─────────────────────────────────────────
    _close_render_window()
    del renderer
    del env

    # ── Save metadata ───────────────────────────────────────────────────────
    meta_path = output_dir / "capture_metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"Environment: {env_name}\n")
        f.write(f"Policy: {policy_path}\n")
        f.write(f"Backend: {backend}\n")
        f.write(f"Num envs: {num_envs}\n")
        f.write(f"Capture every: {_CAPTURE_EVERY.value} steps\n")
        f.write(f"Warmup steps: {_WARMUP_STEPS.value}\n")
        f.write(f"Total frames: {len(captured_frames)}\n")
        f.write(f"Total steps: {step_count}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\nFrames:\n")
        for fp in captured_frames:
            f.write(f"  {fp}\n")

    # ── VLM Analysis ────────────────────────────────────────────────────────
    if _NO_VLM.value:
        print("\nVLM analysis skipped (--no-vlm).")
        print(f"Run analysis later with:")
        print(f'  copilot --model {_VLM_MODEL.value} --allow-all --add-dir "{output_dir}" -p "Analyze robot behavior in these frames" -s')
        return

    print(f"\nSending {len(captured_frames)} frames to VLM ({_VLM_MODEL.value})...")
    analysis = run_vlm_analysis(
        frames_dir=str(output_dir),
        env_name=env_name,
        model=_VLM_MODEL.value,
        custom_prompt=_VLM_PROMPT.value,
        batch_size=_VLM_BATCH_SIZE.value,
    )

    # Save analysis report
    report_path = output_dir / "vlm_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# VLM Policy Analysis Report\n\n")
        f.write(f"**Environment:** `{env_name}`\n")
        f.write(f"**Policy:** `{policy_path}`\n")
        f.write(f"**Model:** `{_VLM_MODEL.value}`\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Frames analyzed:** {len(captured_frames)}\n\n")
        f.write(f"---\n\n")
        f.write(analysis)
        f.write(f"\n")

    print(f"\n{'=' * 60}")
    print(f"VLM Analysis Report saved to: {report_path}")
    print(f"{'=' * 60}")
    print(f"\n{analysis[:2000]}{'...' if len(analysis) > 2000 else ''}")


if __name__ == "__main__":
    app.run(main)
