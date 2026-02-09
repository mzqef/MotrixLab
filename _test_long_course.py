"""Quick test for vbot_navigation_long_course environment"""
import sys
import numpy as np

sys.path.insert(0, "starter_kit/navigation2")
from vbot import VBotLongCourseEnv  # noqa: F401
from motrix_envs import registry

env = registry.make("vbot_navigation_long_course", num_envs=2)
state = env.init_state()

print(f"obs shape: {state.obs.shape}")
print(f"waypoint_idx: {state.info['waypoint_idx']}")
print(f"pose_commands: {state.info['pose_commands']}")
print(f"initial distance: {state.info['initial_distance']}")

# Run a few steps with zero actions
for step in range(10):
    action = np.zeros((2, 12), dtype=np.float32)
    state = env.step(action)
    dist = state.info["metrics"]["distance_to_target"].mean()
    wp = state.info["waypoint_idx"].mean()
    r = state.reward.mean()
    t = state.terminated.any()
    print(f"  step {step}: reward={r:.3f}, term={t}, dist={dist:.2f}, wp_idx={wp:.0f}")

print("\nFull course env test PASSED!")
