"""Check which reward terms are actually firing under the Stage 1 curriculum."""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch, gymnasium as gym
import isaaclab_tasks  # noqa: F401
from curriculum_lift_cfg import FrankaLiftStage1Cfg

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = 1
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)

obs, _ = env.reset()
for step in range(40):
    actions = torch.rand(env.unwrapped.num_envs, *env.action_space.shape[1:],
                         device=env.unwrapped.device) * 2 - 1
    obs, rew, terminated, truncated, info = env.step(actions)
    terms = env.unwrapped.reward_manager.get_active_iterable_terms(env_idx=0)
    print(f"--- step {step} | total: {rew.item():.4f} ---")
    for name, value in terms:
        print(f"    {name}: {value}")

env.close()
simulation_app.close()
