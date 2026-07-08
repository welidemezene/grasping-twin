"""Step the Franka Lift task slowly, printing each reward component separately so you can watch them change."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

task_name = "Isaac-Lift-Cube-Franka-v0"
env_cfg = parse_env_cfg(task_name, num_envs=1)
env = gym.make(task_name, cfg=env_cfg)

obs, _ = env.reset()
for step in range(30):
    actions = torch.rand(env.unwrapped.num_envs, *env.action_space.shape[1:], device=env.unwrapped.device) * 2 - 1
    obs, rew, terminated, truncated, info = env.step(actions)
    reward_terms = env.unwrapped.reward_manager.get_active_iterable_terms(env_idx=0)
    print(f"--- step {step} | total reward: {rew.item():.4f} ---")
    for name, value in reward_terms:
        print(f"    {name}: {value}")

env.close()
simulation_app.close()
