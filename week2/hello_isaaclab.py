"""Isaac Lab smoke test: launch headless, spin up Franka Lift task, step random actions."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401  registers Isaac Lab gym environments
from isaaclab_tasks.utils import parse_env_cfg

task_name = "Isaac-Lift-Cube-Franka-v0"
env_cfg = parse_env_cfg(task_name, num_envs=32)
env = gym.make(task_name, cfg=env_cfg)

print(f"Task: {task_name}")
print(f"Num parallel envs: {env.unwrapped.num_envs}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

obs, _ = env.reset()
for step in range(50):
    actions = torch.rand(env.unwrapped.num_envs, *env.action_space.shape[1:], device=env.unwrapped.device) * 2 - 1
    obs, rew, terminated, truncated, info = env.step(actions)
    if step % 10 == 0:
        print(f"step {step}: mean reward = {rew.mean().item():.4f}")

print("Isaac Lab Franka Lift environment ran successfully on GPU.")
env.close()
simulation_app.close()
