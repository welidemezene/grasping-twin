"""Load trained PPO policy and record a video of the Franka Lift task."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

task_name = "Isaac-Lift-Cube-Franka-v0"
env_cfg = parse_env_cfg(task_name, num_envs=1)
env_cfg.viewer.resolution = (640, 480)

env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
env = RecordVideo(env, video_folder="week2/videos", name_prefix="franka_lift", episode_trigger=lambda x: True)
env = Sb3VecEnvWrapper(env)

model = PPO.load("week2/checkpoints/franka_lift_ppo_final", env=env)

obs = env.reset()
for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

print("Video recording complete. Check week2/videos/")
env.close()
simulation_app.close()
