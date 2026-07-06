"""PPO training on the Isaac Lab Franka Cube Lift task, using stable-baselines3."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=200_000)
args = parser.parse_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import isaaclab_tasks  # noqa: F401  registers Isaac Lab gym environments
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

task_name = "Isaac-Lift-Cube-Franka-v0"
env_cfg = parse_env_cfg(task_name, num_envs=args.num_envs)

env = gym.make(task_name, cfg=env_cfg)
env = Sb3VecEnvWrapper(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=32,
    batch_size=max(args.num_envs * 32 // 4, 8),
    n_epochs=5,
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="week2/logs/franka_lift_ppo",
)

checkpoint_callback = CheckpointCallback(
    save_freq=max(10_000 // args.num_envs, 1),
    save_path="week2/checkpoints",
    name_prefix="franka_lift_ppo",
)

model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
model.save("week2/checkpoints/franka_lift_ppo_final")

print("Training complete. Model saved to week2/checkpoints/franka_lift_ppo_final")

env.close()
simulation_app.close()
