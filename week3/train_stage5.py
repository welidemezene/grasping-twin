"""Stage 1 curriculum training — fixed cube position, lower gate, stronger reaching."""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=2_000_000)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from curriculum_lift_cfg import FrankaLiftStage1Cfg

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=64,
    batch_size=max(args.num_envs * 64 // 4, 8),
    n_epochs=5,
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="week3/logs/stage5",
)

checkpoint_callback = CheckpointCallback(
    save_freq=max(200_000 // args.num_envs, 1),
    save_path="week3/checkpoints",
    name_prefix="stage5",
)

model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
model.save("week3/checkpoints/stage5_final")
print("Stage 1 complete. Model saved.")

env.close()
simulation_app.close()
