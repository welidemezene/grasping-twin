# Grasping Twin · Day 3 · BROKEN training — lr=0.01

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def make_env():
    env = gym.make("Reacher-v5")
    env = Monitor(env, "logs/monitor_broken")
    return env

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.01,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="logs/"
)

print("Day 3 — BROKEN training — lr=0.01")
print("Watch the reward curve. Predict: zigzag.")
print("-" * 50)

model.learn(
    total_timesteps=150_000,
    tb_log_name="ppo_broken_lr_high"
)

print("Done.")
env.close()
