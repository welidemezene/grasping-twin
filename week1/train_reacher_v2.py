# Grasping Twin · Day 2 · Low learning rate experiment

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def make_env():
    env = gym.make("Reacher-v5")
    env = Monitor(env, "logs/monitor_v2")
    return env

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="logs/"
)

print("Experiment 2 — learning_rate=0.0001")
print("This is 3x slower than Day 1")
print("Watch how the reward curve climbs differently")
print("-" * 50)

model.learn(
    total_timesteps=300_000,
    tb_log_name="ppo_reacher_lr_low"
)

model.save("models/reacher_ppo_lr_low")
print("Done. Model saved.")
env.close()
