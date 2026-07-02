# Grasping Twin · Day 4 · Evaluate trained model — watch it move

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

model = PPO.load("models/reacher_ppo_day1")

env = gym.make("Reacher-v5", render_mode="human")

episode_rewards = []
successes = 0
n_episodes = 10

print(f"Running {n_episodes} episodes — watch the window that opens")
print("-" * 50)

for ep in range(n_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()

    episode_rewards.append(total_reward)
    dist = np.linalg.norm(obs[8:10])
    success = dist < 0.05
    if success:
        successes += 1

    print(f"Episode {ep+1}: reward={total_reward:.2f}, final_dist={dist:.4f}, success={success}")

print("-" * 50)
print(f"Mean reward: {np.mean(episode_rewards):.2f}")
print(f"Std reward: {np.std(episode_rewards):.2f}")
print(f"Success rate: {successes}/{n_episodes} = {successes/n_episodes*100:.0f}%")

env.close()
