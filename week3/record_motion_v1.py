"""Run the trained policy and save joint angles + cube position to JSON.
No rendering — pure numbers."""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import json, torch, gymnasium as gym
from stable_baselines3 import PPO
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from curriculum_lift_cfg import FrankaLiftStage1Cfg

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = 512
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = Sb3VecEnvWrapper(env)

model = PPO.load("week3/checkpoints/stage1_final", env=vec)

frames = []
obs = vec.reset()
for step in range(250):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec.step(action)

    joints = raw.scene["robot"].data.joint_pos[0].cpu().tolist()
    cube = raw.scene["object"].data.root_pos_w[0].cpu().tolist()
    frames.append({"step": step, "joints": joints, "cube": cube})

    if step % 25 == 0:
        print(f"step {step:3d} | cube height z = {cube[2]:.4f} m")

with open("motion.json", "w") as f:
    json.dump(frames, f)

heights = [f["cube"][2] for f in frames]
print(f"\nCube height — start {heights[0]:.4f} m | max {max(heights):.4f} m | end {heights[-1]:.4f} m")
print(f"Rose by {max(heights) - heights[0]:.4f} m above its start.")
print("Saved motion.json —", len(frames), "frames")

vec.close()
simulation_app.close()
