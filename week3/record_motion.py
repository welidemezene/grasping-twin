"""Replay a trained checkpoint and save joint angles + cube/gripper/base positions to JSON.

Records the true end-effector frame (same source find_gripper.py uses), the robot
base pose, and the episode-done flag, so gripper-to-cube distance is measured
rather than assumed. Frames after the first episode reset are dropped — the cube
snapping back to its spawn height belongs to the next episode, not this replay.
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import json, sys, torch, gymnasium as gym
from stable_baselines3 import PPO
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from curriculum_lift_cfg import FrankaLiftStage1Cfg

CKPT = sys.argv[1] if len(sys.argv) > 1 else "week3/checkpoints/stage1_final"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "motion.json"

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = 512
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = Sb3VecEnvWrapper(env)

model = PPO.load(CKPT, env=vec)

frames = []
obs = vec.reset()
for step in range(250):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec.step(action)
    frames.append({
        "step": step,
        "joints": raw.scene["robot"].data.joint_pos[0].cpu().tolist(),
        "cube":   raw.scene["object"].data.root_pos_w[0].cpu().tolist(),
        "ee":     raw.scene["ee_frame"].data.target_pos_w[0, 0, :].cpu().tolist(),
        "base":   raw.scene["robot"].data.root_pos_w[0].cpu().tolist(),
        "done":   bool(done[0]),
    })
    if bool(done[0]):
        break

# drop the reset frame itself: its cube position is the next episode's spawn
if frames and frames[-1]["done"]:
    frames.pop()

with open(OUT, "w") as f:
    json.dump(frames, f)

with open(OUT.replace(".json", "_summary.txt"), "w") as f:
    z = [fr["cube"][2] for fr in frames]
    gap = [sum((fr["ee"][i] - fr["cube"][i]) ** 2 for i in range(3)) ** 0.5 for fr in frames]
    grip = [(fr["joints"][7] + fr["joints"][8]) / 2 for fr in frames]
    shut = next((i for i, g in enumerate(grip) if g < 0.005), None)
    f.write(f"checkpoint {CKPT}\nframes {len(frames)}\n")
    f.write(f"cube z: start {z[0]:.4f} min {min(z):.4f} max {max(z):.4f} end {z[-1]:.4f}\n")
    f.write(f"gripper-cube distance: start {gap[0]:.4f} min {min(gap):.4f} (frame {gap.index(min(gap))}) end {gap[-1]:.4f}\n")
    if shut is not None:
        f.write(f"fingers first shut at frame {shut}, gripper-cube distance there {gap[shut]:.4f} m\n")
    else:
        f.write("fingers never fully shut\n")

vec.close()
simulation_app.close()
