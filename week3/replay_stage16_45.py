"""Replay stage16_final on Isaac Sim 4.5 + Isaac Lab 2.1 (EXACT training versions) -> MP4.
Mirrors week3/record_motion_sticky.py: FrankaLiftStage1Cfg, sticky hold=5, deterministic.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import sys
import numpy as np
import gymnasium as gym
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

sys.path.insert(0, r"C:\isaac")
from curriculum_lift_cfg import FrankaLiftStage1Cfg

CKPT = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\default.LAPTOP-OBNFH8RI\grasping_twin\week3\week3\checkpoints\stage16_final.zip"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:\isaac\stage16_lift_45.mp4"
HOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 5


class StickyGripper(VecEnvWrapper):
    def __init__(self, venv, hold=5):
        super().__init__(venv)
        self.hold = hold
        self.count = np.zeros(venv.num_envs, dtype=np.int64)
        self.held = np.ones(venv.num_envs, dtype=np.float32)

    def step_async(self, actions):
        actions = np.array(actions, copy=True)
        refresh = self.count % self.hold == 0
        self.held = np.where(refresh, actions[:, -1], self.held).astype(np.float32)
        actions[:, -1] = self.held
        self.count += 1
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if np.any(dones):
            self.count[dones] = 0
            self.held[dones] = 1.0
        return obs, rewards, dones, infos

    def reset(self):
        self.count[:] = 0
        self.held[:] = 1.0
        return self.venv.reset()


NUM_ENVS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 250
env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = NUM_ENVS
env_cfg.sim.device = "cuda:0"
if NUM_ENVS == 1:
    env_cfg.viewer.eye = (1.4, -1.2, 0.9)
    env_cfg.viewer.lookat = (0.45, 0.0, 0.25)
else:
    # pulled-back shot over the env grid (spacing ~2.5 m, grid centered at origin)
    import math
    half = math.sqrt(NUM_ENVS) * 2.5 / 2
    env_cfg.viewer.eye = (half * 1.7, -half * 1.7, half * 1.1)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
# hide debug visualizations (goal-pose arrows, ee-frame markers) for clean footage
for _term in ("commands", "scene"):
    pass
try:
    env_cfg.commands.object_pose.debug_vis = False
except AttributeError:
    pass
try:
    env_cfg.scene.ee_frame.debug_vis = False
except AttributeError:
    pass

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg, render_mode="rgb_array")
raw = env.unwrapped
vec = StickyGripper(Sb3VecEnvWrapper(env), hold=HOLD)

model = PPO.load(CKPT, env=vec)
print("PPO loaded OK", flush=True)

writer = imageio.get_writer(OUT, fps=30, quality=8)
cube_z, fsum = [], []
obs = vec.reset()
episodes = 0
for step in range(STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec.step(action)
    frame = env.render()
    if frame is not None:
        writer.append_data(np.asarray(frame))
    jp = raw.scene["robot"].data.joint_pos[0]
    cube_z.append(float(raw.scene["object"].data.root_pos_w[0, 2]))
    fsum.append(float(jp[7] + jp[8]))
    # keep filming through episode resets: each reset is a fresh cube + new lift
    if bool(done[0]):
        episodes += 1
print(f"episodes completed (env 0): {episodes}", flush=True)
writer.close()

print(f"REPLAY DONE: {len(cube_z)} frames -> {OUT}", flush=True)
print(f"cube z: start {cube_z[0]:.4f} max {max(cube_z):.4f} end {cube_z[-1]:.4f}", flush=True)
print(f"finger sum: min {min(fsum):.4f}", flush=True)
grasp = [i for i in range(len(cube_z)) if abs(fsum[i] - 0.042) < 0.012]
print(f"frames with fingers on cube-width: {len(grasp)}", flush=True)

vec.close()
app.close()
