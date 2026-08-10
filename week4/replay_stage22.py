"""Film stage22_final on Isaac Sim 4.5 + Isaac Lab 2.1 -> MP4.

Mirrors week3/replay_stage16_45.py, with ONE difference that is the whole point
of the shot: the config is FrankaLiftStage20Cfg, so the cube spawns anywhere in
a +-5 cm box instead of at the fixed point every stage 1-19 used.

That is what makes this footage worth more than week 3's. The nine-robot grid
there was nine arms performing the identical memorised motion, which a viewer
cannot distinguish from a replayed animation. Here every environment draws its
own cube position, so every arm reaches somewhere different and they all still
lift -- a thing an animation cannot fake.

The reward config is irrelevant to a render (the policy is replayed
deterministically and nothing is being scored), so stage 20's config is used
rather than stage 22's; they differ only in the reward gate. The SCENE, the
spawn distribution, the observation space and the action space are identical.

Usage, from the IsaacLab 2.1 launcher:
    isaaclab.bat -p C:\\isaac\\replay_stage22.py <ckpt> <out.mp4> <hold> <envs> <steps> <seed>
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
import math
import numpy as np
import gymnasium as gym
import imageio
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

sys.path.insert(0, r"C:\isaac")
from stage20_cfg import FrankaLiftStage20Cfg, SPAWN_RANGE

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\default.LAPTOP-OBNFH8RI\grasping_twin\week4\checkpoints\stage22_final.zip"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:\isaac\stage22_hero.mp4"
HOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 5
NUM_ENVS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 250
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0


class StickyGripper(VecEnvWrapper):
    """The 5-frame gripper hold the policy trained under. Rendering without it
    would show a different action regime than the one every number describes --
    the mismatch that silently invalidated the stage 11-13 verdicts."""

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


env_cfg = FrankaLiftStage20Cfg()
env_cfg.scene.num_envs = NUM_ENVS
env_cfg.sim.device = "cuda:0"
env_cfg.seed = SEED

if NUM_ENVS == 1:
    env_cfg.viewer.eye = (1.4, -1.2, 0.9)
    env_cfg.viewer.lookat = (0.45, 0.0, 0.25)
else:
    half = math.sqrt(NUM_ENVS) * 2.5 / 2
    env_cfg.viewer.eye = (half * 1.7, -half * 1.7, half * 1.1)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.3)

# Debug visualisations off: goal-pose arrows and ee-frame markers read as UI
# clutter in footage and imply the robot is being told where to go.
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
print("PPO loaded OK: %s" % CKPT, flush=True)
print("spawn range +-%.3f m -- every env draws its own cube position" % SPAWN_RANGE, flush=True)

writer = imageio.get_writer(OUT, fps=30, quality=8)

obs = vec.reset()

# Record where each env's cube actually spawned, so the claim "they are in
# different places" is measured and not just asserted about a video.
spawn_xy = raw.scene["object"].data.root_pos_w[:, :2].clone()
origins = raw.scene.env_origins[:, :2]
rel = (spawn_xy - origins).cpu().numpy()

cube_z, fsum, episodes = [], [], 0
for step in range(STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec.step(action)
    frame = env.render()
    if frame is not None:
        writer.append_data(np.asarray(frame))
    jp = raw.scene["robot"].data.joint_pos[0]
    cube_z.append(float(raw.scene["object"].data.root_pos_w[0, 2]))
    fsum.append(float(jp[7] + jp[8]))
    if bool(done[0]):
        episodes += 1

writer.close()

print("REPLAY DONE: %d frames -> %s" % (len(cube_z), OUT), flush=True)
print("episodes completed (env 0): %d" % episodes, flush=True)
print("cube z (env 0): start %.4f  max %.4f  end %.4f" % (cube_z[0], max(cube_z), cube_z[-1]), flush=True)
print("finger sum (env 0): min %.4f" % min(fsum), flush=True)
print("cube spawn spread across envs: x %.3f..%.3f  y %.3f..%.3f (metres, env-relative)"
      % (rel[:, 0].min(), rel[:, 0].max(), rel[:, 1].min(), rel[:, 1].max()), flush=True)

vec.close()
app.close()
