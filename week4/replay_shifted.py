"""Film one policy at ONE FIXED cube offset -- the shot the grid footage cannot give.

The week 4 result is that stage 16 memorised a trajectory to a single point and
stage 22 reads the cube's actual position. Neither existing video shows it:

  - stage22_grid25.mp4 uses the RANDOMIZED spawn, +-5 cm on a cube 4.2 cm wide,
    across 25 arms at once. Each cube moves at most about two cube-widths in a
    wide shot. A viewer cannot see the displacement, and every arm succeeds, so
    there is no failure in frame to compare against.
  - the number that matters -- 41.0% vs 93.2% at the 5 cm diagonal -- is measured
    by eval_shift.py, which puts ALL 512 envs at the SAME offset. That is exactly
    what the randomized render never shows.

So this films FrankaLiftShiftedCfg: every environment gets the same fixed shift,
the one the sweep scored. Run it twice at the same offset and seed, once per
checkpoint, and the pair is the demonstration. stage 16 closes on air where the
cube used to be; stage 22 goes to where it is.

DEFAULT OFFSET is the 5 cm diagonal (x +0.035, y +0.035), chosen from s16/s22
sweep rows rather than for looks:

    offset            stage 16   stage 22
    +x 4 cm             41.8%      87.7%
    5 cm diagonal       41.0%      93.2%     <- widest honest gap
    7 cm diagonal        0.0%      62.7%

The 7 cm diagonal is the most dramatic stage 16 number (0 of 512) but stage 22
fails 37% of the time there too, so a grid of it shows both policies dropping
cubes. The 5 cm diagonal is where one policy nearly always fails and the other
nearly always works, which is the honest version of the same story.

This is a RENDER, not a scoring run. The on-screen outcome is an illustration of
s16_baseline_sweep.csv and s22_sweep.csv; those files remain the evidence, and a
9-env sample cannot restate a 512-trial number. The per-env outcome is printed
anyway so the clip can never be described as something the numbers do not say.

Usage, Windows, from the Isaac Lab 2.1 venv (isaaclab.bat -p does NOT resolve
isaaclab from a non-interactive shell -- call the venv python directly, and set
OMNI_KIT_ACCEPT_EULA=YES or Kit blocks on the prompt and dies with EOFError):

    C:\\isaac\\venv310\\Scripts\\python.exe C:\\isaac\\replay_shifted.py ^
        <ckpt> <out.mp4> <hold> <envs> <steps> <seed> <shift_x> <shift_y>
"""
import sys
import math

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import gymnasium as gym
import imageio
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

sys.path.insert(0, r"C:\isaac")
from stage20_cfg import FrankaLiftShiftedCfg

CKPT = sys.argv[1] if len(sys.argv) > 1 else r"C:\isaac\stage22_final.zip"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:\isaac\shifted.mp4"
HOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 5
NUM_ENVS = int(sys.argv[4]) if len(sys.argv) > 4 else 9
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 250
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
SHIFT_X = float(sys.argv[7]) if len(sys.argv) > 7 else 0.035
SHIFT_Y = float(sys.argv[8]) if len(sys.argv) > 8 else 0.035

HALF = 0.0210        # cube half-width, as eval_shift.py
AIRBORNE = 0.005


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


env_cfg = FrankaLiftShiftedCfg()
env_cfg.shift_x = SHIFT_X
env_cfg.shift_y = SHIFT_Y
env_cfg.__post_init__()          # re-apply now that the shift is set
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
print("FIXED offset x %+.3f y %+.3f (%.3f m) -- every env identical, as the sweep"
      % (SHIFT_X, SHIFT_Y, math.hypot(SHIFT_X, SHIFT_Y)), flush=True)

writer = imageio.get_writer(OUT, fps=30, quality=8)

obs = vec.reset()
N = raw.num_envs

# Confirm the shift actually landed. A render that silently fell back to the
# home spawn would look like a result and be nothing.
spawn = raw.scene["object"].data.root_pos_w[:, :2].clone()
rel = (spawn - raw.scene.env_origins[:, :2]).cpu().numpy()

ever_grasped = torch.zeros(N, dtype=torch.bool)
ever_both = torch.zeros(N, dtype=torch.bool)

for step in range(STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec.step(action)
    frame = env.render()
    if frame is not None:
        writer.append_data(np.asarray(frame))

    cube = raw.scene["object"]
    robot = raw.scene["robot"]
    eef = raw.scene["ee_frame"]
    pos = cube.data.root_pos_w
    quat = cube.data.root_quat_w
    w, x, y, z_ = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    r20 = 2 * (x * z_ - w * y)
    r21 = 2 * (y * z_ + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    drop = (r20.abs() + r21.abs() + r22.abs()) * HALF
    lowest = (pos[:, 2] - drop).cpu()
    fsum = robot.data.joint_pos[:, -2:].sum(dim=1).cpu()
    gap = torch.norm(pos - eef.data.target_pos_w[:, 0, :], dim=1).cpu()
    g = ((fsum - 0.042).abs() < 0.012) & (gap < 0.03)
    ever_grasped |= g
    ever_both |= (g & (lowest > AIRBORNE))

writer.close()

print("REPLAY DONE -> %s" % OUT, flush=True)
print("cube spawn, env-relative: x %.3f..%.3f  y %.3f..%.3f  (all envs should match)"
      % (rel[:, 0].min(), rel[:, 0].max(), rel[:, 1].min(), rel[:, 1].max()), flush=True)
print("ON-SCREEN OUTCOME: grasped %d/%d, grasped AND airborne %d/%d"
      % (int(ever_grasped.sum()), N, int(ever_both.sum()), N), flush=True)
print("  (illustration of the 512-trial sweep, not a substitute for it)", flush=True)

vec.close()
app.close()
