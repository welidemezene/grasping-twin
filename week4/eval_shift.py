"""Does the policy survive the cube moving? 512 trials per offset, corner geometry.

This is week 4's instrument, and it is deliberately written BEFORE stage 20 runs
so the bar cannot be quietly moved afterwards.

It is check_airborne.py's measurement -- the height of the cube's LOWEST CORNER
computed from its full pose, across all 512 environments at once -- swept over a
list of FIXED spawn offsets. Two things are load-bearing:

  1. THE OFFSET IS THE SAME IN ALL 512 ENVS for a given evaluation. If the offset
     were randomized per env, the 512 trials would average over easy and hard
     placements and the success rate would not correspond to any particular
     displacement. Each offset gets its own 512-trial verdict.

  2. THE OFFSETS ARE HELD OUT. Stage 20 trains on a continuous +-5 cm range, so
     no single offset is literally unseen; what matters is that these specific
     values are not the fixed point the policy spent week 3 memorising, and that
     the sweep includes 0.034 (the diagonal that broke stage16_final on Isaac Lab
     3.0) and offsets at and beyond the training edge. The 0.00 row is the
     control: it must not regress.

WHY CORNER HEIGHT AND NOT CUBE HEIGHT. A 4.2 cm cube resting flat has its centre
at 0.0210; tipped onto a corner that becomes 0.0210*sqrt(3) = 0.0364, which
clears a 0.035 height gate while the cube is still on the table. This project
declared success off cube height twice and was wrong both times.

USE MEDIANS, NOT MEANS. grip_geometry.py found a single flung frame at 340 mm
moving a mean by 1.5 mm.

Usage:
    python eval_shift.py <checkpoint> [out_prefix] [hold]

Runs one offset per simulation, in one process, by rebuilding... it does NOT.
Isaac Lab's simulation context is a singleton and hangs if a second scene is
built in the same process -- eval_stage19.sh learned that. So this script
evaluates ONE offset, taken from --shift_x/--shift_y, and sweep_shift.sh calls
it once per offset in its own container.
"""

import argparse
import csv
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--out_prefix", default="shift")
parser.add_argument("--hold", type=int, default=5,
                    help="MUST match training (5). Judging stages 11-13 at hold 1 "
                         "while they trained at hold 5 invalidated those verdicts.")
parser.add_argument("--shift_x", type=float, default=0.0)
parser.add_argument("--shift_y", type=float, default=0.0)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=250)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from stage20_cfg import FrankaLiftShiftedCfg

HALF = 0.021          # half the cube edge
AIRBORNE = 0.005      # lowest corner above this = genuinely off the table


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
        obs, r, d, i = self.venv.step_wait()
        if np.any(d):
            self.count[d] = 0
            self.held[d] = 1.0
        return obs, r, d, i

    def reset(self):
        self.count[:] = 0
        self.held[:] = 1.0
        return self.venv.reset()


cfg = FrankaLiftShiftedCfg()
cfg.shift_x = args.shift_x
cfg.shift_y = args.shift_y
cfg.__post_init__()          # re-apply now that the shift is set
cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=cfg)
raw = env.unwrapped
vec = StickyGripper(Sb3VecEnvWrapper(env), hold=args.hold)
model = PPO.load(args.checkpoint, env=vec)

N = raw.num_envs
ever_grasped = torch.zeros(N, dtype=torch.bool)
ever_airborne = torch.zeros(N, dtype=torch.bool)
ever_both = torch.zeros(N, dtype=torch.bool)
best_corner = torch.full((N,), -1.0)
best_tilt = torch.zeros(N)

obs = vec.reset()
for step in range(args.steps):
    a, _ = model.predict(obs, deterministic=True)
    obs, rew, done, info = vec.step(a)

    cube = raw.scene["object"]
    robot = raw.scene["robot"]
    eef = raw.scene["ee_frame"]

    pos = cube.data.root_pos_w
    quat = cube.data.root_quat_w
    w, x, y, z_ = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # third row of the rotation matrix is all the lowest corner needs
    r20 = 2 * (x * z_ - w * y)
    r21 = 2 * (y * z_ + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    drop = (r20.abs() + r21.abs() + r22.abs()) * HALF
    lowest = (pos[:, 2] - drop).cpu()

    fsum = robot.data.joint_pos[:, -2:].sum(dim=1).cpu()
    gap = torch.norm(pos - eef.data.target_pos_w[:, 0, :], dim=1).cpu()

    g = ((fsum - 0.042).abs() < 0.012) & (gap < 0.03)
    air = lowest > AIRBORNE
    ever_grasped |= g
    ever_airborne |= air
    ever_both |= (g & air)
    best_corner = torch.maximum(best_corner, torch.where(g, lowest, torch.tensor(-1.0)))

    tilt = torch.rad2deg(torch.acos(r22.clamp(-1.0, 1.0))).cpu()
    tilt = torch.minimum(tilt, 180.0 - tilt)
    best_tilt = torch.maximum(best_tilt, torch.where(g & air, tilt, torch.zeros(N)))

    if bool(done[0]):
        break

held = best_corner[best_corner > 0]
tilts = best_tilt[best_tilt > 0]
success = int(ever_both.sum())
rate = 100.0 * success / N
lift_median = float(held.median()) if len(held) else 0.0
tilt_median = float(tilts.median()) if len(tilts) else 0.0

# Week 4's gate, fixed in advance.
PASS = rate > 90.0 and lift_median > 0.10

shift_mag = math.hypot(args.shift_x, args.shift_y)
lines = [
    "=== SHIFT %.3f m  (x %+.3f, y %+.3f) — %d parallel trials ===" % (
        shift_mag, args.shift_x, args.shift_y, N),
    "checkpoint %s   hold %d" % (args.checkpoint, args.hold),
    "  ever grasped:           %4d / %d  (%.1f%%)" % (
        int(ever_grasped.sum()), N, 100.0 * float(ever_grasped.sum()) / N),
    "  ever airborne:          %4d / %d  (%.1f%%)" % (
        int(ever_airborne.sum()), N, 100.0 * float(ever_airborne.sum()) / N),
    "  GRASPED AND AIRBORNE:   %4d / %d  (%.1f%%)   <- the real success rate" % (
        success, N, rate),
    "  lift while held:        median %.4f m, max %.4f m" % (
        lift_median, float(held.max()) if len(held) else 0.0),
    "  cube tilt while held:   median %.1f deg" % tilt_median,
    "  GATE (>90%% and >0.10 m): %s" % ("PASS" if PASS else "FAIL"),
    "",
    "  reference — stage16_final at shift 0.000: 99.6%%, 0.1349 m, ~40 deg",
]
print("\n".join(lines))

with open("%s_summary.txt" % args.out_prefix, "a") as f:
    f.write("\n".join(lines) + "\n\n")

# One row per offset, appended, so the sweep builds a single comparable table.
row = {
    "checkpoint": os.path.basename(args.checkpoint),
    "shift_x": round(args.shift_x, 4),
    "shift_y": round(args.shift_y, 4),
    "shift_mag": round(shift_mag, 4),
    "grasped": int(ever_grasped.sum()),
    "airborne": int(ever_airborne.sum()),
    "grasped_and_airborne": success,
    "trials": N,
    "success_pct": round(rate, 1),
    "lift_median_m": round(lift_median, 4),
    "tilt_median_deg": round(tilt_median, 1),
    "pass": int(PASS),
}
csv_path = "%s_sweep.csv" % args.out_prefix
new = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(row.keys()))
    if new:
        wr.writeheader()
    wr.writerow(row)

env.close()
simulation_app.close()
