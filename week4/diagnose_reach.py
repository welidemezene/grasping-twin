"""Why does +x 4 cm fail when -y 4 cm does not? Reach failure or perception failure?

THE QUESTION THIS ANSWERS, AND WHY IT COMES BEFORE STAGE 20.

The baseline sweep (s16_baseline_sweep.csv) found that stage16_final's
brittleness is DIRECTIONAL, not radial. Identical 4 cm displacements give:

    0.040 -y  ->  99.6%      0.040 +x  ->  41.8%

A policy that had simply memorised one point would fail the same way in every
direction. This one does not, so "never learned to see" is too crude an
explanation and the sweep cannot refine it. Two candidates remain, and they
have OPPOSITE fixes:

  PERCEPTION failure -- the arm drives to where the cube used to be and closes
      on air. The cube's position is in the observation but the policy learned
      to ignore it, because for 19 stages it never varied. Domain randomization
      (stage 20) fixes exactly this.

  REACH failure -- the arm drives to the right place and cannot get there, or
      arrives in a wrist configuration that cannot close. +x may simply be
      further into the arm's workspace than -y. No amount of randomized
      training creates reach that the kinematics do not have. Stage 20 would
      burn 45 minutes and come back with a number that means nothing.

The sweep reports only the outcome, which is identical in both cases: no lift.
This script reports the CAUSE, by watching where the end-effector actually goes.

THE DISCRIMINATOR. At every step, in all 512 envs, measure two distances:

    d_cube  = |EE - cube|                 where the cube actually is
    d_ghost = |EE - (cube - shift)|       where the cube sat for all of week 3

Perception failure: d_ghost collapses toward zero and d_cube plateaus near the
shift magnitude. The arm is confidently, precisely wrong -- it is running week
3's trajectory.

Reach failure: d_cube falls (it IS tracking the cube) but stalls above the
grasp band, while d_ghost stays large. The arm is trying and cannot arrive.
Arm joints pinned near their limits at the stall corroborate this.

Neither: d_cube reaches the grasp band and the failure is downstream -- closure
or the lift itself -- which is a third answer and also worth knowing.

RUN IT ON BOTH DIRECTIONS. A single +x run is not interpretable on its own;
this project has concluded from one-armed comparisons four times now and been
wrong every time. The -y 4 cm run is the control: same displacement magnitude,
99.6% success. Whatever metric explains the difference between those two runs
is the answer. sweep_diagnose.sh runs the matched pair plus the unshifted
control.

MEDIANS, NOT MEANS -- grip_geometry.py found one flung frame moving a mean by
1.5 mm. Every aggregate here is a median over the 512 envs.

Usage:
    python diagnose_reach.py <checkpoint> --shift_x 0.040 [--out_prefix diag]
"""

import argparse
import csv
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--out_prefix", default="diag")
parser.add_argument("--hold", type=int, default=5,
                    help="MUST match training (5), same reason as eval_shift.py.")
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

HALF = 0.021
AIRBORNE = 0.005
GRASP_BAND = 0.03      # same gap threshold eval_shift.py calls "at the cube"


class StickyGripper(VecEnvWrapper):
    """Identical to eval_shift.py's. Judging a policy in a different action
    regime than it trained in invalidated stages 11-13; do not drop this."""

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
cfg.__post_init__()
cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=cfg)
raw = env.unwrapped
vec = StickyGripper(Sb3VecEnvWrapper(env), hold=args.hold)
model = PPO.load(args.checkpoint, env=vec)

N = raw.num_envs
robot = raw.scene["robot"]

# The ghost is the spawn point every stage 1-19 trained against: the cube's
# current position minus the offset we just applied. Computed per env from the
# live pose rather than hardcoded, so it stays correct if the base cfg moves.
shift_vec = torch.tensor([args.shift_x, args.shift_y, 0.0], device=robot.data.joint_pos.device)

# Franka arm joints are the first 7; the last 2 are the fingers. Saturation of
# the arm joints at the stall is what separates "cannot reach" from "chose not
# to go" -- a policy that simply never commands the motion leaves its joints
# nowhere near a limit.
arm_lo = robot.data.soft_joint_pos_limits[0, :7, 0]
arm_hi = robot.data.soft_joint_pos_limits[0, :7, 1]
arm_span = (arm_hi - arm_lo).clamp(min=1e-6)

trace = []               # one row per step, medians over the 512 envs
min_d_cube = torch.full((N,), 1e9)
min_d_ghost = torch.full((N,), 1e9)
ever_at_cube = torch.zeros(N, dtype=torch.bool)
ever_at_ghost = torch.zeros(N, dtype=torch.bool)
ever_grasped = torch.zeros(N, dtype=torch.bool)
ever_both = torch.zeros(N, dtype=torch.bool)

obs = vec.reset()
for step in range(args.steps):
    a, _ = model.predict(obs, deterministic=True)
    obs, rew, done, info = vec.step(a)

    cube = raw.scene["object"]
    eef = raw.scene["ee_frame"]

    pos = cube.data.root_pos_w
    ee = eef.data.target_pos_w[:, 0, :]
    ghost = pos - shift_vec

    d_cube = torch.norm(pos - ee, dim=1).cpu()
    d_ghost = torch.norm(ghost - ee, dim=1).cpu()

    min_d_cube = torch.minimum(min_d_cube, d_cube)
    min_d_ghost = torch.minimum(min_d_ghost, d_ghost)
    ever_at_cube |= d_cube < GRASP_BAND
    ever_at_ghost |= d_ghost < GRASP_BAND

    # corner height, exactly as eval_shift.py / check_airborne.py compute it
    quat = cube.data.root_quat_w
    w, x, y, z_ = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    r20 = 2 * (x * z_ - w * y)
    r21 = 2 * (y * z_ + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    lowest = (pos[:, 2] - (r20.abs() + r21.abs() + r22.abs()) * HALF).cpu()

    fsum = robot.data.joint_pos[:, -2:].sum(dim=1).cpu()
    g = ((fsum - 0.042).abs() < 0.012) & (d_cube < GRASP_BAND)
    ever_grasped |= g
    ever_both |= (g & (lowest > AIRBORNE))

    # fraction of the arm's 7 joints sitting in the outer 5% of their range
    q = robot.data.joint_pos[:, :7]
    frac = ((q - arm_lo) / arm_span).clamp(0.0, 1.0)
    pinned = ((frac < 0.05) | (frac > 0.95)).float().sum(dim=1).cpu()

    trace.append({
        "step": step,
        "d_cube_median": round(float(d_cube.median()), 5),
        "d_ghost_median": round(float(d_ghost.median()), 5),
        "finger_sum_median": round(float(fsum.median()), 5),
        "cube_corner_median": round(float(lowest.median()), 5),
        "arm_joints_pinned_median": round(float(pinned.median()), 2),
        "grasping_pct": round(100.0 * float(g.float().mean()), 1),
    })

    if bool(done[0]):
        break

shift_mag = math.hypot(args.shift_x, args.shift_y)
md_cube = float(min_d_cube.median())
md_ghost = float(min_d_ghost.median())
at_cube_pct = 100.0 * float(ever_at_cube.float().mean())
at_ghost_pct = 100.0 * float(ever_at_ghost.float().mean())
success_pct = 100.0 * float(ever_both.float().mean())

# Verdict. Ordered so the cheapest explanation is tested first, and worded so a
# genuinely ambiguous run says so instead of picking a side. This decides
# whether stage 20 is worth 45 minutes -- it should not be able to hedge.
if at_cube_pct > 90.0 and success_pct < 60.0:
    verdict = ("DOWNSTREAM — the arm ARRIVES at the cube (%.0f%% of envs reach "
               "within %.0f mm) and the failure is closure or lift, not "
               "getting there. Randomizing the spawn is not the fix."
               % (at_cube_pct, GRASP_BAND * 1000))
elif md_ghost < md_cube and md_cube > GRASP_BAND:
    verdict = ("PERCEPTION — the end-effector goes to the OLD spawn point "
               "(median closest approach to ghost %.4f m vs %.4f m to the real "
               "cube). The policy is replaying week 3's trajectory and ignoring "
               "the cube in its observation. Stage 20 is the right fix."
               % (md_ghost, md_cube))
elif md_cube < md_ghost and md_cube > GRASP_BAND:
    verdict = ("REACH — the arm TRACKS the cube (closest approach %.4f m, "
               "nearer than the ghost's %.4f m) but stalls %.1f mm short of "
               "the %.0f mm grasp band. Check arm_joints_pinned in the trace: "
               "if joints sit at their limits this is kinematic and NO amount "
               "of randomized training fixes it."
               % (md_cube, md_ghost, (md_cube - GRASP_BAND) * 1000, GRASP_BAND * 1000))
else:
    verdict = ("AMBIGUOUS — closest approach %.4f m to cube, %.4f m to ghost. "
               "Neither explanation dominates; read the trace CSV before "
               "committing to a stage." % (md_cube, md_ghost))

lines = [
    "=== REACH/PERCEPTION DIAGNOSIS  shift %.3f m (x %+.3f, y %+.3f) — %d trials ===" % (
        shift_mag, args.shift_x, args.shift_y, N),
    "checkpoint %s   hold %d   steps %d" % (args.checkpoint, args.hold, len(trace)),
    "",
    "  closest approach to the REAL cube:   median %.4f m" % md_cube,
    "  closest approach to the GHOST spot:  median %.4f m" % md_ghost,
    "     (ghost = where the cube sat for every stage 1-19)",
    "  envs that ever reached the cube:     %5.1f%%" % at_cube_pct,
    "  envs that ever reached the ghost:    %5.1f%%" % at_ghost_pct,
    "  ever grasped:                        %5.1f%%" % (100.0 * float(ever_grasped.float().mean())),
    "  GRASPED AND AIRBORNE:                %5.1f%%" % success_pct,
    "",
    "  VERDICT: %s" % verdict,
    "",
    "  compare — s16_baseline_sweep.csv: +x 0.040 = 41.8%, -y 0.040 = 99.6%",
]
print("\n".join(lines))

with open("%s_summary.txt" % args.out_prefix, "a") as f:
    f.write("\n".join(lines) + "\n\n")

tag = "x%+.3f_y%+.3f" % (args.shift_x, args.shift_y)
with open("%s_trace_%s.csv" % (args.out_prefix, tag), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
    wr.writeheader()
    wr.writerows(trace)

env.close()
simulation_app.close()
