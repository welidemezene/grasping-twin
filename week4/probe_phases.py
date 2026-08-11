"""Where does the time go, and where does the shake live?

probe_smoothness.py answered "how much does the arm shake" with one number per
policy for a whole episode. That number cannot see what the footage shows: the
hand arrives at the cube, hesitates, closes, and appears to re-grip before the
lift. A single median over 250 steps averages the hesitation together with the
approach and the carry, so it cannot separate them.

This cuts each episode into phases and reports them per env:

    APPROACH   step 0 .. first step the hand is within GRASP_GAP of the cube
    DWELL      that arrival .. first step the grasp condition is true
    LIFT_LAG   first grasp .. first step the cube is both grasped and airborne
    CARRY      everything after

DWELL is the hesitation. LIFT_LAG is "it has the cube and has not committed to
going up yet". Both are invisible in an episode-wide median.

RE-GRIPS are counted as the number of times the grasp condition goes true, then
false, then true again within one episode. That is the "second time" a viewer
sees. A clean grasp is one rising edge; three is a policy that keeps dropping
and recatching the cube.

The grasp and airborne definitions are COPIED from eval_shift.py rather than
reinvented:

    grasp    = |finger_sum - 0.042| < 0.012  AND  gap < 0.03
    airborne = lowest cube corner > 0.005

so every number here is comparable to the sweeps. Reinventing the ruler is how
this project once compared a 4.6 mm approach against a 30 mm one and called them
the same thing.

FIRST EPISODE ONLY. The sweeps run 250 steps and accumulate "ever" flags, so a
mid-run reset costs them nothing. Phase TIMINGS are destroyed by a reset -- the
clock would restart while the counters kept going -- so each env is frozen at
its first done and everything after is ignored. Envs that never grasp are
reported separately instead of being folded in as zeros.

    probe_phases.py <checkpoint> <label>

Medians, not means: one env that flings the cube moves a mean and says nothing
about what the motion looks like for most of an episode.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("label", type=str)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--hold", type=int, default=5)
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

from stage20_cfg import FrankaLiftStage20Cfg

HALF = 0.0210        # cube half-width, same as eval_shift.py
AIRBORNE = 0.005     # lowest corner above this = genuinely off the table
GRASP_GAP = 0.03     # hand is "at the cube" inside this


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


env_cfg = FrankaLiftStage20Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw = env.unwrapped
vec = StickyGripper(Sb3VecEnvWrapper(env), hold=args.hold)
model = PPO.load(args.checkpoint, env=vec)

N = raw.num_envs
T = args.steps

# per-step history, [T, N]
hist_grasp = np.zeros((T, N), dtype=bool)
hist_air = np.zeros((T, N), dtype=bool)
hist_near = np.zeros((T, N), dtype=bool)
hist_delta = np.full((T, N), np.nan, dtype=np.float64)
hist_valid = np.zeros((T, N), dtype=bool)   # False once an env has reset

alive = np.ones(N, dtype=bool)
obs = vec.reset()
prev_action = None

for t in range(T):
    action, _ = model.predict(obs, deterministic=True)
    action = np.asarray(action)
    if prev_action is not None:
        hist_delta[t] = np.abs(action[:, :-1] - prev_action[:, :-1]).mean(axis=1)
    prev_action = action.copy()

    obs, rew, done, info = vec.step(action)

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
    lowest = (pos[:, 2] - drop).cpu().numpy()

    fsum = robot.data.joint_pos[:, -2:].sum(dim=1).cpu().numpy()
    gap = torch.norm(pos - eef.data.target_pos_w[:, 0, :], dim=1).cpu().numpy()

    hist_valid[t] = alive
    hist_near[t] = gap < GRASP_GAP
    hist_grasp[t] = (np.abs(fsum - 0.042) < 0.012) & (gap < GRASP_GAP)
    hist_air[t] = lowest > AIRBORNE

    # freeze this env AFTER recording the step it finished on
    alive &= ~np.asarray(done, dtype=bool)

vec.close()


def first_true(mask):
    """Index of the first True per column, or -1 if never."""
    any_true = mask.any(axis=0)
    idx = mask.argmax(axis=0)
    return np.where(any_true, idx, -1)


valid = hist_valid
g = hist_grasp & valid
a = hist_air & valid
near = hist_near & valid

t_arrive = first_true(near)
t_grasp = first_true(g)
t_air = first_true(g & a)

# rising edges of the grasp signal: how many separate times it took hold
prev = np.vstack([np.zeros((1, N), dtype=bool), g[:-1]])
rising = (g & ~prev).sum(axis=0)

ok = (t_grasp >= 0) & (t_arrive >= 0)
lifted = ok & (t_air >= 0)

dwell = (t_grasp - t_arrive)[ok]
lift_lag = (t_air - t_grasp)[lifted]
regrips = rising[ok]


def phase_delta(lo, hi):
    """Median arm action delta over [lo, hi) per env, pooled across envs."""
    out = []
    for i in np.where(ok)[0]:
        s, e = lo[i], hi[i]
        if e > s >= 0:
            seg = hist_delta[s:e, i]
            seg = seg[~np.isnan(seg)]
            if len(seg):
                out.append(seg)
    return np.concatenate(out) if out else np.array([np.nan])


zero = np.zeros(N, dtype=int)
d_approach = phase_delta(zero, np.maximum(t_arrive, 0))
d_dwell = phase_delta(np.maximum(t_arrive, 0), np.maximum(t_grasp, 0))
d_carry = phase_delta(np.maximum(np.where(t_air >= 0, t_air, t_grasp), 0),
                      np.full(N, T, dtype=int))


def med(v):
    return float(np.median(v)) if len(v) else float("nan")


lines = [
    "PHASE PROBE -- %s" % args.label,
    "  checkpoint %s" % args.checkpoint,
    "  %d envs x %d steps, deterministic, first episode only" % (N, T),
    "  grasp/airborne definitions copied from eval_shift.py",
    "",
    "  envs that reached the cube      %4d / %d" % (int((t_arrive >= 0).sum()), N),
    "  envs that grasped               %4d / %d" % (int(ok.sum()), N),
    "  envs that grasped AND lifted    %4d / %d" % (int(lifted.sum()), N),
    "",
    "  APPROACH  steps to reach the cube   median %6.1f" % med(t_arrive[t_arrive >= 0]),
    "  DWELL     arrival -> first grasp    median %6.1f   p90 %6.1f   max %6.0f"
    % (med(dwell), np.percentile(dwell, 90) if len(dwell) else float("nan"),
       dwell.max() if len(dwell) else float("nan")),
    "  LIFT_LAG  first grasp -> airborne   median %6.1f   p90 %6.1f   max %6.0f"
    % (med(lift_lag), np.percentile(lift_lag, 90) if len(lift_lag) else float("nan"),
       lift_lag.max() if len(lift_lag) else float("nan")),
    "",
    "  RE-GRIPS  separate times the grasp condition took hold",
    "     median %.1f   mean %.2f   max %d" % (
        med(regrips), float(regrips.mean()) if len(regrips) else float("nan"),
        int(regrips.max()) if len(regrips) else 0),
    "     one rising edge = clean grasp; more = it dropped and recaught",
    "     envs with exactly 1: %d / %d  (%.1f%%)" % (
        int((regrips == 1).sum()), len(regrips),
        100.0 * float((regrips == 1).sum()) / max(len(regrips), 1)),
    "",
    "  ARM ACTION DELTA BY PHASE (same statistic probe_smoothness.py reports",
    "  once for the whole episode -- if the shake is uniform these match)",
    "     approach  median %.4f" % med(d_approach),
    "     dwell     median %.4f" % med(d_dwell),
    "     carry     median %.4f" % med(d_carry),
    "",
    "  Steps, not seconds: divide by the control rate for wall time. A dwell of",
    "  0 means the grasp condition was true on the step the hand arrived, which",
    "  is the clean case.",
]
out = "phases_%s.txt" % args.label
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

simulation_app.close()
