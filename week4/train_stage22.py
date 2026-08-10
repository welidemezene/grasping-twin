"""Stage 22 — is the lift being paid to stop by a boolean?

Stage 21 falsified the squareness-term explanation (lift stayed at 2.7 cm with
the term gone; tilt, predicted to regress to ~79 deg, came out at 19.9). The knee
probe then closed off checkpoint selection: the 3.39M checkpoint's 7.8 cm lift
scores 28.9% at +x 4 cm and lifts 1 mm there, so no checkpoint in stage 20 has
both abilities at once.

What is left is the gate's SHAPE. _held is a boolean multiplying five reward
terms simultaneously, so a cube shifting two millimetres between the fingers
sends five channels to zero together. It flickers hardest during upward
acceleration, and only once the spawn is randomized does the grip vary enough
episode to episode for that to bite -- which is the one fact the cube_upright
story never explained. See stage22_cfg.py for the full argument, the widths, and
the registered falsification condition.

WARM START FROM stage20_final, not stage16_final. Stage 21 needed a common
origin to price a reward term. This run is asking whether the lift can be
recovered ON TOP OF generalization, and stage 20 is the only policy that has it.

JUDGING IT. Two things, and the second is not optional:

  1. sweep_shift.sh at the six held-out offsets, gate >90% and >10 cm.
  2. lift_trajectory.sh across checkpoints. A stage 22 whose lift falls MORE
     SLOWLY than stage 20's has confirmed nothing. The prediction is that the
     decay STOPS, not that it softens.
"""

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str,
                    default="checkpoints/stage20_final",
                    help="generalization is the thing worth keeping -- see above")
parser.add_argument("--ent_coef", type=float, default=0.004,
                    help="stage 20's value, held fixed so the gate is the only change")
parser.add_argument("--hold", type=int, default=5)
parser.add_argument("--grip_std_floor", type=float, default=0.05)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--save_prefix", type=str, default="stage22")
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

# stage22_cfg FIRST: it is what puts week3/ on sys.path, and grasp_reward lives
# there. Importing grasp_reward above this line is a ModuleNotFoundError.
from stage22_cfg import FrankaLiftStage22Cfg, _held_smooth, NEAR_WIDTH, BUMP_WIDTH
from stage20_cfg import SPAWN_RANGE

import grasp_reward


class StickyGripper(VecEnvWrapper):
    """Stage 20's, byte for byte. The eval tooling uses the same hold; judging a
    policy in an action regime it never trained in silently invalidated the
    verdicts for stages 11 through 13."""

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


class GripperFloor(BaseCallback):
    """A FLOOR, only. clamp_(min=...) cannot pull a std down, so this has never
    been able to make the hand decisive -- stage 20 loaded at 1.0317."""

    def __init__(self, floor: float):
        super().__init__()
        self.floor = floor

    def _on_rollout_end(self) -> None:
        with torch.no_grad():
            self.model.policy.log_std.data[-1].clamp_(min=math.log(self.floor))
        self.logger.record("gripper/std", float(self.model.policy.log_std[-1].exp()))

    def _on_step(self) -> bool:
        return True


env_cfg = FrankaLiftStage22Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="logs/stage22",
    custom_objects={"ent_coef": args.ent_coef, "learning_rate": args.learning_rate},
)

# Fail loudly rather than train 45 minutes on the wrong gate. The rebind happens
# in __post_init__, and a config refactor that stopped calling it would produce a
# rerun of stage 20 wearing stage 22's name -- which the two would then be
# compared as if they differed. Stage 21 carried the same guard for its own term.
assert grasp_reward._held is _held_smooth, \
    "_held is still the boolean -- stage 22 has no independent variable"

# The gate must AGREE with the boolean away from its thresholds, or the run is
# testing a looser definition of "held" rather than the removal of the cliff.
_probe = [
    ("clearly held   d=15mm bump=0.90", 0.015, 0.90, 1.0),
    ("at the cliff   d=30mm bump=0.50", 0.030, 0.50, 0.0),
    ("just outside   d=31mm bump=0.48", 0.031, 0.48, 0.0),
    ("clearly not    d=45mm bump=0.20", 0.045, 0.20, 0.0),
]
_gate_lines = []
for _label, _d, _b, _boolean in _probe:
    _v = float(torch.sigmoid(torch.tensor((0.030 - _d) / NEAR_WIDTH))
               * torch.sigmoid(torch.tensor((_b - 0.5) / BUMP_WIDTH)))
    _gate_lines.append("  %-34s smooth %.3f   boolean %.0f" % (_label, _v, _boolean))

with torch.no_grad():
    report = [
        "start_from %s" % args.start_from,
        "  (stage 20's RESULT, not its start -- generalization is the thing being",
        "   kept, and stage 20 is the only policy that has it)",
        "",
        "WHAT IS ALREADY RULED OUT, and why this run is what is left:",
        "  stage 21   cube_upright REMOVED -> lift 2.7 cm, tilt 19.9 deg.",
        "             Both halves of the prediction failed; the term was not the price.",
        "  knee probe 3.39M scores 94.1%%/7.8 cm at CONTROL but 28.9%% at +x 4 cm,",
        "             and lifts 1 mm there. 3.79M scores 4.7%%. No checkpoint in",
        "             stage 20 holds the lift and the generalization at once.",
        "",
        "THE ONE CHANGE: _held stops being a boolean.",
        "  near    = sigmoid((0.030 - d) / %.3f)" % NEAR_WIDTH,
        "  stopped = sigmoid((bump - 0.5) / %.2f)" % BUMP_WIDTH,
        "  held    = near * stopped, in [0,1], multiplying the same five terms.",
        "",
        "GATE AGREEMENT CHECK (tight widths: the definition of held is unchanged,",
        "only the cliff at its boundary is removed):",
    ] + _gate_lines + [
        "",
        "WHY THE BOOLEAN AND NOT SOMETHING ELSE: it is the only candidate that",
        "  explains why RANDOMIZING THE SPAWN triggers the loss at every offset,",
        "  the unshifted control included. A fixed spawn makes every grasp nearly",
        "  identical and the flag stable; a varied grip makes it flicker, and it",
        "  flickers hardest under upward acceleration. Five channels zeroing",
        "  together makes a low still hold the only way to keep them all on.",
        "",
        "PREDICTIONS REGISTERED BEFORE THE RUN:",
        "  1. The lift trajectory STOPS DECAYING. A lift that merely falls more",
        "     slowly than stage 20's confirms nothing -- judge the shape, not the",
        "     final number.",
        "  2. Tilt rises somewhat. Partial credit for a disturbed grip is exactly",
        "     what this change pays for.",
        "  3. FALSIFIED IF the lift still decays to ~2-3 cm. Three explanations",
        "     would then have failed -- the squareness weight, the gate structure,",
        "     checkpoint selection -- and the trade should be called intrinsic to",
        "     this reward family under a randomized spawn. Ship stage 20 and stop.",
        "",
        "NO hysteresis, deliberately: it needs per-env state across steps and would",
        "  be a second variable. It goes on top of this only if this works.",
        "",
        "spawn range +-%.3f m (stage 20's, unchanged)" % SPAWN_RANGE,
        "gripper std at load %.4f" % float(model.policy.log_std[-1].exp()),
        "ent_coef %.4f, learning_rate %.1e, sticky hold %d (all stage 20's)" % (
            model.ent_coef, args.learning_rate, args.hold),
        "",
        "JUDGE WITH BOTH: sweep_shift.sh (six offsets, >90%% and >10 cm) AND",
        "  lift_trajectory.sh s22_traj stage22 (the shape across checkpoints).",
    ]
open("stage22_reset.txt", "w").write("\n".join(report) + "\n")
print("\n".join(report))

os.makedirs("checkpoints", exist_ok=True)
callbacks = [
    # One stage, one prefix. A shared prefix silently overwrote stage 12's
    # series and destroyed the only grasp evidence the project had.
    CheckpointCallback(
        save_freq=max(200_000 // args.num_envs, 1),
        save_path="checkpoints",
        name_prefix=args.save_prefix,
    ),
    GripperFloor(args.grip_std_floor),
]

model.learn(
    total_timesteps=args.total_timesteps,
    callback=callbacks,
    reset_num_timesteps=True,
)
model.save("checkpoints/%s_final" % args.save_prefix)

env.close()
simulation_app.close()
