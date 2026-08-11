"""Stage 23 — widen the spawn box from ±5 cm to ±12 cm. One variable.

The envelope probe found stage22_final's working region is a band about 6 cm wide
in x and the full ±13 cm in y, with 0.01 arm joints pinned on failures — so the
arm can reach everywhere it is failing and this is a training-distribution
problem, not a kinematic one. See stage23_cfg.py for the map, the argument, and
the registered predictions.

WARM START FROM stage22_final. It is the only policy that has both the
generalization and the lift; this run asks whether the envelope can be widened on
top of them.

JUDGING IT — the primary verdict is the probe, not the sweep:
  1. probe_envelope.py checkpoints/stage23_final s23_wide --range 0.15
  2. sweep_shift.sh checkpoints/stage23_final s23
  3. lift_trajectory.sh s23_traj stage23
"""

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str,
                    default="checkpoints/stage22_final",
                    help="the only policy with generalization AND the lift")
parser.add_argument("--ent_coef", type=float, default=0.004,
                    help="stage 22's value, held fixed so the spawn box is the only change")
parser.add_argument("--hold", type=int, default=5)
parser.add_argument("--grip_std_floor", type=float, default=0.05)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--save_prefix", type=str, default="stage23")
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

# stage23_cfg pulls in stage22_cfg, which is what puts week3/ on sys.path where
# grasp_reward lives. Importing grasp_reward above this line is a ModuleNotFound.
from stage23_cfg import FrankaLiftStage23Cfg, SPAWN_RANGE
from stage22_cfg import _held_smooth
from stage20_cfg import SPAWN_RANGE as STAGE20_SPAWN_RANGE

import grasp_reward


class StickyGripper(VecEnvWrapper):
    """Stage 22's, byte for byte. The eval tooling uses the same hold; judging a
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


env_cfg = FrankaLiftStage23Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="logs/stage23",
    custom_objects={"ent_coef": args.ent_coef, "learning_rate": args.learning_rate},
)

# Fail loudly rather than train 45 minutes on the wrong config. Two ways this run
# could silently become a rerun of stage 22 wearing a new name:
_range = env_cfg.events.reset_object_position.params["pose_range"]["x"]
assert abs(_range[1] - SPAWN_RANGE) < 1e-9 and abs(_range[0] + SPAWN_RANGE) < 1e-9, \
    "spawn box is %s, not +-%.3f -- the __post_init__ override did not take" % (
        _range, SPAWN_RANGE)
assert SPAWN_RANGE > STAGE20_SPAWN_RANGE, \
    "spawn box is not wider than stage 20's -- stage 23 has no independent variable"
assert grasp_reward._held is _held_smooth, \
    "_held is the boolean -- stage 22's gate did not survive inheritance"

with torch.no_grad():
    report = [
        "start_from %s" % args.start_from,
        "  (the only policy holding generalization AND the lift; this run asks",
        "   whether the ENVELOPE can widen on top of them)",
        "",
        "THE ONE CHANGE: spawn box +-%.3f -> +-%.3f m." % (
            STAGE20_SPAWN_RANGE, SPAWN_RANGE),
        "  Smooth hold gate, every reward weight, the 0.030 hold distance,",
        "  ent_coef, learning rate and sticky hold are all stage 22's.",
        "",
        "WHY, from probe_envelope.py on stage22_final over a +-0.15 box:",
        "  overall 186/512 lifted (36.3%%)",
        "  the working region is a BAND -- full +-13 cm in y at centre x, and",
        "  zero past |x| ~ 0.06 in BOTH directions, near and far alike.",
        "  arm joints pinned on failures 0.01, same as on successes.",
        "  => the arm CAN reach where it fails. Policy wall, not kinematic,",
        "     so more examples is the intervention that fits.",
        "",
        "PREDICTIONS REGISTERED BEFORE THE RUN:",
        "  1. The x band WIDENS toward +-0.12. If it does not move, widening the",
        "     distribution is not sufficient and the x gain needs another",
        "     mechanism than more examples.",
        "  2. The lift DROPS again, at least early -- a varied grip flickers the",
        "     hold gate, which is what cost 13.5 -> 2.6 cm at +-5 cm.",
        "  3. It RECOVERS by the end if the smooth gate is the real mechanism,",
        "     as it did in stage 22 (0.032 -> 0.030 -> 0.066 -> 0.144). Judge the",
        "     SHAPE across checkpoints, not the final number.",
        "  4. FALSIFIED IF the lift decays and stays down while the band widens.",
        "     The smooth gate would then hold only at +-5 cm, and the honest",
        "     statement is that this reward family trades lift for envelope at a",
        "     fixed rate. Ship stage 22 and stop widening.",
        "  5. Y MUST NOT REGRESS. It already works to +-13 cm. If y worsens while",
        "     x improves, capacity is binding, not the distribution.",
        "",
        "gripper std at load %.4f" % float(model.policy.log_std[-1].exp()),
        "ent_coef %.4f, learning_rate %.1e, sticky hold %d (all stage 22's)" % (
            model.ent_coef, args.learning_rate, args.hold),
        "",
        "JUDGE WITH, in this order:",
        "  1. probe_envelope.py checkpoints/stage23_final s23_wide --range 0.15",
        "     <- PRIMARY. Same instrument and box as the map above.",
        "  2. sweep_shift.sh checkpoints/stage23_final s23",
        "  3. lift_trajectory.sh s23_traj stage23",
    ]
open("stage23_reset.txt", "w").write("\n".join(report) + "\n")
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
