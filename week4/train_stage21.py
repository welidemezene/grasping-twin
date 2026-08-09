"""Stage 21 — is cube_upright_in_hand what the lift is being traded for?

Stage 20 bought generalization and sold the lift. The trade was continuous
across training, not a late collapse, and ep_rew_mean rose 66 -> 96.5 while the
median lift fell 0.1029 -> 0.0257 m: the reward paid for the loss. Lift and tilt
fell together (39.4 -> 20.3 deg), which points at the one term that pays for
squareness at ANY height while the height term needs height.

This run removes that term and changes nothing else. See stage21_cfg.py for why
the weight goes to 0 rather than 2, and for the registered prediction that tilt
will regress if the explanation is right.

WARM START FROM stage16_final -- THE SAME PLACE STAGE 20 STARTED, not from
stage20_final. This is the whole point of the run. Starting from stage 20's
weights would confound "the term was the price" with "the policy could recover
once released", and the two are different claims. Same start, same steps, same
everything but one reward term: then the difference between the two runs is
attributable, and if the lift comes back it came back because of the term.

Everything else is stage 20's: spawn +-5 cm, ent_coef 0.004, lr 1e-4, sticky
hold 5, gripper floor 0.05 (a floor only -- it cannot force the std down, and
the std read 1.0317 at stage 20's load despite stage 16 "annealing" to 0.05).

JUDGING IT. Two things, and the second is not optional:

  1. sweep_shift.sh at the six held-out offsets, gate >90% and >10 cm
     everywhere.
  2. lift_trajectory.sh across checkpoints. Stage 20's final numbers were the
     LEAST informative thing about it -- the shape across training is what
     revealed the trade, and a stage 21 that merely slides down the same hill
     more slowly must not be read as a stage 21 that stays up.
"""

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str,
                    default="../week3/week3/checkpoints/stage16_final",
                    help="stage 20's start, not stage 20's result -- see above")
parser.add_argument("--ent_coef", type=float, default=0.004,
                    help="stage 20's value, held fixed so the reward term is the "
                         "only difference between the two runs")
parser.add_argument("--hold", type=int, default=5)
parser.add_argument("--grip_std_floor", type=float, default=0.05)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--save_prefix", type=str, default="stage21")
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

from stage20_cfg import SPAWN_RANGE
from stage21_cfg import FrankaLiftStage21Cfg


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


env_cfg = FrankaLiftStage21Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="logs/stage21",
    custom_objects={"ent_coef": args.ent_coef, "learning_rate": args.learning_rate},
)

# Fail loudly rather than train for 45 minutes on the wrong reward. A silently
# still-present cube_upright would produce a rerun of stage 20 wearing stage
# 21's name, and the two would then be compared as if they differed.
assert getattr(env_cfg.rewards, "cube_upright", None) is None, \
    "cube_upright is still active -- stage 21 has no independent variable"

with torch.no_grad():
    report = [
        "start_from %s" % args.start_from,
        "  (stage 20's START, not its result -- same origin so the reward term is",
        "   the only difference between the two runs)",
        "",
        "WHAT STAGE 20 SHOWED, and what this run tests:",
        "  generalization SOLVED   +x 4 cm 41.8%% -> 98.6%%, 5 cm diag 41.0%% -> 99.8%%",
        "  lift LOST               0.1345 -> 0.0257 m at every offset incl. control",
        "  traded continuously     2.6M 0.1029/39.4deg -> 10M 0.0257/20.3deg",
        "  and PAID FOR            ep_rew_mean rose 66 -> 96.5 over that same span",
        "",
        "THE ONE CHANGE: cube_upright_in_hand REMOVED (was weight 4.0).",
        "  It pays for squareness at ANY height; lifting_progress needs height.",
        "  Once the spawn moves and a lift is a gamble, a square hold just above",
        "  the table is the guaranteed earner. Week 3's stage 19 already isolated",
        "  this term (4 -> 10 dropped the lift 124.3 -> 15.3 mm); what is new is",
        "  that randomizing the spawn makes weight 4 unsurvivable too.",
        "",
        "PREDICTION REGISTERED BEFORE THE RUN: tilt WILL regress (stage 15, with",
        "  no squareness term, tilted to 79 deg). Lift returning at the cost of",
        "  tilt is a frontier to price, not a failure. What would FALSIFY the",
        "  explanation is the lift staying near 2.6 cm with the term gone.",
        "",
        "spawn range +-%.3f m (stage 20's, unchanged)" % SPAWN_RANGE,
        "gripper std at load %.4f" % float(model.policy.log_std[-1].exp()),
        "ent_coef %.4f, learning_rate %.1e, sticky hold %d (all stage 20's)" % (
            model.ent_coef, args.learning_rate, args.hold),
        "",
        "JUDGE WITH BOTH: sweep_shift.sh (six offsets, >90%% and >10 cm) AND",
        "  lift_trajectory.sh (the shape across checkpoints). Stage 20's final",
        "  numbers were the least informative thing about it.",
    ]
open("stage21_reset.txt", "w").write("\n".join(report) + "\n")
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
