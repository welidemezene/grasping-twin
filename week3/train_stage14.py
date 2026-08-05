"""Stage 14 — give the gripper back its exploration. One change, measured.

THE PROBE ANSWERED IT (probe_s13_summary.txt). On stage13_final, across 249
frames with 127 of them inside 5 mm of the cube:

    raw gripper action   min +1.8585   max +2.2830   mean +2.0720
    close threshold       0.0
    close commands        0

The gripper channel is binarised at zero, and the policy's output never once
goes below +1.86. With stage 13's final std of 0.503, sampling a close is a
4.1-sigma event (~2e-5). Close was not merely unlearned -- it was never
SAMPLED, in training or replay. PPO cannot learn an action it never tries,
however well shaped the reward is. That is the whole explanation for stages
5-13 all converging on "approach, never close".

The probe also killed the alternatives: yaw to the nearest cube face averages
6.0 deg (face-on, so the 0.0602-vs-0.0594 diagonal coincidence was just a
coincidence), the cube moves 0.0014 m at contact (not batted away), and `bump`
climbs 0.054 -> 0.311 as the fingers drift in, so the reward gradient is there
-- nothing samples into it.

AND IT IS A RECURRENCE. The stage 8/9 autopsy found this same mean at +1.53 and
zeroed the action head. It came back at +2.07. That fix failed because it reset
the MEAN but left entropy free to collapse: ent_coef was tapered 0.01 -> 0.003
-> 0.001 across stages 12-13, std fell 0.642 -> 0.503, and the gripper
dimension quietly froze at "open" again. Re-zeroing alone would fail a third
time.

So stage 14 does three things, all aimed at the same target:

  1. Zero the gripper ROW of the action head -- weight and bias -- so the mean
     output is exactly 0.0 and close is sampled ~50% of the time from step one.
  2. Set the gripper's log_std to ~1.0 and FLOOR IT for the whole run. This is
     the step stages 8/9 missed. A per-rollout callback clamps that one element
     so entropy collapse can never re-freeze the channel.
  3. Raise ent_coef back to 0.01 (undoing the taper) to keep overall
     exploration alive while the hand learns.

Deliberately NOT changed: the reward (no new terms -- one variable at a time,
which is the lesson the probe just taught), the sticky hold of 5 frames (a
sampled close must persist ~4 frames to reach 0.0386, and finger_report.txt
says 5 is enough), and the disabled stock curriculum.

The arm is NOT frozen -- 127/249 frames within 5 mm means approach is solved
and the approach term (weight 4.0) is strong enough to hold it -- but the
learning rate is lowered by default so a suddenly-noisy gripper cannot shake
the arm apart. Watch the first replay for arm regression; if the min gap grows
past ~5 mm, freeze the arm rows next time.

WHAT SUCCESS LOOKS LIKE: raw gripper action goes negative at the cube, finger
sum drops below 0.054, and `_held` starts firing. Re-run probe_grasp.py on the
checkpoints -- do not judge this by ep_rew_mean, which lied for nine stages.
"""

import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str, default="week3/checkpoints/stage13_final")
parser.add_argument("--ent_coef", type=float, default=0.01,
                    help="undo the stage 12-13 taper; keep exploration alive")
parser.add_argument("--hold", type=int, default=5)
parser.add_argument("--grip_std", type=float, default=1.0,
                    help="initial std for the gripper channel")
parser.add_argument("--grip_std_floor", type=float, default=0.5,
                    help="the gripper std is never allowed below this")
parser.add_argument("--learning_rate", type=float, default=1e-4,
                    help="lower than SB3 default so the re-randomised gripper "
                         "cannot shake the solved arm apart")
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

from curriculum_lift_cfg import FrankaLiftStage1Cfg


class StickyGripper(VecEnvWrapper):
    """Hold the gripper channel for `hold` frames per decision (see stage 11)."""

    def __init__(self, venv, hold=5):
        super().__init__(venv)
        self.hold = hold
        self.count = np.zeros(venv.num_envs, dtype=np.int64)
        self.held = np.ones(venv.num_envs, dtype=np.float32)  # start open

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


class GripperEntropyFloor(BaseCallback):
    """Stop the gripper channel from freezing shut again.

    Stages 8/9 reset the gripper mean and it drifted back to always-open,
    because nothing stopped that dimension's std from collapsing. This clamps
    the last element of log_std after every rollout, so however far the taper
    pushes the rest of the policy toward exploitation, the hand keeps trying.
    """

    def __init__(self, floor: float):
        super().__init__()
        self.log_floor = math.log(floor)

    def _on_rollout_end(self) -> None:
        with torch.no_grad():
            self.model.policy.log_std.data[-1].clamp_(min=self.log_floor)

    def _on_step(self) -> bool:
        return True


env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="week3/logs/stage14",
    custom_objects={"ent_coef": args.ent_coef, "learning_rate": args.learning_rate},
)

report = ["start_from %s" % args.start_from]
with torch.no_grad():
    net = model.policy.action_net
    before_bias = float(net.bias[-1])
    before_wnorm = float(net.weight[-1].norm())
    before_std = float(model.policy.log_std[-1].exp())

    # 1. zero the gripper row entirely -> output is exactly 0.0, so the
    #    binarised channel is a coin flip instead of a 4.1-sigma long shot
    net.weight[-1].zero_()
    net.bias[-1].zero_()
    # 2. hand the channel its exploration back
    model.policy.log_std.data[-1] = math.log(args.grip_std)

    after_std = float(model.policy.log_std[-1].exp())
    sigmas = before_bias / before_std if before_std > 0 else float("inf")

    report += [
        "PROBE FINDING: gripper action mean was +2.0720, never below +1.8585,",
        "  threshold 0.0, so 0 close commands in 249 frames.",
        "gripper bias   %+.4f -> %+.4f" % (before_bias, float(net.bias[-1])),
        "gripper w-norm  %.4f -> %.4f" % (before_wnorm, float(net.weight[-1].norm())),
        "gripper std     %.4f -> %.4f (floored at %.2f for the whole run)"
        % (before_std, after_std, args.grip_std_floor),
        "close was %.1f sigma away at load; it is now ~0.0 sigma (a coin flip)" % sigmas,
        "ent_coef %.4f (taper 0.01->0.003->0.001 UNDONE; that taper is what let"
        " the stage 8/9 gripper reset drift back to always-open)" % model.ent_coef,
        "learning_rate %.1e (lowered to protect the solved arm)" % args.learning_rate,
        "sticky gripper: decisions held for %d frames" % args.hold,
        "stock curriculum disabled: action_rate and joint_vel stay at base weights",
        "reward UNCHANGED from stage 13 -- one variable at a time",
    ]
open("stage14_reset.txt", "w").write("\n".join(report) + "\n")
print("\n".join(report))

callbacks = [
    CheckpointCallback(
        save_freq=max(200_000 // args.num_envs, 1),
        save_path="week3/checkpoints",
        name_prefix="stage14",          # one stage, one prefix
    ),
    GripperEntropyFloor(args.grip_std_floor),
]

model.learn(
    total_timesteps=args.total_timesteps,
    callback=callbacks,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage14_final")

env.close()
simulation_app.close()
