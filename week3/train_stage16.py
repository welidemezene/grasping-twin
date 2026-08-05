"""Stage 16 — lift it HIGH, and carry it SQUARE. The last two gaps in week 3.

Stage 15 was measured honestly for the first time, with check_airborne.py: cube
POSE rather than cube height (a tipped cube's centre reaches 0.0364 and clears
the 0.035 gate while still resting on the table), and all 512 parallel
environments at once rather than one anecdotal replay.

    ever grasped            512/512  (100%)
    grasped AND airborne    507/512  (99.0%)
    median lift while held  0.0099 m
    cube tilt               up to 79 degrees

The hard part is done. It grasps every single time, and the cube leaves the
table in 99% of trials. What remains is that the lift is minimal and ugly --
a median of 9.9 mm is invisible to the eye, and a 79-degree tilt is a corner
pinch rather than a grip. The user watched the replay and said exactly that
before any of these numbers existed.

Two causes, one change each, both in stage16_cfg:

  1. THE RAMP SATURATED. lifting_progress was tanh(h / 0.02), already ~0.96 by
     4 cm, so a 15 cm lift paid barely more than a 2 cm one. With nothing to
     gain above the minimum the policy took the minimum. Most of the weight now
     sits on tanh(h / 0.10), which keeps paying to ~20 cm; the near ramp stays
     at lower weight so the fine gradient that first found the lift survives.

  2. NOTHING CARED ABOUT ORIENTATION. Tilting the cube cost nothing, so a
     corner pinch scored the same as a clean grip. cube_upright_in_hand pays for
     keeping a face level while held, scaled so a corner-balance is worth zero.

Both additive, never multiplied -- stages 1-9 proved that multiplying channels
makes progress in one punishable by the other.

Training carries on from stage15_final with the gripper floor already annealed
to 0.1, so the hand stays committed while the arm learns to lift properly.

JUDGE WITH check_airborne.py, not by height and not by one replay. That mistake
has now been made twice in this project, once on 2026-07-25 and once today.
"""

import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str, default="week3/checkpoints/stage15_final")
parser.add_argument("--ent_coef", type=float, default=0.002,
                    help="a shade below stage 15's 0.003 -- the hand is decided,"
                         " the arm still has a lift to learn")
parser.add_argument("--hold", type=int, default=5)
parser.add_argument("--grip_std_start", type=float, default=0.1,
                    help="where stage 15's anneal finished -- the hand is committed")
parser.add_argument("--grip_std_end", type=float, default=0.05,
                    help="keep tightening, but never to zero")
parser.add_argument("--learning_rate", type=float, default=1e-4)
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

from stage16_cfg import FrankaLiftStage16Cfg


class StickyGripper(VecEnvWrapper):
    """Hold the gripper channel for `hold` frames per decision (see stage 11)."""

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


class GripperEntropyAnneal(BaseCallback):
    """Walk the gripper's exploration floor down instead of holding it up.

    Stage 14 pinned this floor at 0.5 so the channel could not re-freeze while
    it learned to close. It worked, and then kept working when it should have
    stopped: a permanent floor means a permanently indecisive hand. Here the
    floor decays linearly to `end`, so the policy explores early and commits
    late.
    """

    def __init__(self, start: float, end: float, total_timesteps: int):
        super().__init__()
        self.start, self.end, self.total = start, end, total_timesteps
        self.current = start

    def _on_rollout_end(self) -> None:
        frac = min(1.0, self.num_timesteps / max(1, self.total))
        self.current = self.start + (self.end - self.start) * frac
        with torch.no_grad():
            self.model.policy.log_std.data[-1].clamp_(min=math.log(self.current))
        self.logger.record("gripper/std_floor", self.current)
        self.logger.record("gripper/std", float(self.model.policy.log_std[-1].exp()))

    def _on_step(self) -> bool:
        return True


env_cfg = FrankaLiftStage16Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="week3/logs/stage16",
    custom_objects={"ent_coef": args.ent_coef, "learning_rate": args.learning_rate},
)

report = ["start_from %s" % args.start_from]
with torch.no_grad():
    report += [
        "STAGE 15 MEASURED (check_airborne.py, 512 trials, cube pose not height):",
        "  grasped 512/512 (100%), grasped AND airborne 507/512 (99.0%),",
        "  median lift while held 0.0099 m, cube tilt up to 79 deg.",
        "  The grasp is solved; the lift is minimal and the carry is a corner pinch.",
        "gripper std at load %.4f" % float(model.policy.log_std[-1].exp()),
        "gripper std floor anneals %.2f -> %.2f over %d steps"
        % (args.grip_std_start, args.grip_std_end, args.total_timesteps),
        "ent_coef %.4f (stage 15 used 0.003)" % model.ent_coef,
        "learning_rate %.1e" % args.learning_rate,
        "REWARD CHANGE 1: lifting_progress now 0.4*tanh(h/0.02) + 0.6*tanh(h/0.10),",
        "  weight 8 -> 12. The old single tanh(h/0.02) was saturated by 4 cm, so a",
        "  15 cm lift paid barely more than 2 cm and the policy took the minimum.",
        "REWARD CHANGE 2: cube_upright_in_hand, weight 4. Pays for keeping a face",
        "  level while held; a corner balance is worth exactly zero. Additive, not",
        "  multiplied -- stages 1-9 proved multiplied channels create valleys.",
        "sticky gripper: decisions held for %d frames" % args.hold,
        "stock curriculum disabled: action_rate and joint_vel stay at base weights",
    ]
open("stage16_reset.txt", "w").write("\n".join(report) + "\n")
print("\n".join(report))

callbacks = [
    CheckpointCallback(
        save_freq=max(200_000 // args.num_envs, 1),
        save_path="week3/checkpoints",
        name_prefix="stage16",          # one stage, one prefix
    ),
    GripperEntropyAnneal(args.grip_std_start, args.grip_std_end, args.total_timesteps),
]

model.learn(
    total_timesteps=args.total_timesteps,
    callback=callbacks,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage16_final")

env.close()
simulation_app.close()
