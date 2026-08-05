"""Stage 17 — make "held" mean actually held.

Stage 16 lifts the cube 18 cm in 99.6% of 512 trials, and does it by hooking a
corner. Measured during the lift: the cube's centre sits 28.4 mm from the
fingertip centre — further than the cube's own half-width of 21 mm — and it is
carried at a tilt that never drops below 28.6 degrees, averaging 39.8.

It lifts anyway because friction does not care where you squeeze. Two fingers
pinching any part of a cube will hold it against gravity, the same way you can
lift a book by pinching one corner and letting it dangle.

The cause is one number. `_held` accepted anything within HOLD_DISTANCE of the
fingertips and HOLD_DISTANCE was 0.03; the pinch sits at 0.0286, inside the gate
by 1.4 mm. So a corner pinch scored exactly the same as a proper grip, and the
pinch is easier to find. The policy did precisely what it was paid to do.

This is the original week 3 reward hack one level finer. There, "height" could
be earned without a grasp. Here, "held" can be earned without a grip.

  1. HOLD_DISTANCE 0.03 -> 0.015 on every term that gates on it. Half the cube's
     width, so it must sit between the fingers. The load-bearing change.
  2. cube_upright weight 4 -> 10, because at weight 4 a 40-degree carry cost
     almost nothing, which is why it stayed at 40.

Lifting still outweighs squareness (12 vs 10) so a bad lift beats no lift — the
robot must never be better off leaving the cube alone.

Expect the success rate to DROP at first. The old pinch no longer counts as a
hold, so stage 16's habit stops paying and a real grip has to be found. Judge
with check_airborne.py plus the grip geometry, not by the reward curve.
"""

import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str, default="week3/checkpoints/stage16_final")
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

from stage17_cfg import FrankaLiftStage17Cfg


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


env_cfg = FrankaLiftStage17Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="week3/logs/stage17",
    custom_objects={"ent_coef": args.ent_coef, "learning_rate": args.learning_rate},
)

report = ["start_from %s" % args.start_from]
with torch.no_grad():
    report += [
        "STAGE 16 MEASURED: grasped AND airborne 510/512 (99.6%), median lift",
        "  0.1349 m -- but the cube centre sits 28.4 mm from the fingertips (its",
        "  own half-width is 21 mm) at a tilt never below 28.6 deg. A corner hook.",
        "gripper std at load %.4f" % float(model.policy.log_std[-1].exp()),
        "gripper std floor anneals %.2f -> %.2f over %d steps"
        % (args.grip_std_start, args.grip_std_end, args.total_timesteps),
        "ent_coef %.4f (stage 15 used 0.003)" % model.ent_coef,
        "learning_rate %.1e" % args.learning_rate,
        "CHANGE 1: hold_distance 0.03 -> 0.015 on EVERY term that gates on it.",
        "  Stage 16's corner pinch sat at 0.0286 -- inside the old gate by 1.4 mm --",
        "  so a pinch and a real grip paid the same. Now the cube must be IN the hand.",
        "CHANGE 2: cube_upright weight 4 -> 10. At weight 4 a 40 deg carry cost",
        "  almost nothing, which is why stage 16 stayed at 40 deg.",
        "Lifting (12) still outweighs squareness (10): a bad lift must beat no lift.",
        "sticky gripper: decisions held for %d frames" % args.hold,
        "stock curriculum disabled: action_rate and joint_vel stay at base weights",
    ]
open("stage17_reset.txt", "w").write("\n".join(report) + "\n")
print("\n".join(report))

callbacks = [
    CheckpointCallback(
        save_freq=max(200_000 // args.num_envs, 1),
        save_path="week3/checkpoints",
        name_prefix="stage17",          # one stage, one prefix
    ),
    GripperEntropyAnneal(args.grip_std_start, args.grip_std_end, args.total_timesteps),
]

model.learn(
    total_timesteps=args.total_timesteps,
    callback=callbacks,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage17_final")

env.close()
simulation_app.close()
