"""Stage 18 -- the middle of the hold-gate bracket, with squareness back to timid.

Stage 17 moved two levers at once. The grip fix landed (cube 31.5 -> 12.7 mm
from the fingertips, genuinely inside the 21 mm half-width) and the lift
collapsed (140.9 -> 31.6 mm). See stage18_cfg.py for why each change did what
it did; in short, 0.015 turned the boolean hold gate into a flickering cliff,
and cube_upright at weight 10 paid a guaranteed 10 for hovering at rest against
a gamble for 12 by lifting.

  1. hold_distance 0.015 -> 0.02. The untried middle. 0.03 admitted the corner
     pinch at 0.0286; 0.015 left only 2.3 mm of margin above a real grip.
  2. cube_upright weight 10 -> 4, stage 16's value, so squareness can never
     out-earn lifting at rest.

Warm-starts from stage16_final, exactly as stage 17 did, so this is directly
comparable to both. Everything else is stage 17's setup untouched.

Checkpoint prefix is "stage18" -- NOT a rerun of train_stage17.py, which would
overwrite stage 17's checkpoints by prefix. One stage, one prefix.

Judge with check_airborne.py (512 trials, corner geometry) plus a recorded
replay and the grip geometry. Never from one replay, never from cube height.
"""

import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512,
                    help="512 deliberately. n_steps=64 is frozen into the "
                         "checkpoints, so num_envs sets the rollout size: at "
                         "1024 this run would get HALF the gradient updates of "
                         "stages 16 and 17 and would not be comparable to "
                         "either. Measured gain for that was only 17%% "
                         "(43.0 -> 36.7 min). Not worth a confounded result.")
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str, default="week3/checkpoints/stage16_final")
parser.add_argument("--ent_coef", type=float, default=0.002)
parser.add_argument("--hold", type=int, default=5)
parser.add_argument("--grip_std_start", type=float, default=0.1)
parser.add_argument("--grip_std_end", type=float, default=0.05)
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

from stage18_cfg import FrankaLiftStage18Cfg, MID_HOLD, UPRIGHT_WEIGHT


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
    """Walk the gripper's exploration floor down so it explores early, commits late.

    NOTE: clamp_(min=...) is a FLOOR, never a cap. Stage 17 loaded with the
    gripper std at 1.0317 despite stage 16 "annealing" it to 0.05 -- the floor
    never bound, PPO simply pushed the std back up. What actually stopped the
    stage 15 chatter was the gripper MEAN committing to -2.15, not this.
    Kept for parity with stages 15-17; do not credit it with more than it does.
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


env_cfg = FrankaLiftStage18Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="week3/logs/stage18",
    custom_objects={"ent_coef": args.ent_coef, "learning_rate": args.learning_rate},
)

report = ["start_from %s" % args.start_from]
with torch.no_grad():
    report += [
        "THE BRACKET: hold_distance 0.03 was too loose (corner pinch at 0.0286",
        "  passed, cube hung off the fingertips at 31.5 mm). 0.015 was too tight",
        "  (only 2.3 mm above a real 12.7 mm grip, so the boolean gate flickered",
        "  and the lift fell 140.9 -> 31.6 mm). This run takes the middle.",
        "CHANGE 1: hold_distance -> %.3f on every term that gates on it." % MID_HOLD,
        "CHANGE 2: cube_upright weight -> %.1f (stage 17 used 10.0). At 10 it paid"
        % UPRIGHT_WEIGHT,
        "  a guaranteed 10 for hovering square at rest vs a gamble for 12 by",
        "  lifting, so the policy stayed low. Lifting must dominate at rest.",
        "num_envs %d -- SAME as stages 16 and 17. n_steps=64 is frozen in the"
        % args.num_envs,
        "  checkpoint, so raising num_envs would halve the gradient updates and",
        "  make this incomparable to the runs it exists to sit between.",
        "gripper std at load %.4f" % float(model.policy.log_std[-1].exp()),
        "ent_coef %.4f" % model.ent_coef,
        "learning_rate %.1e" % args.learning_rate,
        "sticky gripper: decisions held for %d frames" % args.hold,
        "stock curriculum disabled: action_rate and joint_vel stay at base weights",
        "",
        "SUCCESS = stage 16's lift (median ~0.13 m) with stage 17's grip",
        "  (cube within ~21 mm of the fingertips). Judge with check_airborne.py",
        "  over 512 trials plus a replay -- never one replay, never by height.",
    ]
open("stage18_reset.txt", "w").write("\n".join(report) + "\n")
print("\n".join(report))

callbacks = [
    CheckpointCallback(
        save_freq=max(200_000 // args.num_envs, 1),
        save_path="week3/checkpoints",
        name_prefix="stage18",          # one stage, one prefix
    ),
    GripperEntropyAnneal(args.grip_std_start, args.grip_std_end, args.total_timesteps),
]

model.learn(
    total_timesteps=args.total_timesteps,
    callback=callbacks,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage18_final")

env.close()
simulation_app.close()
