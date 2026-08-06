"""Stage 19 -- keep the gate that lifts, fix only the tilt. The last week 3 run.

Measured across the whole bracket (check_airborne.py, 512 trials each):

    hold gate   lift median   tilt median   grasped+airborne
    0.030          124.3 mm      40.2 deg       99.6%   stage 16
    0.020           23.1 mm      25.2 deg      100.0%   stage 18
    0.015           21.5 mm      72.7 deg       97.7%   stage 17

Tightening the hold gate at all destroys the lift, monotonically -- there is no
good middle. Stage 18 also refuted my explanation for stage 17: I claimed
cube_upright at weight 10 out-competed lifting, but stage 18 reverted it to 4
and the lift still did not return. The gate is doing all the work.

So the gate is left ALONE at stage 16's 0.03, and the single remaining
complaint about stage 16 -- 40 deg of tilt -- is addressed directly:

    ONE CHANGE: cube_upright weight 4 -> 10. Nothing else moves.

This is the one untested cell. Stages 17 and 18 both varied the tilt weight
while the gate was tightened, so the tilt weight has never been tried against
the gate that actually lifts.

Warm-starts from stage16_final like stages 17 and 18, so all four are directly
comparable.

Checkpoint prefix is "stage19" -- one stage, one prefix, so no earlier stage
gets overwritten by SB3's prefix+step naming.

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

from stage19_cfg import FrankaLiftStage19Cfg, UPRIGHT_WEIGHT


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


env_cfg = FrankaLiftStage19Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="week3/logs/stage19",
    custom_objects={"ent_coef": args.ent_coef, "learning_rate": args.learning_rate},
)

report = ["start_from %s" % args.start_from]
with torch.no_grad():
    report += [
        "THE BRACKET IS REFUTED. Measured over 512 trials each:",
        "  gate 0.030 -> lift 124.3 mm, tilt 40.2 deg  (stage 16)",
        "  gate 0.020 -> lift  23.1 mm, tilt 25.2 deg  (stage 18)",
        "  gate 0.015 -> lift  21.5 mm, tilt 72.7 deg  (stage 17)",
        "Tightening the gate destroys the lift monotonically; no good middle",
        "exists. Stage 18 also disproved the cube_upright explanation for stage",
        "17's collapse -- weight went back to 4 and the lift did NOT return.",
        "",
        "ONLY CHANGE: cube_upright weight -> %.1f. hold_distance stays at stage"
        % UPRIGHT_WEIGHT,
        "  16's 0.03, untouched, because it is the only gate that lifts. The",
        "  tilt weight has never been tested against it -- stages 17 and 18 both",
        "  moved it while the gate was tight. This is that missing cell.",
        "num_envs %d -- SAME as stages 16, 17 and 18. n_steps=64 is frozen in the"
        % args.num_envs,
        "  checkpoint, so raising num_envs would halve the gradient updates and",
        "  make this incomparable to the runs it exists to be compared with.",
        "gripper std at load %.4f" % float(model.policy.log_std[-1].exp()),
        "ent_coef %.4f" % model.ent_coef,
        "learning_rate %.1e" % args.learning_rate,
        "sticky gripper: decisions held for %d frames" % args.hold,
        "stock curriculum disabled: action_rate and joint_vel stay at base weights",
        "",
        "SUCCESS = stage 16's lift (median ~0.12 m) with tilt under 30 deg.",
        "FAILURE = the lift collapses again, which would mean cube_upright at 10",
        "  is harmful regardless of the gate, and stage 16 is week 3's result.",
        "Either way week 3 ends here. Judge with check_airborne.py over 512",
        "  trials plus a replay -- never one replay, never by cube height.",
    ]
open("stage19_reset.txt", "w").write("\n".join(report) + "\n")
print("\n".join(report))

callbacks = [
    CheckpointCallback(
        save_freq=max(200_000 // args.num_envs, 1),
        save_path="week3/checkpoints",
        name_prefix="stage19",          # one stage, one prefix
    ),
    GripperEntropyAnneal(args.grip_std_start, args.grip_std_end, args.total_timesteps),
]

model.learn(
    total_timesteps=args.total_timesteps,
    callback=callbacks,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage19_final")

env.close()
simulation_app.close()
