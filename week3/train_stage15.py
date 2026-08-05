"""Stage 15 — stop the chatter, and pay for the lift that is already happening.

WHERE STAGE 14 LEFT IT. The exploration fix worked: the gripper commands close
(68/249 frames negative, against 0 of 249 in stage 13), the fingers stop at
0.0429 on a cube whose stop point is 0.0420, and the cube leaves the table.
Watching the replay shows the two things the summary hid:

  1. IT CHATTERS. 133 finger-sum direction reversals in 249 frames, 51 frames
     with a >4 mm jump, and the arm moving 4.3 mm every frame while it should be
     holding still. The grip opens and shuts instead of closing and staying shut.
  2. IT DROPS THE CUBE. Best lift 13.2 mm above resting, then back to 0.0210.

Both have the same two causes, and stage 15 addresses one each.

CAUSE 1 -- I never let stage 14 stop exploring. It floored the gripper's log_std
at 0.5 for the entire run, deliberately, to force the discovery. That floor is
still holding the output near zero, so the binarised channel flips sign on tiny
state changes every 5-frame decision. The chatter IS the exploration noise,
still switched on. Stage 14 was the explore half; this is the exploit half. The
floor now anneals 0.5 -> 0.1 across training, and ent_coef drops 0.01 -> 0.003,
so the policy can finally commit to "closed" instead of hovering at the boundary.

CAUSE 2 -- there is no reward for lifting until 0.035 m. `object_is_lifted` is a
step function, so carrying the cube 13 mm pays exactly the same as not lifting
it at all: nothing. No gradient, no learning, and the only route to a lift is to
stumble over the whole threshold in one go. That is the gripper bug one layer
up. stage15_cfg adds `lifting_progress`, a tanh ramp from the resting height
that pays for every millimetre while held, with the step bonus left on top as
the target.

Deliberately NOT added: a penalty on flipping the gripper command. It would
attack the chatter directly, but the entropy anneal should remove the cause
rather than mask the symptom -- and two changes at once is how the last nine
stages stayed un-diagnosable. If chatter survives this run, that penalty is the
next single change.

JUDGE BY THE PROBE, and by watching the replay. ep_rew_mean lied for nine
stages, and the viewer drew the gripper 72 mm from its real position until
commit a4cc4dd -- so trust probe_grasp.py numbers first, the fixed viewer
second, and the reward curve last.
"""

import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str, default="week3/checkpoints/stage14_final")
parser.add_argument("--ent_coef", type=float, default=0.003,
                    help="down from stage 14's 0.01: explore less, commit more")
parser.add_argument("--hold", type=int, default=5)
parser.add_argument("--grip_std_start", type=float, default=0.5,
                    help="stage 14's floor, where this run begins")
parser.add_argument("--grip_std_end", type=float, default=0.1,
                    help="floor at the end of training, low enough to commit")
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

from stage15_cfg import FrankaLiftStage15Cfg


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


env_cfg = FrankaLiftStage15Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="week3/logs/stage15",
    custom_objects={"ent_coef": args.ent_coef, "learning_rate": args.learning_rate},
)

report = ["start_from %s" % args.start_from]
with torch.no_grad():
    report += [
        "STAGE 14 LEFT: grasp real (finger sum 0.0429 vs cube-stop 0.0420),",
        "  but 133 finger direction reversals in 249 frames and the cube",
        "  lifted only 13.2 mm before being dropped.",
        "gripper std at load %.4f" % float(model.policy.log_std[-1].exp()),
        "gripper std floor anneals %.2f -> %.2f over %d steps"
        % (args.grip_std_start, args.grip_std_end, args.total_timesteps),
        "ent_coef %.4f (down from stage 14's 0.010 -- explore less, commit more)"
        % model.ent_coef,
        "learning_rate %.1e" % args.learning_rate,
        "NEW REWARD: lifting_progress, tanh ramp from rest height 0.0210 while",
        "  held, weight 8.0. The step bonus at 0.035 stays on top at weight 15.",
        "  Stage 14 earned NOTHING for 13.2 mm of lift; now every mm pays.",
        "sticky gripper: decisions held for %d frames" % args.hold,
        "stock curriculum disabled: action_rate and joint_vel stay at base weights",
    ]
open("stage15_reset.txt", "w").write("\n".join(report) + "\n")
print("\n".join(report))

callbacks = [
    CheckpointCallback(
        save_freq=max(200_000 // args.num_envs, 1),
        save_path="week3/checkpoints",
        name_prefix="stage15",          # one stage, one prefix
    ),
    GripperEntropyAnneal(args.grip_std_start, args.grip_std_end, args.total_timesteps),
]

model.learn(
    total_timesteps=args.total_timesteps,
    callback=callbacks,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage15_final")

env.close()
simulation_app.close()
