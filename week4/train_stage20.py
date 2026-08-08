"""Stage 20 — teach the policy to LOOK at the cube.

Week 3 shipped stage16_final: 100% grasp, 13.5 cm median lift, 99.6% of 512
trials. Then the scene moved 3.4 cm and it grasped air.

The mechanism is not mysterious, and inspect_obs.py already recorded the proof.
The policy's observation group contains:

    object_position          -> object_position_in_robot_root_frame

So the cube's position has been in the input vector the entire time. It was also
CONSTANT for all of stages 1-19, because curriculum_lift_cfg.py pins the spawn
pose_range to (0,0,0). A constant input is indistinguishable from a bias: there
is no gradient that rewards attending to it, because varying it never changed
the optimal action. Twenty stages of PPO therefore learned an excellent
open-loop trajectory to one point in space and ignored the channel that would
have made it a closed-loop grasp.

That means this is a distribution problem, not an architecture problem and not a
reward problem. The fix is to make the input vary. Nothing else changes.

WHAT IS DELIBERATELY NOT CHANGED: the rewards, all of stage 16's weights, the
hold gate (0.030) and the cube_upright weight (4). Week 3's completed 2x2 showed
both of those levers independently destroy the lift, and stage 18 wasted a run
by moving two variables at once so that neither could be isolated -- stage 19
had to be run purely to undo that confusion. One variable moves here.

WARM START, NOT SCRATCH. The arm is solved (sub-millimetre approach that holds),
the grasp is solved (512/512), and the lift is solved at a fixed point. Throwing
that away costs a day and buys nothing. Learning rate stays low so the existing
behaviour is refined rather than overwritten.

EXPLORATION. Stage 16 annealed the gripper floor to 0.05, and the standing
correction applies: GripperEntropyAnneal only clamps a MINIMUM, so it is a floor
and never a cap -- it cannot force the std down, and at stage 17's load the std
read 1.0317 despite the "anneal". What actually made stage 15/16's hand decisive
was the MEAN committing to -2.15. So the floor is held at 0.05 here purely to
stop a re-freeze, and it is not expected to do any real work. Do not
re-attribute behaviour changes to it.

Judge with eval_shift.py, at offsets this run never trained on. Judging inside
the training distribution measures memorisation with extra steps.
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
                    help="week 3 saved under a doubled path (cwd was week3 and the"
                         " save path also began with week3/) -- this is that file")
parser.add_argument("--ent_coef", type=float, default=0.004,
                    help="ABOVE stage 16's 0.002. The task just got harder: the"
                         " policy has to find a behaviour conditioned on an input"
                         " it has been ignoring, and a nearly-deterministic policy"
                         " will not stumble into that. Still well under the 0.01"
                         " that produced per-step dither in stage 10.")
parser.add_argument("--hold", type=int, default=5)
parser.add_argument("--grip_std_floor", type=float, default=0.05)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--save_prefix", type=str, default="stage20")
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

from stage20_cfg import FrankaLiftStage20Cfg, SPAWN_RANGE


class StickyGripper(VecEnvWrapper):
    """Hold the gripper channel for `hold` frames per decision (stage 11).

    Kept identical to stage 16's, including the reset semantics. The replay and
    eval tooling MUST use the same hold or the policy is judged in an action
    regime it never trained in -- that mismatch silently invalidated the verdicts
    for stages 11 through 13.
    """

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
    """Keep the gripper's exploration from collapsing to zero. A FLOOR, only.

    Named honestly this time. clamp_(min=...) cannot pull a std down, so this
    term has never been able to make the hand decisive -- see the stage 17 load
    reading 1.0317. It exists to prevent a re-freeze of the kind that stalled
    stages 5-13, nothing more.
    """

    def __init__(self, floor: float):
        super().__init__()
        self.floor = floor

    def _on_rollout_end(self) -> None:
        with torch.no_grad():
            self.model.policy.log_std.data[-1].clamp_(min=math.log(self.floor))
        self.logger.record("gripper/std", float(self.model.policy.log_std[-1].exp()))

    def _on_step(self) -> bool:
        return True


env_cfg = FrankaLiftStage20Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="logs/stage20",
    custom_objects={"ent_coef": args.ent_coef, "learning_rate": args.learning_rate},
)

with torch.no_grad():
    report = [
        "start_from %s" % args.start_from,
        "WEEK 3 RESULT BEING GENERALIZED (stage16_final, check_airborne.py, 512 trials):",
        "  grasped 512/512 (100%), grasped AND airborne 510/512 (99.6%),",
        "  median lift while held 0.1349 m, cube tilt up to ~40 deg.",
        "  ...and it grasps AIR once the scene moves 3.4 cm. That is what this fixes.",
        "",
        "THE ONE CHANGE: cube spawn pose_range x,y = (-%.3f, %.3f), was (0,0) for"
        % (SPAWN_RANGE, SPAWN_RANGE),
        "  every stage 1-19. object_position_in_robot_root_frame was always in the",
        "  observation vector; it was simply constant, so it acted as a bias and the",
        "  policy had no gradient telling it to look. Rewards are stage 16's, exactly.",
        "",
        "gripper std at load %.4f" % float(model.policy.log_std[-1].exp()),
        "gripper std floor %.3f (a floor only -- it cannot force the std down)"
        % args.grip_std_floor,
        "ent_coef %.4f (RAISED from stage 16's 0.002 -- harder task needs exploration)"
        % model.ent_coef,
        "learning_rate %.1e (low: refine the solved arm/grasp, do not overwrite it)"
        % args.learning_rate,
        "sticky gripper: decisions held for %d frames" % args.hold,
        "stock curriculum still disabled (action_rate, joint_vel at base weights)",
        "",
        "SUCCESS TEST: eval_shift.py, at offsets NOT in the training range,",
        "  including the 3.4 cm diagonal. Gate: >90%% grasped+airborne and >10 cm",
        "  median lift at every held-out offset. Never from one replay, never from",
        "  cube height.",
    ]
open("stage20_reset.txt", "w").write("\n".join(report) + "\n")
print("\n".join(report))

os.makedirs("checkpoints", exist_ok=True)
callbacks = [
    # One stage, one prefix -- rerunning a script with a shared prefix silently
    # overwrote stage 12's checkpoint series and destroyed the only grasp
    # evidence the project had at that point.
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
