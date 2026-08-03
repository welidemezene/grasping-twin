"""Stage 11 — a gripper that commits. Sticky close/open held for 5 frames.

Stage 10 proved the reward fix worked for the arm: the 2M checkpoint parked
1.6 mm from the cube centre (0.8 mm off-centre, straddling it perfectly) and
spent 81 frames a step inside 1 cm — the best approach this project has ever
produced. It also exposed the final blocker, in one number: cube_xy_travel
4.28 cm. The policy attempted 31 closes, 64 close-frames right at the cube,
and every one of them dithered — fingers need 4-5 consecutive close commands
to reach the cube surface (finger_report.txt), but per-step exploration flips
the channel every frame or two, so each half-close just knocked the cube
off-centre and burned approach reward. PPO drew the rational conclusion:
closing near the cube loses money. By 10M it hovered at 2 cm again.

So the action, not the reward, gets fixed: the gripper command is sampled
once every HOLD frames and held in between. A sampled close is now a
committed 5-frame grasp attempt that physically reaches the cube; a sampled
open is a real release. With ~16 frames inside 1 cm per episode, close wins
the coin flip for several windows a rollout — contact happens, the
fingers_on_cube bump pays, and for the first time the jackpot is reachable.

Starts from stage 10's 2M checkpoint (the 1.6 mm parker) — the arm skill is
the asset; the hand relearns on top of it.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str, default="week3/checkpoints/stage10_1996800_steps")
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--hold", type=int, default=5, help="frames each gripper decision is held")
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from curriculum_lift_cfg import FrankaLiftStage1Cfg


class StickyGripper(VecEnvWrapper):
    """Hold the gripper channel for `hold` frames per decision.

    The arm channels pass through untouched every frame. The gripper obeys the
    policy only on frames where the countdown hits zero; in between, the last
    decision keeps acting. From PPO's point of view this is just environment
    dynamics ("my hand acts on a slower clock"), the standard action-repeat
    trick — it turns a 3% chance of 5 consecutive closes into a coin flip.
    """

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


env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)
env = StickyGripper(env, hold=args.hold)

model = PPO.load(
    args.start_from, env=env, tensorboard_log="week3/logs/stage11",
    custom_objects={"ent_coef": args.ent_coef},
)

report = []
with torch.no_grad():
    report.append("start_from %s" % args.start_from)
    report.append("gripper bias %.4f  log_std %.3f (kept — stage 10's head knows the cube)"
                  % (float(model.policy.action_net.bias[-1]), float(model.policy.log_std[-1])))
    if float(model.policy.log_std[-1]) < -0.7:
        model.policy.log_std[-1] = -0.5
        report.append("gripper log_std raised to -0.5 (was collapsing)")
    report.append("sticky gripper: decisions held for %d frames" % args.hold)
    report.append("ent_coef %.4f" % model.ent_coef)
open("stage11_reset.txt", "w").write("\n".join(report) + "\n")

checkpoint_callback = CheckpointCallback(
    save_freq=max(200_000 // args.num_envs, 1),
    save_path="week3/checkpoints",
    name_prefix="stage11",
)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=checkpoint_callback,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage11_final")

env.close()
simulation_app.close()
