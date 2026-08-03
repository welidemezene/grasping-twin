"""Stage 6 — continue training from the checkpoint that actually reached the cube.

The 2M checkpoint of stage5 got the gripper to 0.0134 m and spent 8 frames inside
the 3 cm grasp zone. The reward then punished it for being there, so later
checkpoints retreated. That reaching skill is expensive to discover and cheap to
keep, so this starts from that checkpoint instead of from random weights.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--start_from", type=str, default="week3/checkpoints/stage5_1996800_steps")
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from curriculum_lift_cfg import FrankaLiftStage1Cfg

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
env = Sb3VecEnvWrapper(env)

# weights and PPO settings come from the checkpoint; only the reward changed
model = PPO.load(args.start_from, env=env, tensorboard_log="week3/logs/stage6")

checkpoint_callback = CheckpointCallback(
    save_freq=max(200_000 // args.num_envs, 1),
    save_path="week3/checkpoints",
    name_prefix="stage6",
)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=checkpoint_callback,
    reset_num_timesteps=True,
)
model.save("week3/checkpoints/stage6_final")
print("Stage 6 complete. Model saved.")

env.close()
simulation_app.close()
