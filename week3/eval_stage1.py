"""Load the TRAINED Stage 1 policy and watch the real reward components.
The question: does lifting_object finally go above 0.0?"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch, gymnasium as gym
from stable_baselines3 import PPO
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from curriculum_lift_cfg import FrankaLiftStage1Cfg

env_cfg = FrankaLiftStage1Cfg()
env_cfg.scene.num_envs = 1
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
raw_env = env.unwrapped
vec_env = Sb3VecEnvWrapper(env)

model = PPO.load("week3/checkpoints/stage1_final", env=vec_env)

obs = vec_env.reset()
lift_hits = 0
for step in range(120):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    terms = raw_env.reward_manager.get_active_iterable_terms(env_idx=0)
    row = {name: float(v[0]) if hasattr(v, "__len__") else float(v) for name, v in terms}
    lift = row.get("lifting_object", 0.0)
    if lift > 0:
        lift_hits += 1
    if step % 10 == 0:
        print(f"step {step:3d} | reach {row.get('reaching_object',0):.3f} "
              f"| grasp {row.get('grasping_object',0):.3f} "
              f"| LIFT {lift:.3f}")

print(f"\nLifting happened on {lift_hits} of 120 steps.")
if lift_hits > 0:
    print(">>> SUCCESS: the robot is lifting the cube.")
else:
    print(">>> Not lifting yet — grips but doesn't raise it.")

vec_env.close()
simulation_app.close()
