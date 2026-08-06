"""Measure simulation throughput vs num_envs, and find the VRAM ceiling.

Memory said the RTX 5070 Ti sat at ~9% GPU utilisation with 512 envs, i.e. 512
is leaving a large speedup unused. This answers two questions with numbers
instead of a guess:

  1. steps/s at each env count -> what a 10M-step run would actually cost
  2. where it runs out of the 12 GB and dies

It drives the REAL stage 17 policy through the REAL env (sticky gripper and
all), so the number is what training would see, not a synthetic loop. The
optimiser is not run -- this measures the rollout side, which is what scales
with num_envs. Gradient cost per step FALLS as num_envs rises (see the
n_steps note below), so the true training speedup is at least this good.

ONE env count per process -- see --envs. Appends a line to --out so a shell
loop over container runs builds the whole table.

Usage (inside the container, from /workspace/week3):
    bench_envs.py --envs 2048 --steps 300
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--envs", type=int, required=True,
                    help="ONE env count per process. Isaac Lab's simulation "
                         "context is a singleton -- closing an env and building "
                         "another in the same process hangs forever (observed: "
                         "21 min at 0%% GPU before being killed). The caller "
                         "loops by re-running the container.")
parser.add_argument("--steps", type=int, default=300,
                    help="timed steps per env count, after warmup")
parser.add_argument("--warmup", type=int, default=50,
                    help="untimed steps first -- CUDA kernels compile on the "
                         "first few and would otherwise be charged to the run")
parser.add_argument("--checkpoint", type=str,
                    default="week3/checkpoints/stage16_final")
parser.add_argument("--hold", type=int, default=5)
parser.add_argument("--out", type=str, default="bench_envs.txt")
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import subprocess
import time
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from stable_baselines3.common.vec_env import VecEnvWrapper

from stage17_cfg import FrankaLiftStage17Cfg


class StickyGripper(VecEnvWrapper):
    """Copy of the wrapper in train_stage17.py -- importing it there would
    re-run that module's argparse and AppLauncher at import time."""

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


def vram_gb():
    """Total GPU memory in use, in GB, as the driver sees it.

    NOT torch.cuda.max_memory_allocated() -- that counts only PyTorch's own
    allocations and reported 0.01 GB here, because the buffers that actually
    scale with num_envs belong to PhysX/Isaac, outside PyTorch. nvidia-smi is
    the only honest source.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], timeout=10)
        return int(out.decode().strip().splitlines()[0]) / 1024.0
    except Exception:
        return float("nan")


def bench(num_envs):
    """Return (steps_per_sec, vram_gb) or raise if the GPU can't take it."""
    cfg = FrankaLiftStage17Cfg()
    cfg.scene.num_envs = num_envs

    env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=cfg)
    env = Sb3VecEnvWrapper(env)
    env = StickyGripper(env, hold=args.hold)

    model = PPO.load(args.checkpoint, env=env)

    obs = env.reset()
    for _ in range(args.warmup):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _ = env.step(action)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _ = env.step(action)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Sampled while the scene is still alive -- tearing it down first would
    # measure nothing. The process exits right after, which frees it.
    peak = vram_gb()

    # Every env advances on every step, so a step is num_envs samples.
    return args.steps * num_envs / elapsed, peak


print("\n[bench] === num_envs = %d ===" % args.envs, flush=True)
try:
    rate, peak = bench(args.envs)
    eta_min = 10_000_000 / rate / 60.0
    line = ("%8d %12.0f %10.2f %11.1f min  ok"
            % (args.envs, rate, peak, eta_min))
    print("[bench] %d envs: %.0f steps/s, %.2f GB in use, 10M would take %.1f min"
          % (args.envs, rate, peak, eta_min), flush=True)
except Exception as exc:
    # An OOM is the answer we are looking for, not a crash. Record it.
    msg = "OOM" if "out of memory" in str(exc).lower() else type(exc).__name__
    line = "%8d %12s %10s %14s  %s" % (args.envs, "-", "-", "-", msg)
    print("[bench] %d envs FAILED: %s" % (args.envs, msg), flush=True)

# Append, so a loop of container runs accumulates one table.
with open(args.out, "a") as fh:
    fh.write(line + "\n")
print("\n" + line)

simulation_app.close()
