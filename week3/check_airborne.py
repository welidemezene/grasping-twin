"""Is the cube actually OFF THE TABLE, or just tipped onto a corner?

curriculum_lift_cfg.py warns about exactly this: a 4.2 cm cube resting flat has
its centre at 0.0210, but tipped onto a corner that becomes 0.0210*sqrt(3) =
0.0364 -- which clears the 0.035 height gate while the cube is still sitting on
the table. Stage 15's cube z hovers between 0.033 and 0.038, straddling that
number, and the replay looks wrong to the eye.

Height alone cannot tell these apart. The cube's ORIENTATION can. This replays a
checkpoint, records the cube's full pose, and computes the height of its LOWEST
CORNER:

    lowest corner ~ 0.000  ->  still on the table (tip exploit, not a lift)
    lowest corner >  0.005  ->  genuinely airborne

Also reports the tilt of the cube's up-axis: ~0 deg is flat, ~54.7 deg is
balanced corner-down.

Usage: python week3/check_airborne.py <checkpoint> [out_prefix] [hold]
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import csv, math, sys, torch, numpy as np, gymnasium as gym
from stable_baselines3 import PPO
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3.common.vec_env import VecEnvWrapper
from curriculum_lift_cfg import FrankaLiftStage1Cfg

HALF = 0.021          # half the cube edge
TABLE_Z = 0.0         # cube z is reported relative to the table top


class StickyGripper(VecEnvWrapper):
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
        obs, r, d, i = self.venv.step_wait()
        if np.any(d):
            self.count[d] = 0
            self.held[d] = 1.0
        return obs, r, d, i

    def reset(self):
        self.count[:] = 0
        self.held[:] = 1.0
        return self.venv.reset()


def quat_to_R(q):
    """(w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


CKPT   = sys.argv[1] if len(sys.argv) > 1 else "week3/checkpoints/stage15_final"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "airborne_s15"
HOLD   = int(sys.argv[3]) if len(sys.argv) > 3 else 5

cfg = FrankaLiftStage1Cfg()
cfg.scene.num_envs = 512
env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=cfg)
raw = env.unwrapped
vec = StickyGripper(Sb3VecEnvWrapper(env), hold=HOLD)
model = PPO.load(CKPT, env=vec)

N = raw.num_envs
# per-env success flags, accumulated over the whole episode across ALL envs --
# 512 independent trials per run instead of 1, which is the only honest way to
# ask "how often does this work" rather than "did it work that one time"
ever_grasped = torch.zeros(N, dtype=torch.bool)
ever_airborne = torch.zeros(N, dtype=torch.bool)
ever_both = torch.zeros(N, dtype=torch.bool)
best_corner = torch.full((N,), -1.0)

rows = []
obs = vec.reset()
for step in range(250):
    a, _ = model.predict(obs, deterministic=True)
    obs, rew, done, info = vec.step(a)

    cube = raw.scene["object"]
    robot = raw.scene["robot"]
    eef = raw.scene["ee_frame"]

    pos = cube.data.root_pos_w            # (N,3)
    quat = cube.data.root_quat_w          # (N,4) w,x,y,z
    w, x, y, z_ = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # third row of the rotation matrix is all we need for the lowest corner
    r20 = 2 * (x * z_ - w * y)
    r21 = 2 * (y * z_ + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    drop = (r20.abs() + r21.abs() + r22.abs()) * HALF
    lowest_all = (pos[:, 2] - drop).cpu()

    fsum_all = robot.data.joint_pos[:, -2:].sum(dim=1).cpu()
    gap_all = torch.norm(pos - eef.data.target_pos_w[:, 0, :], dim=1).cpu()

    g = ((fsum_all - 0.042).abs() < 0.012) & (gap_all < 0.03)
    a_ = lowest_all > 0.005
    ever_grasped |= g
    ever_airborne |= a_
    ever_both |= (g & a_)
    best_corner = torch.maximum(best_corner, torch.where(g, lowest_all, torch.tensor(-1.0)))

    # env 0 trace, for the per-frame CSV
    cz = float(pos[0, 2])
    R = quat_to_R(quat[0])
    lowest = float(lowest_all[0])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    tilt = min(tilt, 180.0 - tilt)
    rows.append({"step": step, "cube_z": round(cz, 5), "lowest_corner": round(lowest, 5),
                 "tilt_deg": round(tilt, 1), "finger_sum": round(float(fsum_all[0]), 5),
                 "gap": round(float(gap_all[0]), 5)})
    if bool(done[0]):
        rows.pop()
        break

SUCCESS = int(ever_both.sum())
ok = best_corner[best_corner > 0]
reliability = [
    "",
    "=== RELIABILITY ACROSS %d PARALLEL ENVIRONMENTS (512 trials, not 1) ===" % N,
    "  ever grasped:              %4d / %d  (%.1f%%)" % (int(ever_grasped.sum()), N, 100*float(ever_grasped.sum())/N),
    "  ever airborne:             %4d / %d  (%.1f%%)" % (int(ever_airborne.sum()), N, 100*float(ever_airborne.sum())/N),
    "  GRASPED AND AIRBORNE:      %4d / %d  (%.1f%%)   <- the real success rate" % (SUCCESS, N, 100*float(SUCCESS)/N),
]
if len(ok):
    reliability.append("  lift height while held: median %.4f m, max %.4f m"
                       % (float(ok.median()), float(ok.max())))
reliability.append("")

with open(f"{PREFIX}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

air = [r for r in rows if r["lowest_corner"] > 0.005]
grasped = [r for r in rows if abs(r["finger_sum"] - 0.042) < 0.012 and r["gap"] < 0.03]
both = [r for r in grasped if r["lowest_corner"] > 0.005]

out = reliability + [f"--- env 0 detail ---",
       f"checkpoint {CKPT}   frames {len(rows)}",
       f"cube half-edge {HALF}; resting flat centre = {HALF:.4f}, "
       f"balanced on a corner = {HALF*math.sqrt(3):.4f}",
       "",
       f"cube z:          min {min(r['cube_z'] for r in rows):.4f}  max {max(r['cube_z'] for r in rows):.4f}",
       f"lowest corner:   min {min(r['lowest_corner'] for r in rows):.4f}  max {max(r['lowest_corner'] for r in rows):.4f}",
       f"tilt:            min {min(r['tilt_deg'] for r in rows):.1f} deg  max {max(r['tilt_deg'] for r in rows):.1f} deg",
       "",
       f"frames AIRBORNE (lowest corner > 5 mm):        {len(air)}",
       f"frames GRASPED  (fingers on cube, gap < 3 cm): {len(grasped)}",
       f"frames GRASPED **AND** AIRBORNE:               {len(both)}",
       ""]

if both:
    out.append(f"REAL LIFT: {len(both)} frames with the cube held AND off the table, "
               f"first at step {both[0]['step']} (corner {both[0]['lowest_corner']:.4f} m up)")
elif air:
    out.append("AIRBORNE BUT NOT GRASPED: the cube leaves the table without the "
               "fingers on it — batted or flicked, not carried.")
elif grasped:
    mt = max(r["tilt_deg"] for r in grasped)
    out.append("NO REAL LIFT. The fingers are on the cube but its lowest corner never "
               "leaves the table.")
    if mt > 30:
        out.append(f"  Cube tilts up to {mt:.1f} deg — it is being TIPPED ONTO A CORNER, "
                   f"which raises the centre toward {HALF*math.sqrt(3):.4f} and clears the "
                   f"0.035 height gate WITHOUT a lift. This is the exploit "
                   f"curriculum_lift_cfg.py warned about.")
    else:
        out.append(f"  Cube stays fairly flat (max tilt {mt:.1f} deg); the height gate is "
                   f"being cleared by something other than a tip — check the CSV.")
else:
    out.append("NEITHER: no grasp and no lift in this replay.")

text = "\n".join(out)
open(f"{PREFIX}_summary.txt", "w").write(text + "\n")
print(text)

vec.close()
simulation_app.close()
