"""Rewards that pay for closing on the cube — and only count a lift that is held.

Two lessons are baked in here, both found by replaying checkpoints rather than
reading reward curves:

1. Every term is smooth. An earlier version flipped at exactly 3 cm, which halved
   the pay the instant the gripper arrived, so the policy learned to hover just
   outside the grasp zone instead of entering it.
2. Height alone is not a lift. The stock lifting and goal terms pay whenever the
   centre of the cube passes a height, so batting the cube over pays the biggest
   prize in the environment (weight 15 + 16) with the hand wide open. Here that
   height only counts while the cube is actually in a closed gripper.
"""

from __future__ import annotations
import torch
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp

# fingers are commanded open (0.04 each) or shut (0.0); a cube this size stops
# them near 0.042 total, so 0.06 separates "shut on something" from "open"
CLOSED_SUM = 0.06
OPEN_SUM = 0.08


def _held(
    env: ManagerBasedRLEnv,
    hold_distance: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """True where the cube is at the gripper AND the fingers are shut on it."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    near = torch.norm(obj.data.root_pos_w - ee_pos, dim=1) < hold_distance
    shut = robot.data.joint_pos[:, -2:].sum(dim=1) < CLOSED_SUM
    return near & shut


def object_is_grasped(
    env: ManagerBasedRLEnv,
    grasp_distance: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Grows as the gripper closes on the cube, and grows further when the fingers
    do what that distance asks for: open on the way in, shut at the cube."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w - ee_pos, dim=1)

    # coarse pulls it across the table; fine pays for the last few centimetres,
    # where the coarse ramp has already flattened out
    coarse = 1.0 - torch.tanh(distance / 0.25)
    fine = 1.0 - torch.tanh(distance / 0.03)
    approach = 0.6 * coarse + 0.4 * fine

    # How far the fingers have travelled from wide open to shut, as 0..1.
    # A threshold here pays nothing for fingers that move 8 mm, and measurement
    # shows they need three consecutive close commands to cross any threshold —
    # which independent per-step exploration effectively never samples. Continuous
    # means one close command already pays slightly more than none.
    closed_frac = (1.0 - robot.data.joint_pos[:, -2:].sum(dim=1) / OPEN_SUM).clamp(0.0, 1.0)
    # Crossover near 1.65 cm — the distance at which the cube (0.042 m wide) is
    # actually between the fingers (0.08 m span). This was 0.08, i.e. told to
    # close from 4.4 cm, set back when the arm could only reach 2.4 cm. The arm
    # then learned to reach 0.3 cm, and the stale value made it shut its fist
    # 5.7 cm out and jam against the cube it could no longer enclose. A reward
    # parameter encodes an assumption about the current policy; when the policy
    # improves, recheck it.
    want_closed = 1.0 - torch.tanh(distance / 0.03)
    hand_ok = 1.0 - (closed_frac - want_closed).abs()

    # The hand term was 0.75 + 0.25 * hand_ok, which paid +0.10 per step for a
    # first close — too small to compete, and the gripper output drifted further
    # open across a whole 10M run. At 0.5 it pays +0.20 for the first close and
    # +1.80 for a full grasp, while approach stays monotonic (checked at every
    # distance the arm visits, open hand).
    return approach * (0.5 + 0.5 * hand_ok)


def object_lifted_in_hand(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_distance: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The stock height check, but only while the cube is held."""
    lifted = lift_mdp.object_is_lifted(env, minimal_height=minimal_height, object_cfg=object_cfg)
    return lifted * _held(env, hold_distance, object_cfg, ee_frame_cfg, robot_cfg).float()


def object_goal_distance_in_hand(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    hold_distance: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Carrying the cube toward the goal, counted only while it is held."""
    tracking = lift_mdp.object_goal_distance(
        env, std=std, minimal_height=minimal_height, command_name=command_name,
        robot_cfg=robot_cfg, object_cfg=object_cfg,
    )
    return tracking * _held(env, hold_distance, object_cfg, ee_frame_cfg, robot_cfg).float()
