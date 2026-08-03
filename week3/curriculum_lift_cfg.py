"""Stage 10: decoupled channels. Approach always pays; closing pays only on the cube.

Stages 1-9 taught one lesson the hard way: the arm and the hand cannot share a
multiplied reward term. Multiplying approach by hand-correctness made proximity
punishable, and every long run retreated to a 2-3 cm hover. Here the two
channels are separate, monotonic terms — approach_object pulls the arm in
regardless of the hand, fingers_on_cube pays the hand only where a close is a
grasp — and the height terms still pay only for a genuine hold.
"""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)

import grasp_reward

# a cube resting flat has its centre at 0.0210 m; tipped onto a corner that
# becomes 0.0210 * sqrt(3) = 0.0364, which clears a bare 0.035 height check —
# so height alone is never enough to call something a lift
LIFT_HEIGHT = 0.035
# a real grasp puts the cube centre ~0-1.5 cm from the TCP; 3 cm gives slack
# without letting "held" trigger from outside the hand (the old 5 cm did)
HOLD_DISTANCE = 0.03


@configclass
class FrankaLiftStage1Cfg(FrankaCubeLiftEnvCfg):
    """reach (breadcrumb) -> approach (always pays) -> close ON the cube -> LIFT."""

    def __post_init__(self):
        super().__post_init__()

        # KILL the stock curriculum. The base lift task silently multiplies the
        # action_rate and joint_vel penalties by 1000x (-1e-4 -> -1e-1) after
        # 10,000 per-env steps = ~5.1M total steps at 512 envs. Stage 10 and 11
        # both collapsed from ep_rew ~19.6 to ~0.3 at exactly that step, and it
        # is why every 10M final in this project froze far from the cube: the
        # second half of every long run was trained under a movement tax that
        # punished the micro-adjustments parking needs and the finger flips
        # closing needs. Penalties stay at their gentle base weights instead.
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None

        # Cube spawns in the SAME spot every episode (randomization is a later
        # stage — first the grasp itself has to exist)
        self.events.reset_object_position.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }

        # Reaching stays a small breadcrumb
        self.rewards.reaching_object.weight = 2.0

        # Channel 1 — the ARM: smooth pull to the cube, hand state ignored.
        # Replaces the old approach*hand product whose flat top parked the arm
        # at 2-3 cm.
        self.rewards.grasping_object = RewTerm(
            func=grasp_reward.approach_object,
            weight=4.0,
        )

        # Channel 2 — the HAND: pays only when the fingers close on the cube
        # itself (position-gated bump peaking at the cube-stop point). This is
        # the term that finally makes "close" the winning action somewhere.
        self.rewards.fingers_on_cube = RewTerm(
            func=grasp_reward.fingers_on_cube,
            params={"in_position_std": 0.015},
            weight=6.0,
        )

        # The height-gated prizes, worth 36 between them: unchanged in spirit,
        # but "held" now requires fingers stopped BY the cube, so neither a
        # batted cube nor an empty fist beside it can collect.
        self.rewards.lifting_object = RewTerm(
            func=grasp_reward.object_lifted_in_hand,
            params={"minimal_height": LIFT_HEIGHT, "hold_distance": HOLD_DISTANCE},
            weight=15.0,
        )
        self.rewards.object_goal_tracking = RewTerm(
            func=grasp_reward.object_goal_distance_in_hand,
            params={
                "std": 0.3,
                "minimal_height": LIFT_HEIGHT,
                "command_name": "object_pose",
                "hold_distance": HOLD_DISTANCE,
            },
            weight=16.0,
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=grasp_reward.object_goal_distance_in_hand,
            params={
                "std": 0.05,
                "minimal_height": LIFT_HEIGHT,
                "command_name": "object_pose",
                "hold_distance": HOLD_DISTANCE,
            },
            weight=5.0,
        )
