"""Stage 17 config: make "held" mean actually held.

Stage 16 lifts the cube 18 cm, 99.6% of the time — and does it by hooking a
corner. Measured during the lift:

    cube centre vs fingertip centre   28.4 mm above, 2.8 mm sideways
    cube half-width                   21.0 mm
    cube tilt                         mean 39.8 deg, never below 28.6

The cube's centre is further from the fingertips than the cube's own half-width,
so it is not between the fingers at all — it hangs off them, tilted at a
permanent 40 degrees. It lifts anyway because friction does not care where you
squeeze: two fingers pinching any part of a cube will hold it against gravity.

The cause is one number I chose. `_held` accepts anything within HOLD_DISTANCE
of the fingertips, and HOLD_DISTANCE was 0.03. The corner pinch sits at 0.0286
— inside the gate by 1.4 mm. So a corner pinch and a proper grip scored
identically, and the pinch is easier to find, so that is what the policy found.

The robot did exactly what the reward said. The reward said it sloppily. This is
the same failure as the original week 3 reward hacking, one level finer: there,
"height" could be earned without a grasp; here, "held" can be earned without a
grip.

Two changes:

  1. HOLD_DISTANCE 0.03 -> 0.015, applied to EVERY term that gates on it. Half
     the cube's width, so the cube must genuinely sit between the fingers. This
     is the load-bearing change — it redefines what the robot is being paid for.
  2. cube_upright weight 4 -> 10. Stage 16 cut the tilt from 79 to 42 degrees
     with weight 4, which was deliberately timid so squareness could never
     outweigh lifting. 40 degrees costing almost nothing is why it stayed at 40.

Lifting still outweighs squareness (12 vs 10), so lifting badly continues to
beat not lifting — the robot must never be better off leaving the cube alone.
"""

from isaaclab.utils import configclass

from stage16_cfg import FrankaLiftStage16Cfg

TIGHT_HOLD = 0.015   # half the cube's width: it has to be IN the gripper


@configclass
class FrankaLiftStage17Cfg(FrankaLiftStage16Cfg):
    """Stage 16's rewards with an honest definition of 'held'."""

    def __post_init__(self):
        super().__post_init__()

        # Tighten every gate that asks "is the cube held?" -- lifting_object,
        # object_goal_tracking, object_goal_tracking_fine_grained,
        # lifting_progress and cube_upright all take hold_distance, and leaving
        # any of them loose would keep paying for the pinch through that term.
        tightened = []
        for name in dir(self.rewards):
            if name.startswith("_"):
                continue
            term = getattr(self.rewards, name, None)
            params = getattr(term, "params", None)
            if isinstance(params, dict) and "hold_distance" in params:
                params["hold_distance"] = TIGHT_HOLD
                tightened.append(name)
        print("[stage17] hold_distance -> %.3f on: %s" % (TIGHT_HOLD, ", ".join(sorted(tightened))))

        # Make a crooked carry actually cost something.
        self.rewards.cube_upright.weight = 10.0
