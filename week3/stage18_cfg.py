"""Stage 18 config: the middle of the bracket, and stop paying to stay low.

Stage 17 changed two things at once and both mattered, in opposite directions.
The grip fix WORKED and the lift DIED:

                          stage 16      stage 17
    cube off fingertips    31.5 mm      12.7 mm   <- fixed (half-width is 21)
    lift height, mean     140.9 mm      31.6 mm   <- destroyed
    tilt while carried       37.5 deg   72.0 deg  <- worse, despite weight 10
    grasped AND airborne     99.6%      97.7%

Two separate causes, one per change.

CHANGE 1 -- hold_distance 0.015 -> 0.02.

`_held` is a BOOLEAN and five reward terms multiply by it (lifting_object,
object_goal_tracking, object_goal_tracking_fine_grained, lifting_progress,
cube_upright). It is not shaping, it is an on/off switch on most of the reward.

It gates on cube-centre to fingertip-centre distance. Stage 17 measured a
proper grip at 12.7 mm against a 15 mm gate -- 2.3 mm of headroom on a quantity
that moves every frame as the cube shifts in the fingers. So the switch
flickered, and every frame it was off the policy earned nothing no matter how
well it was doing. That is a cliff, the same class of bug as the old
`object_is_lifted` step function at 0.035 that stage 15 had to remove.

0.03 was too loose (the 0.0286 corner pinch passed). 0.015 is too tight. 0.02
is the untried middle: 7.3 mm of margin above a real grip, still 10 mm below
the pinch it must exclude.

CHANGE 2 -- cube_upright weight 10 -> 4, back to stage 16's value.

Stage 17 raised this to 10 reasoning "lifting is 12, squareness is 10, so
lifting still wins". That comparison does not hold, because the two are not
alternatives at the same height:

    lifting_progress_in_hand  weight 12   tanh(h/...) -> ~0 at rest
    cube_upright_in_hand      weight 10   full value at h = 0

At rest the policy banks a GUARANTEED 10 for holding the cube square on the
table. Lifting means gambling that 10 -- height tilts the cube and shakes it
out of the hold gate -- for at most 12. Under a flickering gate that is a bad
bet, so it took the sure thing and hovered at 31.6 mm.

This also explains the result that otherwise looks contradictory: tilt got
WORSE while the squareness weight was tripled. cube_upright is gated by _held
too, so in the few frames the cube was actually airborne the gate was off and
squareness paid nothing there. Stage 17 paid heavily for uprightness exactly
where the cube was already sitting still, and not at all where it was wanted.

So: fix the gate, and put squareness back to the timid weight that already cut
tilt 79 -> 42 deg in stage 16. Tilt is the third priority. Get the real grip
AND the lift first; square it up afterwards if it still matters.
"""

from isaaclab.utils import configclass

from stage16_cfg import FrankaLiftStage16Cfg

MID_HOLD = 0.02      # 0.03 passed the corner pinch, 0.015 flickered off a real grip
UPRIGHT_WEIGHT = 4.0  # stage 16's value -- must not out-earn lifting at rest


@configclass
class FrankaLiftStage18Cfg(FrankaLiftStage16Cfg):
    """Stage 16's rewards with a hold gate that is honest but not brittle."""

    def __post_init__(self):
        super().__post_init__()

        # Every term that asks "is the cube held?" must use the same number --
        # leaving any one loose keeps paying for the pinch through that term.
        tightened = []
        for name in dir(self.rewards):
            if name.startswith("_"):
                continue
            term = getattr(self.rewards, name, None)
            params = getattr(term, "params", None)
            if isinstance(params, dict) and "hold_distance" in params:
                params["hold_distance"] = MID_HOLD
                tightened.append(name)
        print("[stage18] hold_distance -> %.3f on: %s"
              % (MID_HOLD, ", ".join(sorted(tightened))))

        self.rewards.cube_upright.weight = UPRIGHT_WEIGHT
        print("[stage18] cube_upright weight -> %.1f (stage 17 used 10.0)"
              % UPRIGHT_WEIGHT)
