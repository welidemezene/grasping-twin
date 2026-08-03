from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import isaaclab_tasks  # noqa: F401
from curriculum_lift_cfg import FrankaLiftStage1Cfg
c = FrankaLiftStage1Cfg()
out = []
for n, t in c.rewards.__dict__.items():
    if hasattr(t, "func"):
        f = t.func
        out.append("%-34s %s.%s" % (n, getattr(f, "__module__", "?"), getattr(f, "__name__", "?")))
import inspect
try:
    from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
    out.append("IMPORT lift.mdp OK")
    out.append("has object_is_lifted: %s" % hasattr(lift_mdp, "object_is_lifted"))
    out.append("has object_goal_distance: %s" % hasattr(lift_mdp, "object_goal_distance"))
    out.append("sig object_is_lifted: %s" % str(inspect.signature(lift_mdp.object_is_lifted)))
    out.append("sig object_goal_distance: %s" % str(inspect.signature(lift_mdp.object_goal_distance)))
except Exception as e:
    out.append("IMPORT FAILED: %r" % (e,))
open("cfg_report.txt", "w").write("\n".join(out) + "\n")
app.close()
