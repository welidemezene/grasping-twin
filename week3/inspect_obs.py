"""What does the policy actually see? Decides whether this could ever run on real hardware."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import isaaclab_tasks  # noqa: F401
from curriculum_lift_cfg import FrankaLiftStage1Cfg
c = FrankaLiftStage1Cfg()
out = ["--- OBSERVATIONS ---"]
for group_name, group in c.observations.__dict__.items():
    if hasattr(group, "__dict__"):
        out.append("group: %s" % group_name)
        for n, t in group.__dict__.items():
            if hasattr(t, "func"):
                out.append("   %-38s %s" % (n, t.func.__name__))
out.append("--- CUBE SPAWN RANDOMISATION ---")
out.append(str(c.events.reset_object_position.params.get("pose_range")))
open("obs_report.txt", "w").write("\n".join(out) + "\n")
app.close()
