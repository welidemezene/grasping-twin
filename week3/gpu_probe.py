"""Does this container have GRAPHICS, or only COMPUTE?

Isaac Sim's renderer (and therefore any camera) needs a Vulkan device. The
NVIDIA container runtime only injects the graphics driver libraries when
NVIDIA_DRIVER_CAPABILITIES asks for them; the default is compute,utility.
That is why physics has always worked here while `enable_cameras=True` dies
with "Failed to create any GPU devices".

Prints what is actually present so the fix can be verified, not guessed.
"""

import glob
import os
import subprocess

print("NVIDIA_DRIVER_CAPABILITIES =", os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "(unset)"))
print("vulkan ICDs:", glob.glob("/usr/share/vulkan/icd.d/*") or "NONE")
print("libGLX_nvidia:", glob.glob("/usr/lib/x86_64-linux-gnu/libGLX_nvidia*") or "NONE")
print("libEGL_nvidia:", glob.glob("/usr/lib/x86_64-linux-gnu/libEGL_nvidia*") or "NONE")
print("wsl libs mounted:", glob.glob("/usr/lib/wsl/lib/*")[:4] or "NONE")

for tool in ("vulkaninfo", "nvidia-smi"):
    try:
        out = subprocess.run([tool, "--version"] if tool == "vulkaninfo" else [tool, "-L"],
                             capture_output=True, timeout=30)
        print("%s -> %s" % (tool, (out.stdout or out.stderr).decode().strip()[:200]))
    except Exception as exc:
        print("%s -> %s" % (tool, type(exc).__name__))
