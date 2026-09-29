"""Warm-model SDXL benchmark: runs txt2img 3x, reports each time (first includes model load)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from comfy_api import Comfy, load_workflow  # noqa: E402

label = sys.argv[1] if len(sys.argv) > 1 else "run"
c = Comfy()
times = []
for i in range(3):
    wf = load_workflow("sdxl_txt2img")
    wf["5"]["inputs"]["seed"] = 1000 + i
    wf["7"]["inputs"]["filename_prefix"] = f"bench/{label}"
    _, secs = c.run(wf)
    times.append(secs)
print(label, "times:", times, "warm avg:", round(sum(times[1:]) / 2, 1), "stats:", c.stats())
