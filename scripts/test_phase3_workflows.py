"""Phase 3 test for the remaining workflows: pose control, turnaround sheet, refine + upscale."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from comfy_api import Comfy, load_workflow, ROOT, INPUT_DIR  # noqa: E402
from build_sdxl_workflows import QUALITY, STYLE  # noqa: E402
from test_phase3 import CONCEPTS, Peak  # noqa: E402

OUT = ROOT / "project" / "_comfy_output" / "test_phase3"
c = Comfy()
results = []


def run(name, wf):
    with Peak() as p:
        paths, secs = c.run(wf)
    r = {"job": name, "seconds": secs, "peak_vram_mb": p.vram_mb, "peak_ram_used_gb": round(p.ram_used_gb, 1),
         "files": [str(x) for x in paths]}
    print(json.dumps(r))
    results.append(r)
    return paths


# 1) Pose reference: a generic full-body action pose, then re-draw concept C in that exact pose.
wf = load_workflow("sdxl_txt2img")
wf["2"]["inputs"]["text"] = f"{QUALITY}, {STYLE}, 1boy, solo, full body, dynamic pose, fighting stance, lunging, simple background"
wf["5"]["inputs"]["seed"] = 777
wf["7"]["inputs"]["filename_prefix"] = "test_phase3/pose_source"
src = run("pose_source", wf)[0]
shutil.copy(src, INPUT_DIR / "pose_ref.png")

wf = load_workflow("sdxl_pose")
wf["2"]["inputs"]["text"] = f"{QUALITY}, {STYLE}, {CONCEPTS['C_gravity_mechanic'].replace('upper body', 'full body')}, simple background"
wf["5"]["inputs"]["seed"] = 20260928
wf["7"]["inputs"]["filename_prefix"] = "test_phase3/pose_C"
wf["15"]["inputs"]["filename_prefix"] = "test_phase3/pose_skeleton"
run("pose_controlled_C", wf)

# 2) Turnaround sheet for concept B
wf = load_workflow("sdxl_turnaround")
desc = CONCEPTS["B_ink_swordswoman"].replace("upper body, ", "").replace("black ink splashes, shadow aura, ", "")
wf["2"]["inputs"]["text"] = (f"{QUALITY}, {STYLE}, character reference sheet, multiple views, front view, side view, "
                             f"back view, full body, standing, white background, simple background, {desc}")
wf["5"]["inputs"]["seed"] = 20260928
wf["7"]["inputs"]["filename_prefix"] = "test_phase3/turnaround_B"
run("turnaround_B", wf)

# 3) Refine + upscale concept A (832x1216 -> 1248x1824)
shutil.copy(OUT / "A_storm_courier_00001_.png", INPUT_DIR / "refine_in.png")
wf = load_workflow("sdxl_refine_upscale")
wf["2"]["inputs"]["text"] = f"{QUALITY}, {STYLE}, {CONCEPTS['A_storm_courier']}"
wf["5"]["inputs"]["seed"] = 20260928
wf["7"]["inputs"]["filename_prefix"] = "test_phase3/refined_A"
run("refine_upscale_A", wf)

(OUT / "results_workflows.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
