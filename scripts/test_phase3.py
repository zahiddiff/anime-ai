"""Phase 3 smoke test: 3 original character concepts + 1 background, with timing and peak VRAM/RAM.

Run with ComfyUI already started (run_comfyui.bat):
    comfyui\\python_embeded\\python.exe scripts\\test_phase3.py
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from comfy_api import Comfy, load_workflow, ROOT  # noqa: E402
from build_sdxl_workflows import QUALITY, STYLE  # noqa: E402

CONCEPTS = {
    "A_storm_courier": "1boy, solo, spiky white hair, blue streaked hair, yellow eyes, sharp eyes, "
                       "black high-collar courier jacket, orange jacket lining, fingerless gloves, "
                       "holding tonfa, electricity, blue lightning, upper body, determined, clenched teeth",
    "B_ink_swordswoman": "1girl, solo, short hair, ash pink hair, bob cut, green eyes, bandaged arm, "
                         "oversized olive bomber jacket, black turtleneck, katana wrapped in cloth, "
                         "black ink splashes, shadow aura, upper body, serious expression",
    "C_gravity_mechanic": "1boy, solo, messy dark red hair, goggles on head, heterochromia, orange jumpsuit tied at waist, "
                          "black tank top, holding giant wrench, floating debris, floating rocks, upper body, grin",
}
BACKGROUND = "ruined rooftop at dusk, city skyline, broken water tower, orange sky, long shadows, power lines"


class Peak:
    """Samples nvidia-smi and free RAM while a job runs."""

    def __init__(self):
        self.vram_mb = 0
        self.ram_used_gb = 0.0
        self._stop = False

    def _loop(self):
        while not self._stop:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
                self.vram_mb = max(self.vram_mb, int(out.strip().splitlines()[0]))
                ps = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command",
                     "$o=Get-CimInstance Win32_OperatingSystem; ($o.TotalVisibleMemorySize-$o.FreePhysicalMemory)/1MB"],
                    text=True)
                self.ram_used_gb = max(self.ram_used_gb, float(ps.strip().replace(",", ".")))
            except Exception:
                pass
            time.sleep(0.5)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join()


def main():
    c = Comfy()
    if not c.alive():
        sys.exit("ComfyUI is not running - start run_comfyui.bat first")
    print("before:", c.stats())
    results = []

    jobs = []
    for name, desc in CONCEPTS.items():
        wf = load_workflow("sdxl_txt2img")
        wf["2"]["inputs"]["text"] = f"{QUALITY}, {STYLE}, {desc}"
        wf["5"]["inputs"]["seed"] = 20260928
        wf["7"]["inputs"]["filename_prefix"] = f"test_phase3/{name}"
        jobs.append((name, wf))
    wf = load_workflow("sdxl_background")
    wf["2"]["inputs"]["text"] = f"{QUALITY}, anime background, scenery, no humans, {BACKGROUND}, cinematic lighting, wide shot"
    wf["5"]["inputs"]["seed"] = 20260928
    wf["7"]["inputs"]["filename_prefix"] = "test_phase3/background_rooftop"
    jobs.append(("background_rooftop", wf))

    for name, wf in jobs:
        with Peak() as p:
            paths, secs = c.run(wf)
        r = {"job": name, "seconds": secs, "peak_vram_mb": p.vram_mb, "peak_ram_used_gb": round(p.ram_used_gb, 1),
             "files": [str(x) for x in paths]}
        print(json.dumps(r))
        results.append(r)

    log = ROOT / "project" / "_comfy_output" / "test_phase3" / "results.json"
    log.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("after:", c.stats())


if __name__ == "__main__":
    main()
