"""Generate LoRA training candidates for one character with varied pose / angle / framing / expression / background.

    comfyui\\python_embeded\\python.exe scripts\\gen_lora_dataset.py sumi 48

Writes to project/characters/<name>/candidates/. Off-model images are deleted before prep (keep 20-40).
The character sheet (fixed identity + outfit tags) lives in project/characters/<name>/character.json.
"""
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from comfy_api import Comfy, load_workflow, ROOT  # noqa: E402
from build_sdxl_workflows import QUALITY, STYLE  # noqa: E402

FRAMING = ["upper body", "cowboy shot", "full body", "portrait", "close-up", "cowboy shot", "full body"]
ANGLE = ["from front", "from side", "three-quarter view", "from above", "from below", "from behind, looking back",
         "dutch angle", "from front"]
EXPRESSION = ["serious", "light smile", "angry", "surprised", "closed mouth", "determined", "smirk", "tired",
              "expressionless", "shouting"]
ACTION = ["standing", "walking", "running", "unsheathing sword", "holding sword", "crossed arms", "sitting",
          "crouching", "fighting stance", "hand on hip", "looking at viewer", "jumping", "leaning against wall"]
BACKGROUND = ["simple background, white background", "simple background, grey background", "city street",
              "rooftop, night sky", "forest", "classroom", "night, rain", "dojo", "train station", "sunset, outdoors",
              "alley", "simple background, white background"]
SIZES = {"portrait": (832, 1216), "landscape": (1216, 832), "square": (1024, 1024)}


def main(name, count):
    char_dir = ROOT / "project" / "characters" / name
    sheet = json.loads((char_dir / "character.json").read_text(encoding="utf-8"))
    out = char_dir / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(sheet.get("seed", 1234))
    c = Comfy()
    log = []
    for i in range(count):
        framing = rng.choice(FRAMING)
        action = rng.choice(ACTION)
        extras = sheet["optional"] if rng.random() < 0.35 else ""
        parts = [QUALITY, STYLE, sheet["identity"], sheet["outfit"], framing, rng.choice(ANGLE),
                 rng.choice(EXPRESSION), action, rng.choice(BACKGROUND), extras]
        prompt = ", ".join(p for p in parts if p)
        w, h = SIZES["landscape" if action in ("running", "jumping") and rng.random() < .5 else
                     "square" if framing in ("portrait", "close-up") else "portrait"]
        wf = load_workflow("sdxl_txt2img")
        wf["2"]["inputs"]["text"] = prompt
        wf["4"]["inputs"].update(width=w, height=h)
        seed = rng.randrange(2**31)
        wf["5"]["inputs"]["seed"] = seed
        wf["7"]["inputs"]["filename_prefix"] = f"lora_{name}/cand"
        paths, secs = c.run(wf)
        dst = out / f"{name}_{i:03d}.png"
        shutil.copy(paths[0], dst)
        log.append({"file": dst.name, "seed": seed, "size": [w, h], "prompt": prompt, "seconds": secs})
        print(f"[{i + 1}/{count}] {dst.name} {secs}s")
    (out / "_prompts.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
