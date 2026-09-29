"""Compare a character LoRA's saved epochs against no LoRA on unseen prompts; writes a labelled grid.

    comfyui\\python_embeded\\python.exe scripts\\test_lora.py sumi
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from comfy_api import Comfy, load_workflow, ROOT  # noqa: E402
from build_sdxl_workflows import QUALITY, STYLE  # noqa: E402

LORA_DIR = ROOT / "comfyui" / "ComfyUI" / "models" / "loras"


def lora_graph(wf, lora_name, strength=1.0):
    """Insert a LoraLoader between the checkpoint and the sampler / text encoders."""
    wf["20"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["1", 0], "clip": ["1", 1], "lora_name": lora_name,
        "strength_model": strength, "strength_clip": strength}}
    wf["5"]["inputs"]["model"] = ["20", 0]
    wf["2"]["inputs"]["clip"] = ["20", 1]
    wf["3"]["inputs"]["clip"] = ["20", 1]
    return wf


def main(name):
    sheet = json.loads((ROOT / "project" / "characters" / name / "character.json").read_text(encoding="utf-8"))
    trig = sheet["trigger"]
    # Unseen situations. Identity tags (hair/eyes) are deliberately left out: the trigger must supply them.
    prompts = {
        "cafe": f"{trig}, 1girl, solo, upper body, light smile, sitting in cafe, holding cup, window, daytime",
        "run": f"{trig}, 1girl, solo, full body, running, green jacket, city street, night, neon lights",
        "casual": f"{trig}, 1girl, solo, cowboy shot, white t-shirt, blue jeans, park, cherry blossoms",
        "fight": f"{trig}, 1girl, solo, holding katana, fighting stance, rain, night, rooftop, bandages",
    }
    loras = sorted(p.name for p in LORA_DIR.glob(f"{name}_v1*.safetensors"))
    columns = ["none"] + loras
    c = Comfy()
    grid = {}
    for pk, text in prompts.items():
        for col in columns:
            wf = load_workflow("sdxl_txt2img")
            wf["2"]["inputs"]["text"] = f"{QUALITY}, {STYLE}, {text}"
            wf["4"]["inputs"].update(width=832, height=1216)
            wf["5"]["inputs"]["seed"] = 4242
            wf["7"]["inputs"]["filename_prefix"] = f"lora_test/{name}_{pk}_{col.replace('.safetensors', '')}"
            if col != "none":
                wf = lora_graph(wf, col)
            paths, secs = c.run(wf)
            grid[(pk, col)] = paths[0]
            print(pk, col, secs)

    cell_w, cell_h, head = 280, 410, 40
    img = Image.new("RGB", (cell_w * len(columns), head + cell_h * len(prompts)), (20, 20, 24))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for ci, col in enumerate(columns):
        label = "no LoRA" if col == "none" else col.replace(f"{name}_v1", "").replace(".safetensors", "").strip("-") or "final"
        label = label.lstrip("0") and ("epoch " + label.lstrip("0")) if label.isdigit() else label
        d.text((ci * cell_w + 10, 10), label, fill=(255, 220, 0), font=font)
        for ri, pk in enumerate(prompts):
            t = Image.open(grid[(pk, col)]).convert("RGB").resize((cell_w - 4, cell_h - 4))
            img.paste(t, (ci * cell_w + 2, head + ri * cell_h + 2))
    out = ROOT / "project" / "characters" / name / "lora_test_grid.jpg"
    img.save(out, quality=88)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
