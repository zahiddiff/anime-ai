"""Builds the Phase 3 still-image workflows (ComfyUI API format) into workflows/*.api.json.

Defaults follow the NoobAI-XL 1.1 model card: Euler a, CFG 5-6, 25-30 steps, ~1 MP buckets.
Every script fills in the prompt / seed / size fields by node id (see PATCH_POINTS below).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / "workflows"

CKPT = "NoobAI-XL-v1.1.safetensors"
CONTROLNET = "xinsir-controlnet-union-sdxl-promax.safetensors"
UPSCALER = "4x-AnimeSharp.safetensors"  # swap to RealESRGAN_x4plus_anime_6B.pth for commercial work

QUALITY = "masterpiece, best quality, newest, absurdres, highres, safe"
NEGATIVE = ("nsfw, worst quality, old, early, low quality, lowres, signature, username, logo, "
            "bad hands, mutated hands, mammal, anthro, furry, ambiguous form, feral, semi-anthro, "
            "watermark, text, blurry, jpeg artifacts")
# House style for the modern shonen-action look
STYLE = "anime screencap, anime coloring, cel shading, bold lineart, dramatic lighting, dynamic angle"


def base_graph(prompt, width, height, prefix, steps=28, cfg=5.5, seed=0):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler_ancestral", "scheduler": "normal",
            "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "denoise": 1.0}},
        "6": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["5", 0], "vae": ["1", 2], "tile_size": 512, "overlap": 64,
            "temporal_size": 64, "temporal_overlap": 8}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }


def txt2img():
    return base_graph(f"{QUALITY}, {STYLE}, 1boy, solo, upper body, looking at viewer",
                      832, 1216, "keyframes/t2i")


def pose():
    g = base_graph(f"{QUALITY}, {STYLE}, 1boy, solo, full body, fighting stance",
                   832, 1216, "keyframes/pose")
    g.update({
        "10": {"class_type": "LoadImage", "inputs": {"image": "pose_ref.png"}},
        # DWPreprocessor returns blank maps on this setup (torch 2.13/cu130, 2026-09-29) - OpenPose works.
        "11": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["10", 0], "detect_hand": "enable", "detect_body": "enable", "detect_face": "enable",
            "resolution": 1024, "scale_stick_for_xinsr_cn": "enable"}},
        "12": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROLNET}},
        "13": {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["12", 0], "type": "openpose"}},
        "14": {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["2", 0], "negative": ["3", 0], "control_net": ["13", 0], "image": ["11", 0],
            "strength": 0.8, "start_percent": 0.0, "end_percent": 0.8, "vae": ["1", 2]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "keyframes/pose_skeleton"}},
    })
    g["5"]["inputs"]["positive"] = ["14", 0]
    g["5"]["inputs"]["negative"] = ["14", 1]
    return g


def turnaround():
    return base_graph(
        f"{QUALITY}, {STYLE}, character reference sheet, multiple views, front view, side view, back view, "
        "full body, standing, simple background, white background, 1boy, solo",
        1344, 768, "characters/turnaround", steps=30)


def background():
    g = base_graph(f"{QUALITY}, anime background, scenery, no humans, detailed background, cinematic lighting, "
                   "wide shot", 1344, 768, "backgrounds/bg", steps=30)
    g["3"]["inputs"]["text"] = NEGATIVE + ", 1girl, 1boy, person, character"
    return g


def refine_upscale():
    """Load an image -> 4x anime upscaler -> scale to 1.5x original -> low-denoise img2img -> save."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": f"{QUALITY}, {STYLE}", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "refine_in.png"}},
        "11": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}},
        "12": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["11", 0], "image": ["10", 0]}},
        "13": {"class_type": "ImageScaleBy", "inputs": {"image": ["12", 0], "upscale_method": "lanczos", "scale_by": 0.375}},
        "14": {"class_type": "VAEEncodeTiled", "inputs": {
            "pixels": ["13", 0], "vae": ["1", 2], "tile_size": 512, "overlap": 64,
            "temporal_size": 64, "temporal_overlap": 8}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": 0, "steps": 20, "cfg": 5.0,
            "sampler_name": "euler_ancestral", "scheduler": "normal",
            "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["14", 0], "denoise": 0.35}},
        "6": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["5", 0], "vae": ["1", 2], "tile_size": 512, "overlap": 64,
            "temporal_size": 64, "temporal_overlap": 8}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "keyframes/refined"}},
    }


# Node ids scripts patch: prompt text, negative, seed, width/height, input image, output prefix
PATCH_POINTS = {"prompt": "2", "negative": "3", "sampler": "5", "latent": "4", "save": "7", "load_image": "10"}

if __name__ == "__main__":
    WF.mkdir(exist_ok=True)
    for name, fn in [("sdxl_txt2img", txt2img), ("sdxl_pose", pose), ("sdxl_turnaround", turnaround),
                     ("sdxl_background", background), ("sdxl_refine_upscale", refine_upscale)]:
        (WF / f"{name}.api.json").write_text(json.dumps(fn(), indent=1), encoding="utf-8")
        print("wrote", name)
    (WF / "PATCH_POINTS.json").write_text(json.dumps(PATCH_POINTS, indent=1), encoding="utf-8")
