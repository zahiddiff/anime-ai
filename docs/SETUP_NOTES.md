# Setup notes & benchmarks

Build log for AnimeAI on a single mid-range laptop. Numbers are measured; values marked `~` are scaled from a measurement.

## Test machine

| Component | Spec |
|---|---|
| Laptop | Acer Nitro V 16 (ANV16-71) |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM, 55 W default / 85 W boost |
| CPU | Intel Core i5-14450HX (10 cores / 16 threads) |
| RAM | 16 GB DDR5-5600, single channel |
| Storage | NVMe SSD, single drive |
| OS / driver | Windows 11, NVIDIA 596.49 (CUDA 13.2) |

**Bottlenecks:** 8 GB VRAM means models are quantised or offloaded. 16 GB single-channel RAM is the tighter limit, since offloaded weights live there. A second 16 GB stick (32 GB, dual channel) is the most useful upgrade.

---

## Engine: ComfyUI

- **ComfyUI v0.37.0** Windows portable (NVIDIA), embedded Python 3.13.14, PyTorch 2.13.0+cu130.
- ComfyUI-Manager 4.2.2 (`--enable-manager`).
- Portable build chosen over launcher-based installs for pinned versions, a self-contained folder and direct HTTP API access.

| Custom node | Commit | Purpose |
|---|---|---|
| ComfyUI-GGUF | 6ea2651 | quantised (GGUF) model loading |
| ComfyUI-VideoHelperSuite | 4d907be | video load/save |
| ComfyUI-KJNodes | d3cfe21 | utility nodes, block swap |
| ComfyUI-Frame-Interpolation | 26545cc | RIFE interpolation |
| comfyui_controlnet_aux | 0cd2904 | pose / depth / lineart preprocessors |

### Launch flags (NoobAI SDXL, 832×1216, 28 steps, warm model)

| Flags | s / image | Verdict |
|---|---|---|
| default | 33.7 | |
| `--fast fp16_accumulation` | 28.9 | ✅ used, no visible change on same seed |
| `--highvram --fast fp16_accumulation` | 29.4 | ❌ SDXL + ControlNet exceeds 8 GB, spills into shared memory, **225 s/step** |

`--lowvram` isn't needed; dynamic VRAM loading keeps the peak at 7.2–7.7 GB.

### Laptop thermal profile

The GPU was capped at **1185 MHz / ~20 W (P4, "SW thermal slowdown")** at only 58 °C. That's the vendor's quiet/balanced fan profile, not heat. Switching to the **Turbo** profile gave 2385 MHz, 55–91 W, P0, 85 °C peak.

| Benchmark | Quiet profile | Turbo |
|---|---|---|
| SDXL keyframe, 832×1216, 28 steps | 28.9 s | **18.4 s** |
| LoRA training | 4.7–5.2 s/step | **1.84 s/step** |

---

## Still images

| Model | Size | Licence |
|---|---|---|
| [NoobAI-XL v1.1](https://huggingface.co/Laxhar/noobai-XL-1.1) (eps-pred) | 7.1 GB | fair-ai-public-license-1.0-sd |
| [xinsir ControlNet Union SDXL ProMax](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) | 2.5 GB | Apache-2.0 |
| [4x-AnimeSharp](https://huggingface.co/Kim2091/AnimeSharp) | 33 MB | CC BY-NC-SA 4.0 |
| [RealESRGAN x4plus anime 6B](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.2.4) | 18 MB | BSD-3 |

**Sampling:** Euler a, CFG 5.5, 28 steps, ~1 MP buckets (832×1216 portrait, 1344×768 wide).
**Prompt prefix:** `masterpiece, best quality, newest, absurdres, highres, safe` + style `anime screencap, anime coloring, cel shading, bold lineart, dramatic lighting, dynamic angle`.

| Workflow | Time (Turbo) | Peak VRAM |
|---|---|---|
| `sdxl_txt2img` | 18.4 s | 7.5 GB |
| `sdxl_background` (1344×768) | ~25 s | 7.5 GB |
| `sdxl_turnaround` (1344×768) | ~25 s | 7.2 GB |
| `sdxl_pose` (OpenPose → ControlNet Union) | 51 s | 7.3 GB |
| `sdxl_refine_upscale` (4× → 1.5×, denoise 0.35) | 55 s | 7.7 GB |

Background and turnaround times are scaled from the pre-Turbo measurement at the same resolution.

**Known issues**
- `DWPreprocessor` returns empty pose maps on torch 2.13 / cu130 (all onnx and torchscript variants). `OpenposePreprocessor` with `scale_stick_for_xinsr_cn=enable` works and is used instead.
- Silhouette-style pose references produce no skeleton. Use clearly drawn figures.
- Base-model prompts drop small attributes (e.g. a cloth-wrapped sword). Character LoRAs fix this.

---

## Character LoRA

**Trainer:** [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) v0.12.0. Python 3.10.20 venv (uv), torch 2.8.0+cu128, bitsandbytes 0.50.2, diffusers 0.40, transformers 5.17, accelerate 1.15, plus onnx / onnxruntime-gpu for the tagger.
Chosen for the lowest SDXL VRAM footprint (fp8 base, cached text-encoder outputs) and headless scripting.

| Step | Detail |
|---|---|
| Dataset | 48 generated → 37 curated, varied framing / angle / expression / action / background |
| Captions | WD14 `wd-vit-tagger-v3` (general ≥ 0.35), trigger first, identity tags pruned so the trigger absorbs them |
| Network | LoRA dim 16 / alpha 8, UNet only |
| Precision | fp8 base, bf16 mixed, fp16 save |
| Optimiser | AdamW8bit, lr 1e-4, cosine, 50 warm-up steps, min-SNR γ 5 |
| Memory | batch 1, gradient checkpointing, SDPA, latents + text-encoder outputs cached to disk |
| Schedule | 10 epochs × 37 images × 4 repeats = 1,480 steps |
| Result | **46 min, 5.8 GB peak VRAM**. Epochs 2–6 unstable; epoch 8 and final reproduce the character reliably |

**Gotchas**
- `shuffle_caption` can't be combined with `cache_text_encoder_outputs`. Caching wins on VRAM.
- sd-scripts logs Japanese text; a redirected Windows console (cp1252) crashes. Set `PYTHONUTF8=1`.
- The tagger needs `onnx`, which isn't in `requirements.txt`.
- `uv venv --python 3.10` may fail with a "minor version link" error; `uv python install 3.10 --reinstall` fixes it.

---

## Disk budget

| Part | Size |
|---|---|
| ComfyUI + custom nodes + dependencies | 5.9 GB |
| Image models | 9.0 GB |
| sd-scripts venv | 8.0 GB |
| One character LoRA (5 checkpoints) | 0.4 GB |
| Video models (planned, Wan 2.2 5B + 14B GGUF) | ~40 GB |
