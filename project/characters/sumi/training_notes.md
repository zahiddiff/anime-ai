# Sumi — LoRA training notes (v1, 2026-09-29)

**Trigger word:** `sumi_ink`
**Base model:** NoobAI-XL v1.1 (eps-pred, SDXL/Illustrious family)
**Trainer:** kohya-ss/sd-scripts v0.12.0 (`sdxl_train_network.py`), torch 2.8.0+cu128, Python 3.10.20 venv
**Output:** `lora/sumi_v1*.safetensors` (85 MB each, rank 16), copied to `comfyui/ComfyUI/models/loras/`

## Dataset
- 48 candidates generated with `scripts/gen_lora_dataset.py` (NoobAI + a fixed identity/outfit prompt from `character.json`, randomised framing / angle / expression / action / background, 832x1216, 1216x832 or 1024x1024).
- **37 kept** after curation. Rejected (moved to `rejected/`): 000, 037 (hair too red), 023 (hair washed out), 005 (jacket turned grey-blue, too small), 001, 002, 009 (bad crops), 010, 022, 043, 045 (messy/faded/tiny).
- Captions: WD14 `SmilingWolf/wd-vit-tagger-v3` (onnx, general ≥ 0.35, character ≥ 0.85), trigger word first. **Identity tags pruned** (pink/ash pink hair, short hair, bob cut, blunt bangs, green eyes) so the trigger absorbs them. Outfit tags kept, so the outfit can still be changed by prompt. Junk tags removed (`virtual youtuber`, etc.).
- Folder: `dataset/4_sumi_ink` → 4 repeats.

## Settings (8 GB-safe) — full file: `training_config.toml`
| Setting | Value | Why |
|---|---|---|
| resolution / buckets | 1024, buckets 640–1536 step 64, no upscale | SDXL native |
| network | LoRA, dim 16, alpha 8 | enough for one character, small file |
| train | UNet only | text encoders frozen → saves VRAM |
| base precision | `fp8_base` | UNet weights 5 GB → ~2.5 GB |
| mixed precision | bf16, saved as fp16 | |
| optimizer | AdamW8bit, lr 1e-4, cosine, 50 warmup steps | |
| min_snr_gamma | 5 | steadier loss |
| batch | 1, gradient checkpointing, SDPA attention | |
| caching | latents + text-encoder outputs cached to disk | lowest VRAM/RAM |
| epochs / steps | 10 epochs × 37 × 4 = **1,480 steps**, saved every 2 epochs | |
| shuffle_caption | **off** | sd-scripts forbids it together with cached text-encoder outputs |

## Run stats
| | |
|---|---|
| Time | ~46 min (1.84 s/step at the end) |
| Peak VRAM | 5.8 GB |
| Peak GPU power / temp | 91 W / 85 °C (NitroSense **Turbo**) |
| Final avg loss | ~0.062 |

## Problems hit
1. `shuffle_caption` + `cache_text_encoder_outputs` → AssertionError. Removed shuffle.
2. `UnicodeEncodeError` (sd-scripts prints Japanese; redirected Windows console is cp1252). Fixed by setting `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` for the training subprocess.
3. First ~40 steps ran at 4.7–5.2 s/step because the laptop was in a quiet/balanced thermal profile (GPU stuck at 1185 MHz, ~20 W, "SW thermal slowdown"). **Switching NitroSense to Turbo gave 2385 MHz, 55–90 W and 1.84 s/step.**

## Re-train / new character
```
comfyui\python_embeded\python.exe scripts\gen_lora_dataset.py <name> 48      # needs ComfyUI running + character.json
(delete bad images in characters\<name>\candidates)
trainers\sd-scripts\venv\Scripts\python.exe scripts\lora_pipeline.py prep <name>
trainers\sd-scripts\venv\Scripts\python.exe scripts\lora_pipeline.py train <name>   # close ComfyUI first
comfyui\python_embeded\python.exe scripts\test_lora.py <name>                  # needs ComfyUI running
```
