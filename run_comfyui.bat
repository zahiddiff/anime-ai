@echo off
REM AnimeAI - ComfyUI launcher, RTX 4060 Laptop 8 GB + 16 GB RAM
REM Flags benchmarked 2026-09-29 (NoobAI SDXL, 832x1216, 28 steps, warm model):
REM   default                          33.7 s/img
REM   --fast fp16_accumulation         28.9 s/img   <- used here (18.4 s/img with NitroSense Turbo!)
REM   --highvram --fast ...            29.4 s/img, but with ControlNet it overflows 8 GB into shared memory (225 s/step!) - don't use
REM Before running: close browsers, chat apps and game launchers, and stop Ollama (frees RAM/VRAM);
REM plug in the charger and set NitroSense to TURBO (the quiet profile throttles the GPU to ~1.2 GHz / 20 W).
cd /d "%~dp0comfyui"
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --enable-manager --preview-method auto --fast fp16_accumulation --output-directory "%~dp0project\_comfy_output" %*
pause
