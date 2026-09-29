"""Numbered contact sheet of a folder of images (for curating LoRA candidates / keyframes).

    comfyui\\python_embeded\\python.exe scripts\\contact_sheet.py project\\characters\\sumi\\candidates out.jpg [cols]
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 8
cell = 300
files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell, rows * cell), (20, 20, 24))
try:
    font = ImageFont.truetype("arialbd.ttf", 34)
except OSError:
    font = ImageFont.load_default()
draw = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB")
    im.thumbnail((cell - 6, cell - 6))
    x, y = (i % cols) * cell, (i // cols) * cell
    sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
    label = f.stem.split("_")[-1]
    draw.rectangle([x + 4, y + 4, x + 70, y + 44], fill=(0, 0, 0))
    draw.text((x + 8, y + 5), label, fill=(255, 220, 0), font=font)
sheet.save(dst, quality=88)
print(dst, len(files), "images")
