#!/usr/bin/env python3
"""Create a small sample image for demos (no external download)."""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "sample.jpg"


def main() -> None:
    img = Image.new("RGB", (512, 384), color=(240, 248, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((80, 120, 220, 280), fill=(70, 130, 180), outline=(30, 60, 120), width=3)
    draw.ellipse((280, 100, 440, 260), fill=(255, 200, 100), outline=(200, 140, 40), width=3)
    draw.text((140, 40), "Lenscribe demo", fill=(40, 40, 60))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, quality=92)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
