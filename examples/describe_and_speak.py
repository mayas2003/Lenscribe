#!/usr/bin/env python3
"""
CLI portfolio demo: image → caption → optional speech.

  python examples/describe_and_speak.py
  python examples/describe_and_speak.py path/to/image.jpg --emotion happy
  python examples/describe_and_speak.py --labels --no-speak
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import LenscribePipeline

DEFAULT_IMAGE = ROOT / "examples" / "assets" / "sample.jpg"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lenscribe: describe an image and optionally speak it")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE), help="Path to image")
    parser.add_argument("--emotion", default="neutral", help="TTS emotion (happy, sad, calm, …)")
    parser.add_argument("--labels", action="store_true", help="Include VGG16 top-5 tags")
    parser.add_argument("--no-speak", action="store_true", help="Skip text-to-speech")
    parser.add_argument("--question", default=None, help="Optional VQA question")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        print("Run: python examples/assets/create_sample.py")
        return 1

    pipe = LenscribePipeline(enable_voice=not args.no_speak, enable_vgg16=args.labels)

    if args.question:
        answer = pipe.ask_about_image(str(image_path), args.question)
        print(f"Q: {args.question}")
        print(f"A: {answer}\n")

    result = pipe.describe_image(
        str(image_path),
        include_labels=args.labels,
        speak=not args.no_speak,
        emotion=args.emotion,
    )

    if result.get("error") and not result.get("caption"):
        print(result["error"])
        return 1

    print(f"Caption: {result['caption']}\n")

    if args.labels and result.get("labels"):
        print("Top tags (VGG16):")
        for item in result["labels"]:
            print(f"  {item['name']}: {item['confidence']:.1%}")
        print()

    if result.get("audio_path"):
        print(f"Audio saved: {result['audio_path']}")

    if result.get("error"):
        print(f"Note: {result['error']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
