#!/usr/bin/env python3
"""
Lenscribe portfolio demo — Gradio UI.

Run:  python app.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import gradio as gr

from src.pipeline import LenscribePipeline

SAMPLE_IMAGE = ROOT / "examples" / "assets" / "sample.jpg"
_pipeline: LenscribePipeline | None = None


def get_pipeline() -> LenscribePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LenscribePipeline(enable_voice=True, enable_vgg16=False)
    return _pipeline


def run_describe(
    image,
    include_labels: bool,
    speak: bool,
    emotion: str,
):
    if image is None:
        return "Upload an image to begin.", "", None

    path = image if isinstance(image, str) else getattr(image, "name", None)
    if not path or not os.path.isfile(path):
        return "Could not read the uploaded image.", "", None

    pipe = get_pipeline()
    out = pipe.describe_image(
        path,
        include_labels=include_labels,
        speak=speak,
        emotion=emotion,
    )

    if out.get("error") and not out.get("caption"):
        return out["error"], "", None

    caption = out.get("caption") or ""
    labels_md = pipe.format_labels(out.get("labels"))
    if include_labels and out.get("labels") is not None and len(out["labels"]) == 0:
        labels_md = (
            "_Could not load VGG16 (install TensorFlow: `pip install tensorflow`)._"
        )

    status = out.get("error") or "Done."
    audio = out.get("audio_path") if speak and out.get("audio_path") else None
    return caption, labels_md if include_labels else "", audio


def run_vqa(image, question: str):
    if image is None:
        return "Upload an image first."
    path = image if isinstance(image, str) else getattr(image, "name", None)
    if not path:
        return "Could not read the uploaded image."
    return get_pipeline().ask_about_image(path, question or "")


def build_ui() -> gr.Blocks:
    emotions = [
        "neutral",
        "happy",
        "sad",
        "calm",
        "excited",
        "angry",
    ]

    with gr.Blocks(
        title="Lenscribe",
        theme=gr.themes.Soft(primary_hue="indigo"),
    ) as demo:
        gr.Markdown(
            """
            # Lenscribe
            **Describe any image in words — and hear it aloud.**

            Portfolio demo: BLIP captioning & visual Q&A, optional ImageNet tags (VGG16), emotion-controlled speech (Edge TTS).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="filepath", label="Image")
                if SAMPLE_IMAGE.is_file():
                    gr.Examples(
                        examples=[[str(SAMPLE_IMAGE)]],
                        inputs=image_in,
                        label="Sample image",
                    )
                include_labels = gr.Checkbox(
                    label="Show VGG16 classification tags (requires TensorFlow)",
                    value=False,
                )
                speak = gr.Checkbox(label="Read caption aloud", value=True)
                emotion = gr.Dropdown(
                    choices=emotions,
                    value="neutral",
                    label="Voice emotion",
                )
                describe_btn = gr.Button("Describe image", variant="primary")

            with gr.Column(scale=1):
                caption_out = gr.Textbox(label="BLIP caption", lines=4)
                labels_out = gr.Markdown(label="Tags")
                audio_out = gr.Audio(label="Narration", type="filepath")

        gr.Markdown("### Ask about this image (visual Q&A)")
        with gr.Row():
            question_in = gr.Textbox(
                label="Question",
                placeholder="What is in this image?",
                scale=3,
            )
            vqa_btn = gr.Button("Ask", scale=1)
        vqa_out = gr.Textbox(label="Answer", lines=2)

        describe_btn.click(
            fn=run_describe,
            inputs=[image_in, include_labels, speak, emotion],
            outputs=[caption_out, labels_out, audio_out],
        )
        vqa_btn.click(fn=run_vqa, inputs=[image_in, question_in], outputs=vqa_out)

        gr.Markdown(
            """
            ---
            **Stack:** PyTorch · Hugging Face BLIP · optional TensorFlow VGG16 · Edge TTS  
            **Note:** First run downloads BLIP weights (~1 GB). CPU is fine but slower.
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
