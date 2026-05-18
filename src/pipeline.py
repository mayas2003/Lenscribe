"""
Unified portfolio pipeline: image → caption / VQA / labels → speech.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def create_voice_processor(seed: Optional[int] = None):
    """Prefer lightweight Edge TTS; fall back to Coqui XTTS when installed."""
    from .models import voice_processor as vp_module

    if vp_module.TTS_AVAILABLE:
        try:
            return vp_module.VoiceProcessor(seed=seed)
        except Exception as exc:
            logger.warning("XTTS init failed (%s); using Edge TTS", exc)

    from .models.advanced_voice_processor import AdvancedVoiceProcessor

    return AdvancedVoiceProcessor(seed=seed)


class LenscribePipeline:
    """
    Lazy-loaded multimodal pipeline for portfolio demos.

    Flow: image → BLIP caption (and optional VQA) → optional VGG16 tags → TTS.
    """

    def __init__(
        self,
        *,
        enable_voice: bool = True,
        enable_vgg16: bool = False,
    ):
        self.enable_voice = enable_voice
        self.enable_vgg16 = enable_vgg16
        self._blip = None
        self._vgg16 = None
        self._voice = None

    @property
    def blip(self):
        if self._blip is None:
            from .models.blip_processor import BLIPProcessor

            logger.info("Loading BLIP models…")
            self._blip = BLIPProcessor()
        return self._blip

    @property
    def vgg16(self):
        if self._vgg16 is None:
            from .models.vgg16_classifier import VGG16Classifier

            logger.info("Loading VGG16…")
            self._vgg16 = VGG16Classifier()
        return self._vgg16

    @property
    def voice(self):
        if self._voice is None:
            logger.info("Loading voice backend…")
            self._voice = create_voice_processor()
        return self._voice

    def describe_image(
        self,
        image_path: str,
        *,
        include_labels: Optional[bool] = None,
        speak: bool = True,
        emotion: str = "neutral",
        caption_prompt: Optional[str] = None,
        max_caption_length: int = 50,
    ) -> Dict[str, Any]:
        """
        Generate a caption, optional ImageNet labels, and optional speech.

        Returns:
            dict with keys: caption, labels (list or None), audio_path (str or None), error (str or None)
        """
        result: Dict[str, Any] = {
            "caption": "",
            "labels": None,
            "audio_path": None,
            "error": None,
        }

        if not image_path or not os.path.isfile(image_path):
            result["error"] = f"Image not found: {image_path}"
            return result

        try:
            result["caption"] = self.blip.generate_caption(
                image_path,
                prompt=caption_prompt,
                max_length=max_caption_length,
                num_beams=5,
            )
        except Exception as exc:
            logger.exception("Caption failed")
            result["error"] = f"Caption failed: {exc}"
            return result

        use_labels = include_labels if include_labels is not None else self.enable_vgg16
        if use_labels:
            try:
                preds = self.vgg16.predict(image_path, top_k=5)
                result["labels"] = [
                    {"name": name, "confidence": float(conf)}
                    for _id, name, conf in preds
                ]
            except Exception as exc:
                logger.warning("VGG16 labels skipped: %s", exc)
                result["labels"] = []

        if speak and self.enable_voice and result["caption"]:
            try:
                fd, audio_path = tempfile.mkstemp(suffix=".wav", prefix="lenscribe_")
                os.close(fd)
                self.voice.synthesize_with_emotion(
                    text=result["caption"],
                    emotion=emotion,
                    output_path=audio_path,
                )
                result["audio_path"] = audio_path
            except Exception as exc:
                logger.warning("Speech synthesis failed: %s", exc)
                result["error"] = (
                    (result["error"] or "") + f" Speech failed: {exc}"
                ).strip()

        return result

    def ask_about_image(
        self,
        image_path: str,
        question: str,
        *,
        max_length: int = 40,
    ) -> str:
        if not question.strip():
            return "Please enter a question."
        if not image_path or not os.path.isfile(image_path):
            return f"Image not found: {image_path}"
        return self.blip.answer_question(image_path, question.strip(), max_length=max_length)

    @staticmethod
    def format_labels(labels: Optional[List[Dict[str, Any]]]) -> str:
        if not labels:
            return "_Classification tags unavailable (install TensorFlow and enable labels)._"
        lines = [
            f"- **{item['name']}** — {item['confidence']:.1%}"
            for item in labels
        ]
        return "\n".join(lines)
