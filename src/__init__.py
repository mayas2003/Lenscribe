# Lenscribe - Computer Vision and Vision-Language Processing

__version__ = "0.1.0"
__author__ = "Mayas Phool"
__email__ = "mayasphool@gmail.com"

__all__ = [
    "BLIPProcessor",
    "VGG16Classifier",
    "VoiceProcessor",
    "VoiceTrainer",
    "AdvancedVoiceProcessor",
    "ImageProcessor",
    "LenscribePipeline",
    "create_voice_processor",
]


def __getattr__(name: str):
    if name == "BLIPProcessor":
        from .models.blip_processor import BLIPProcessor

        return BLIPProcessor
    if name == "VGG16Classifier":
        from .models.vgg16_classifier import VGG16Classifier

        return VGG16Classifier
    if name in ("VoiceProcessor", "VoiceTrainer"):
        from .models import voice_processor as vp

        return getattr(vp, name)
    if name == "AdvancedVoiceProcessor":
        from .models.advanced_voice_processor import AdvancedVoiceProcessor

        return AdvancedVoiceProcessor
    if name == "ImageProcessor":
        from .utils.image_processor import ImageProcessor

        return ImageProcessor
    if name == "LenscribePipeline":
        from .pipeline import LenscribePipeline

        return LenscribePipeline
    if name == "create_voice_processor":
        from .pipeline import create_voice_processor

        return create_voice_processor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
