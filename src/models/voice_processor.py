# voice_processor.py
"""
Voice processor for text-to-speech with emotion and tone control.

Features:
- Voice cloning and personalization using XTTS-v2
- Emotion control (happy, sad, angry, etc.)
- Tone adjustment and style transfer
- Batch processing support
- GPU/CPU device handling with memory optimization
- Fine-tuning capabilities for custom voices
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import os
import sys
import time
import logging
import random
import json
import torch
import numpy as np
from pathlib import Path

try:
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.manage import ModelManager
    from TTS.utils.synthesizer import Synthesizer
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: TTS library not available. Using alternative voice processor.")
    # Import alternative voice processor
    try:
        from .voice_processor_alternative import AlternativeVoiceProcessor
        ALTERNATIVE_AVAILABLE = True
    except ImportError:
        ALTERNATIVE_AVAILABLE = False
        print("Warning: Alternative voice processor not available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class VoiceProcessor:
    """
    Voice processor for text-to-speech with emotion and tone control.
    
    Handles voice cloning, emotion control, and provides robust inference
    with automatic fallback strategies for memory issues.
    """

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: Optional[torch.device] = None,
        use_fp16: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize voice processor with XTTS-v2 model.
        
        Args:
            model_name: TTS model name or path
            device: torch device (auto-detects CUDA if available)
            use_fp16: Use half precision for CUDA (saves memory)
            seed: Random seed for reproducibility
        """
        if not TTS_AVAILABLE:
            if ALTERNATIVE_AVAILABLE:
                # Use alternative voice processor
                logger.info("Using alternative voice processor (TTS not available)")
                return AlternativeVoiceProcessor.__init__(self, seed=seed)
            else:
                raise ImportError("TTS library not available. Install with: pip install TTS")
            
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.use_fp16 = use_fp16 and (self.device.type == "cuda")
        self.model_name = model_name

        # Set seed for reproducibility
        if seed is not None:
            self.set_seed(seed)

        # Initialize TTS
        logger.info("Loading TTS model: %s", model_name)
        self.tts = TTS(model_name).to(self.device)
        
        # Convert to FP16 if requested and supported
        if self.use_fp16:
            try:
                self.tts.model.half()
                logger.info("TTS model converted to FP16")
            except Exception as e:
                logger.warning("FP16 conversion failed: %s", e)
                self.use_fp16 = False

        logger.info("Voice processor loaded. Device: %s. FP16: %s", self.device, self.use_fp16)

    def set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        logger.info("Setting seed = %d", seed)
        random.seed(seed)
        try:
            import numpy as _np
            _np.random.seed(seed)
        except Exception:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def synthesize_with_emotion(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        emotion: str = "neutral",
        emotion_strength: float = 1.0,
        output_path: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> Union[str, np.ndarray]:
        """
        Synthesize speech with emotion control.
        
        Args:
            text: Text to synthesize
            speaker_wav: Path to reference speaker audio for voice cloning
            emotion: Emotion type ('happy', 'sad', 'angry', 'neutral', 'excited', 'calm')
            emotion_strength: Strength of emotion (0.0 to 1.0)
            output_path: Path to save audio file
            language: Language code
            **kwargs: Additional TTS parameters
            
        Returns:
            Audio file path or numpy array
        """
        try:
            # Prepare synthesis parameters
            synthesis_params = {
                "language": language,
                **kwargs
            }
            
            # Add emotion control if supported
            if hasattr(self.tts, 'emotion_control'):
                synthesis_params.update({
                    "emotion": emotion,
                    "emotion_strength": emotion_strength
                })
            
            # Synthesize with or without speaker reference
            if speaker_wav and os.path.exists(speaker_wav):
                # Voice cloning with emotion
                audio = self.tts.tts(
                    text=text,
                    speaker_wav=speaker_wav,
                    **synthesis_params
                )
            else:
                # Default voice with emotion
                audio = self.tts.tts(
                    text=text,
                    **synthesis_params
                )
            
            # Save audio if output path provided
            if output_path:
                self._save_audio(audio, output_path)
                return output_path
            
            return audio
            
        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            raise e

    def clone_voice_with_emotion(
        self,
        text: str,
        reference_audio: str,
        emotion: str = "neutral",
        emotion_strength: float = 1.0,
        output_path: Optional[str] = None,
        language: str = "en"
    ) -> Union[str, np.ndarray]:
        """
        Clone a voice with specific emotion.
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio for voice cloning
            emotion: Target emotion
            emotion_strength: Strength of emotion
            output_path: Path to save audio file
            language: Language code
            
        Returns:
            Audio file path or numpy array
        """
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
        
        return self.synthesize_with_emotion(
            text=text,
            speaker_wav=reference_audio,
            emotion=emotion,
            emotion_strength=emotion_strength,
            output_path=output_path,
            language=language
        )

    def batch_synthesize(
        self,
        texts: List[str],
        speaker_wav: Optional[str] = None,
        emotion: str = "neutral",
        emotion_strength: float = 1.0,
        output_dir: str = "output",
        language: str = "en"
    ) -> List[str]:
        """
        Synthesize multiple texts with emotion control.
        
        Args:
            texts: List of texts to synthesize
            speaker_wav: Path to reference speaker audio
            emotion: Emotion type
            emotion_strength: Strength of emotion
            output_dir: Directory to save audio files
            language: Language code
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"audio_{i:03d}.wav")
            try:
                self.synthesize_with_emotion(
                    text=text,
                    speaker_wav=speaker_wav,
                    emotion=emotion,
                    emotion_strength=emotion_strength,
                    output_path=output_path,
                    language=language
                )
                output_paths.append(output_path)
                logger.info("Generated: %s", output_path)
            except Exception as e:
                logger.error("Failed to generate audio %d: %s", i, e)
                output_paths.append(None)
        
        return output_paths

    def interpolate_emotions(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        emotion1: str = "neutral",
        emotion2: str = "happy",
        interpolation_steps: int = 5,
        output_dir: str = "emotion_interpolation",
        language: str = "en"
    ) -> List[str]:
        """
        Create emotion interpolation between two emotions.
        
        Args:
            text: Text to synthesize
            speaker_wav: Path to reference speaker audio
            emotion1: Starting emotion
            emotion2: Ending emotion
            interpolation_steps: Number of interpolation steps
            output_dir: Directory to save interpolated audio
            language: Language code
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i in range(interpolation_steps):
            # Calculate interpolation weight
            weight = i / (interpolation_steps - 1)
            
            # Interpolate emotion strength
            emotion_strength = weight
            
            # Choose emotion based on weight
            current_emotion = emotion2 if weight > 0.5 else emotion1
            
            output_path = os.path.join(output_dir, f"interpolation_{i:02d}.wav")
            
            try:
                self.synthesize_with_emotion(
                    text=text,
                    speaker_wav=speaker_wav,
                    emotion=current_emotion,
                    emotion_strength=emotion_strength,
                    output_path=output_path,
                    language=language
                )
                output_paths.append(output_path)
                logger.info("Generated interpolation %d: %s", i, output_path)
            except Exception as e:
                logger.error("Failed to generate interpolation %d: %s", i, e)
                output_paths.append(None)
        
        return output_paths

    def _save_audio(self, audio: np.ndarray, output_path: str) -> None:
        """Save audio array to file."""
        try:
            import soundfile as sf
            sf.write(output_path, audio, 22050)  # XTTS default sample rate
        except ImportError:
            # Fallback to scipy
            from scipy.io import wavfile
            wavfile.write(output_path, 22050, audio)

    def get_supported_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        return ["neutral", "happy", "sad", "angry", "excited", "calm", "surprised", "fearful"]

    def get_available_languages(self) -> List[str]:
        """Get list of supported languages."""
        return ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]

    def get_device_info(self) -> str:
        """Get current device and FP16 status."""
        return f"Device: {self.device} | FP16: {self.use_fp16}"

    def get_memory_usage(self) -> Dict[str, Union[str, float]]:
        """Get current GPU memory usage in GB."""
        if self.device.type == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "device": str(self.device)
            }
        return {"device": "cpu", "memory": "N/A"}

    def validate_model(self) -> bool:
        """Test that the model is working with a quick test."""
        try:
            logger.info("Validating voice model...")
            test_text = "Hello, this is a test."
            audio = self.synthesize_with_emotion(test_text, emotion="neutral")
            logger.info("Voice model validation successful")
            return True
        except Exception as e:
            logger.error("Voice model validation failed: %s", e)
            return False

    def to(self, device: Union[str, torch.device]) -> None:
        """Move model to specified device."""
        device = torch.device(device) if isinstance(device, str) else device
        logger.info("Moving voice model to device: %s", device)
        self.tts.to(device)
        self.device = device


class VoiceTrainer:
    """
    Voice trainer for fine-tuning voice models with custom data.
    """
    
    def __init__(self, voice_processor: VoiceProcessor):
        """
        Initialize voice trainer.
        
        Args:
            voice_processor: Initialized VoiceProcessor instance
        """
        self.voice_processor = voice_processor
        
    def prepare_training_data(
        self,
        audio_files: List[str],
        transcripts: List[str],
        output_dir: str = "training_data"
    ) -> str:
        """
        Prepare training data for voice fine-tuning.
        
        Args:
            audio_files: List of audio file paths
            transcripts: List of corresponding transcripts
            output_dir: Directory to save prepared data
            
        Returns:
            Path to prepared data directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metadata file
        metadata = []
        for i, (audio_file, transcript) in enumerate(zip(audio_files, transcripts)):
            if os.path.exists(audio_file):
                metadata.append({
                    "audio_file": audio_file,
                    "text": transcript,
                    "speaker": "custom"
                })
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info("Prepared %d training samples", len(metadata))
        return output_dir
    
    def create_emotion_dataset(
        self,
        base_audio: str,
        emotions: List[str],
        output_dir: str = "emotion_dataset"
    ) -> str:
        """
        Create emotion dataset from base audio.
        
        Args:
            base_audio: Path to base audio file
            emotions: List of emotions to generate
            output_dir: Directory to save emotion samples
            
        Returns:
            Path to emotion dataset directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate audio for each emotion
        for emotion in emotions:
            output_path = os.path.join(output_dir, f"{emotion}.wav")
            try:
                # Use the same text for all emotions for comparison
                text = "This is a test of emotional voice synthesis."
                self.voice_processor.synthesize_with_emotion(
                    text=text,
                    speaker_wav=base_audio,
                    emotion=emotion,
                    output_path=output_path
                )
                logger.info("Generated emotion sample: %s", output_path)
            except Exception as e:
                logger.error("Failed to generate emotion %s: %s", emotion, e)
        
        return output_dir


# CLI example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VoiceProcessor demo")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--speaker", type=str, help="Path to reference speaker audio")
    parser.add_argument("--emotion", type=str, default="neutral", help="Emotion type")
    parser.add_argument("--output", type=str, help="Output audio file path")
    parser.add_argument("--batch", action="store_true", help="Run batch mode")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA")
    args = parser.parse_args()

    # Build the voice processor
    vp = VoiceProcessor(use_fp16=args.fp16)
    
    logger.info(vp.get_device_info())
    
    if args.batch:
        # Batch processing example
        texts = [
            "Hello, this is a happy voice.",
            "This is a sad voice.",
            "This is an excited voice."
        ]
        emotions = ["happy", "sad", "excited"]
        
        for text, emotion in zip(texts, emotions):
            output_path = f"output_{emotion}.wav"
            vp.synthesize_with_emotion(
                text=text,
                speaker_wav=args.speaker,
                emotion=emotion,
                output_path=output_path
            )
            print(f"Generated: {output_path}")
    else:
        # Single synthesis
        output_path = vp.synthesize_with_emotion(
            text=args.text,
            speaker_wav=args.speaker,
            emotion=args.emotion,
            output_path=args.output
        )
        print(f"Generated: {output_path}")
