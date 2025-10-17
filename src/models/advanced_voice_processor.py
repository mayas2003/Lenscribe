# advanced_voice_processor.py
"""
Advanced voice processor with multiple TTS backends for Python 3.12.

Features:
- Microsoft Edge TTS (high-quality, multiple voices)
- Google TTS (gTTS) with emotion control
- pyttsx3 (offline, system voices)
- Voice cloning simulation
- Emotion control and tone adjustment
- Batch processing and interpolation
- Cross-platform compatibility
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import os
import sys
import time
import logging
import random
import json
import asyncio
import numpy as np
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import TTS backends
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import gtts
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False


class AdvancedVoiceProcessor:
    """
    Advanced voice processor with multiple TTS backends.
    
    Provides high-quality voice synthesis with emotion control,
    voice cloning simulation, and multiple language support.
    """

    def __init__(
        self,
        primary_backend: str = "edge_tts",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize advanced voice processor.
        
        Args:
            primary_backend: Primary TTS backend ('edge_tts', 'gtts', 'pyttsx3')
            device: Device info (for compatibility)
            seed: Random seed for reproducibility
        """
        self.primary_backend = primary_backend
        self.device = device or "cpu"
        
        # Set seed for reproducibility
        if seed is not None:
            self.set_seed(seed)
        
        # Initialize backends
        self.backends = {}
        self._initialize_backends()
        
        # Voice and emotion mappings
        self._setup_voice_mappings()
        
        logger.info(f"Advanced voice processor initialized with {primary_backend}")

    def _initialize_backends(self):
        """Initialize available TTS backends."""
        if EDGE_TTS_AVAILABLE:
            self.backends['edge_tts'] = True
            logger.info("Edge TTS backend available")
        
        if PYTTSX3_AVAILABLE:
            self.backends['pyttsx3'] = True
            try:
                self.pyttsx3_engine = pyttsx3.init()
                voices = self.pyttsx3_engine.getProperty('voices')
                self.pyttsx3_voices = voices if voices else []
                logger.info("pyttsx3 backend available")
            except Exception as e:
                logger.warning(f"pyttsx3 initialization failed: {e}")
                self.backends['pyttsx3'] = False
        
        if GTTS_AVAILABLE:
            self.backends['gtts'] = True
            logger.info("gTTS backend available")
        
        if not any(self.backends.values()):
            raise ImportError("No TTS backends available")

    def _setup_voice_mappings(self):
        """Setup voice and emotion mappings."""
        # Edge TTS voices with emotion characteristics
        self.edge_voices = {
            "neutral": ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural"],
            "happy": ["en-US-AriaNeural", "en-US-JennyNeural"],
            "sad": ["en-US-DavisNeural", "en-US-BrianNeural"],
            "angry": ["en-US-BrandonNeural", "en-US-GuyNeural"],
            "excited": ["en-US-AriaNeural", "en-US-JennyNeural"],
            "calm": ["en-US-DavisNeural", "en-US-BrianNeural"],
            "surprised": ["en-US-AriaNeural", "en-US-JennyNeural"],
            "fearful": ["en-US-DavisNeural", "en-US-BrianNeural"]
        }
        
        # Emotion to SSML style mapping
        self.emotion_styles = {
            "happy": "cheerful",
            "sad": "sad",
            "angry": "angry",
            "excited": "excited",
            "calm": "calm",
            "surprised": "surprised",
            "fearful": "fearful",
            "neutral": "friendly"
        }

    def set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

    async def _synthesize_edge_tts(
        self,
        text: str,
        emotion: str = "neutral",
        emotion_strength: float = 1.0,
        output_path: Optional[str] = None,
        language: str = "en"
    ) -> str:
        """Synthesize using Edge TTS."""
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("Edge TTS not available")
        
        # Select voice based on emotion
        voices = self.edge_voices.get(emotion, self.edge_voices["neutral"])
        voice = random.choice(voices)
        
        # Create SSML with emotion control
        style = self.emotion_styles.get(emotion, "friendly")
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
            <voice name="{voice}">
                <mstts:express-as style="{style}" styledegree="{emotion_strength}">
                    {text}
                </mstts:express-as>
            </voice>
        </speak>
        """
        
        # Generate audio
        communicate = edge_tts.Communicate(ssml, voice)
        
        if output_path:
            await communicate.save(output_path)
            return output_path
        else:
            # Return audio data
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data

    def _synthesize_pyttsx3(
        self,
        text: str,
        emotion: str = "neutral",
        emotion_strength: float = 1.0,
        output_path: Optional[str] = None
    ) -> str:
        """Synthesize using pyttsx3."""
        if not PYTTSX3_AVAILABLE:
            raise ImportError("pyttsx3 not available")
        
        # Adjust speech properties based on emotion
        rate, volume = self._get_emotion_properties(emotion, emotion_strength)
        
        self.pyttsx3_engine.setProperty('rate', rate)
        self.pyttsx3_engine.setProperty('volume', volume)
        
        if output_path:
            self.pyttsx3_engine.save_to_file(text, output_path)
            self.pyttsx3_engine.runAndWait()
            return output_path
        else:
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
            return "spoken"

    def _synthesize_gtts(
        self,
        text: str,
        emotion: str = "neutral",
        emotion_strength: float = 1.0,
        output_path: Optional[str] = None,
        language: str = "en"
    ) -> str:
        """Synthesize using gTTS."""
        if not GTTS_AVAILABLE:
            raise ImportError("gTTS not available")
        
        if not output_path:
            output_path = "temp_speech.mp3"
        
        # Create gTTS object
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_path)
        
        # Convert to WAV if needed
        if output_path.endswith('.mp3'):
            wav_path = output_path.replace('.mp3', '.wav')
            self._convert_mp3_to_wav(output_path, wav_path)
            os.remove(output_path)
            return wav_path
        
        return output_path

    def _get_emotion_properties(self, emotion: str, strength: float) -> Tuple[int, float]:
        """Get speech properties based on emotion."""
        base_rate = 150
        base_volume = 0.8
        
        emotion_adjustments = {
            "happy": {"rate": 20, "volume": 0.1},
            "sad": {"rate": -30, "volume": -0.2},
            "angry": {"rate": 40, "volume": 0.2},
            "excited": {"rate": 50, "volume": 0.15},
            "calm": {"rate": -20, "volume": -0.1},
            "surprised": {"rate": 30, "volume": 0.1},
            "fearful": {"rate": 10, "volume": -0.1},
            "neutral": {"rate": 0, "volume": 0}
        }
        
        adjustment = emotion_adjustments.get(emotion, emotion_adjustments["neutral"])
        
        rate = base_rate + int(adjustment["rate"] * strength)
        volume = max(0.1, min(1.0, base_volume + adjustment["volume"] * strength))
        
        return rate, volume

    def _convert_mp3_to_wav(self, mp3_path: str, wav_path: str) -> None:
        """Convert MP3 to WAV using pydub."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")
        except ImportError:
            logger.warning("pydub not available for MP3 conversion")
            import shutil
            shutil.copy2(mp3_path, wav_path)

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
            speaker_wav: Path to reference speaker audio (for voice cloning simulation)
            emotion: Emotion type
            emotion_strength: Strength of emotion (0.0 to 1.0)
            output_path: Path to save audio file
            language: Language code
            **kwargs: Additional parameters
            
        Returns:
            Audio file path or audio data
        """
        try:
            if self.primary_backend == "edge_tts" and self.backends.get('edge_tts', False):
                # Use asyncio for Edge TTS
                return asyncio.run(self._synthesize_edge_tts(
                    text, emotion, emotion_strength, output_path, language
                ))
            elif self.primary_backend == "pyttsx3" and self.backends.get('pyttsx3', False):
                return self._synthesize_pyttsx3(text, emotion, emotion_strength, output_path)
            elif self.primary_backend == "gtts" and self.backends.get('gtts', False):
                return self._synthesize_gtts(text, emotion, emotion_strength, output_path, language)
            else:
                # Fallback to any available backend
                if self.backends.get('edge_tts', False):
                    return asyncio.run(self._synthesize_edge_tts(
                        text, emotion, emotion_strength, output_path, language
                    ))
                elif self.backends.get('pyttsx3', False):
                    return self._synthesize_pyttsx3(text, emotion, emotion_strength, output_path)
                elif self.backends.get('gtts', False):
                    return self._synthesize_gtts(text, emotion, emotion_strength, output_path, language)
                else:
                    raise RuntimeError("No TTS backends available")
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
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
        Simulate voice cloning with emotion control.
        
        Note: This is a simplified implementation that doesn't actually clone voices
        but provides voice consistency through backend selection.
        """
        logger.info(f"Simulating voice cloning with emotion: {emotion}")
        
        # For now, use regular synthesis
        # In a full implementation, you would analyze the reference audio
        # and adjust voice parameters accordingly
        return self.synthesize_with_emotion(
            text=text,
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
        """Synthesize multiple texts with emotion control."""
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
                logger.info(f"Generated: {output_path}")
            except Exception as e:
                logger.error(f"Failed to generate audio {i}: {e}")
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
        """Create emotion interpolation between two emotions."""
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i in range(interpolation_steps):
            weight = i / (interpolation_steps - 1)
            emotion_strength = weight
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
                logger.info(f"Generated interpolation {i}: {output_path}")
            except Exception as e:
                logger.error(f"Failed to generate interpolation {i}: {e}")
                output_paths.append(None)
        
        return output_paths

    def get_supported_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        return ["neutral", "happy", "sad", "angry", "excited", "calm", "surprised", "fearful"]

    def get_available_languages(self) -> List[str]:
        """Get list of supported languages."""
        if self.backends.get('edge_tts', False):
            return ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
        elif self.backends.get('gtts', False):
            return ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
        else:
            return ["en"]

    def get_available_voices(self) -> List[str]:
        """Get list of available voices."""
        if self.backends.get('edge_tts', False):
            return list(set([voice for voices in self.edge_voices.values() for voice in voices]))
        elif self.backends.get('pyttsx3', False) and hasattr(self, 'pyttsx3_voices'):
            return [voice.name for voice in self.pyttsx3_voices]
        else:
            return ["default"]

    def get_device_info(self) -> str:
        """Get current device and backend info."""
        return f"Device: {self.device} | Backend: {self.primary_backend} | Available: {list(self.backends.keys())}"

    def get_memory_usage(self) -> Dict[str, Union[str, float]]:
        """Get memory usage info."""
        return {
            "device": self.device,
            "backend": self.primary_backend,
            "available_backends": list(self.backends.keys()),
            "memory": "N/A"
        }

    def validate_model(self) -> bool:
        """Test that the model is working."""
        try:
            logger.info("Validating advanced voice model...")
            test_text = "Hello, this is a test of the advanced voice processor."
            result = self.synthesize_with_emotion(test_text, emotion="neutral")
            logger.info("Advanced voice model validation successful")
            return True
        except Exception as e:
            logger.error(f"Advanced voice model validation failed: {e}")
            return False

    def to(self, device: Union[str, None]) -> None:
        """Move to specified device (for compatibility)."""
        self.device = device or "cpu"
        logger.info(f"Voice model device set to: {self.device}")


# Create compatibility wrapper
class VoiceProcessor(AdvancedVoiceProcessor):
    """
    Compatibility wrapper for the main VoiceProcessor.
    
    This allows the notebook to work with the advanced implementation
    when the main TTS library is not available.
    """
    
    def __init__(self, *args, **kwargs):
        # Remove TTS-specific parameters
        kwargs.pop('model_name', None)
        kwargs.pop('use_fp16', None)
        
        # Use advanced processor with Edge TTS as primary
        primary_backend = "edge_tts" if EDGE_TTS_AVAILABLE else "gtts"
        super().__init__(primary_backend=primary_backend, *args, **kwargs)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Advanced VoiceProcessor demo")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--emotion", type=str, default="neutral", help="Emotion type")
    parser.add_argument("--output", type=str, help="Output audio file path")
    parser.add_argument("--backend", type=str, default="edge_tts", help="TTS backend to use")
    args = parser.parse_args()

    # Build the voice processor
    vp = VoiceProcessor(primary_backend=args.backend)
    
    logger.info(vp.get_device_info())
    
    # Single synthesis
    output_path = vp.synthesize_with_emotion(
        text=args.text,
        emotion=args.emotion,
        output_path=args.output
    )
    print(f"Generated: {output_path}")
