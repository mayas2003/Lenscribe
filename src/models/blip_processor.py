# blip_processor.py
"""
BLIP processor for image captioning and visual question answering.

Features:
- Image captioning with beam search or sampling
- Visual Question Answering (VQA)
- Batch processing support
- CUDA/CPU device handling with OOM fallback
- FP16 inference for memory efficiency
- Deterministic seeding for reproducibility
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import os
import sys
import time
import math
import logging
import random

from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)

# Configure a simple logger - in real projects use structured logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class BLIPProcessor:
    """
    BLIP processor for image captioning and visual question answering.
    
    Handles model loading, device management, and provides robust inference
    with automatic fallback strategies for memory issues.
    """

    def __init__(
        self,
        caption_model_name: str = "Salesforce/blip-image-captioning-base",
        vqa_model_name: str = "Salesforce/blip-vqa-base",
        device: Optional[torch.device] = None,
        use_fp16: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize BLIP processor with captioning and VQA models.
        
        Args:
            caption_model_name: HuggingFace model ID for captioning
            vqa_model_name: HuggingFace model ID for VQA
            device: torch device (auto-detects CUDA if available)
            use_fp16: Use half precision for CUDA (saves memory)
            seed: Random seed for reproducibility
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.use_fp16 = use_fp16 and (self.device.type == "cuda")
        self.caption_model_name = caption_model_name
        self.vqa_model_name = vqa_model_name

        # Set seed for reproducibility
        if seed is not None:
            self.set_seed(seed)

        # Load captioning model
        logger.info("Loading caption model: %s", caption_model_name)
        self.processor = BlipProcessor.from_pretrained(caption_model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(caption_model_name).to(self.device)
        self.model.eval()

        # Convert to FP16 if requested and supported
        if self.use_fp16:
            try:
                self.model.half()
                logger.info("Caption model converted to FP16")
            except Exception as e:
                logger.warning("FP16 conversion failed: %s", e)
                self.use_fp16 = False

        # Load VQA model
        logger.info("Loading VQA model: %s", vqa_model_name)
        self.qa_processor = BlipProcessor.from_pretrained(vqa_model_name, use_fast=True)
        self.qa_model = BlipForQuestionAnswering.from_pretrained(vqa_model_name).to(self.device)
        self.qa_model.eval()

        if self.use_fp16:
            try:
                self.qa_model.half()
            except Exception:
                logger.debug("VQA model FP16 conversion failed")

        logger.info("Models loaded. Device: %s. FP16: %s", self.device, self.use_fp16)

    # Utility methods
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

    @staticmethod
    def _open_image(image_path: str) -> Image.Image:
        """Open and convert image to RGB format."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """Move all tensors in batch to specified device."""
        return {k: v.to(device) for k, v in batch.items()}

    # Caption generation
    def generate_caption(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_length: int = 40,
        num_beams: int = 5,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        return_full_output: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate caption for a single image.
        
        Args:
            image_path: Path to image file
            prompt: Optional prompt prefix
            max_length: Maximum caption length
            num_beams: Beam search width (higher = better quality, slower)
            do_sample: Use sampling instead of beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            return_full_output: Return metadata dict instead of just caption
            
        Returns:
            Caption string or metadata dict
        """
        image = self._open_image(image_path)

        # Process image and optional prompt
        if prompt:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")

        inputs = self._move_batch_to_device(inputs, self.device)

        # Configure generation parameters
        gen_kwargs: Dict[str, Any] = {"max_length": max_length}

        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
            })
            if top_k is not None:
                gen_kwargs["top_k"] = int(top_k)
            gen_kwargs["num_return_sequences"] = 1
        else:
            gen_kwargs["num_beams"] = max(1, int(num_beams))
            gen_kwargs["num_return_sequences"] = 1

        # Generate with OOM fallback
        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device.type == "cuda":
                logger.warning("CUDA OOM - trying fallback strategies")
                torch.cuda.empty_cache()
                try:
                    # Try with reduced beams
                    fallback_kwargs = gen_kwargs.copy()
                    if "num_beams" in fallback_kwargs and fallback_kwargs["num_beams"] > 1:
                        fallback_kwargs["num_beams"] = 1
                        fallback_kwargs["do_sample"] = True
                        fallback_kwargs["temperature"] = min(1.0, gen_kwargs.get("temperature", 1.0))
                        fallback_kwargs.pop("num_return_sequences", None)
                        logger.info("Retrying with fallback sampling")
                        with torch.no_grad():
                            output_ids = self.model.generate(**inputs, **fallback_kwargs)
                    else:
                        # Final fallback: CPU
                        logger.info("Moving to CPU (slow but safe)")
                        cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
                        model_cpu = self.model.to(torch.device("cpu"))
                        with torch.no_grad():
                            output_ids = model_cpu.generate(**cpu_inputs, **gen_kwargs)
                        self.model.to(self.device)
                except RuntimeError as e2:
                    logger.exception("Fallback failed: %s", e2)
                    raise e2
            else:
                logger.exception("Generation error: %s", e)
                raise e

        # Decode and return
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

        if return_full_output:
            return {
                "caption": caption,
                "model_name": self.caption_model_name,
                "device": str(self.device),
                "gen_kwargs": gen_kwargs,
                "timestamp": time.time(),
            }
        return caption
    
    def generate_captions_batch(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        max_length: int = 40,
        num_beams: int = 5,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate captions for multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            prompt: Optional prompt prefix for all images
            max_length: Maximum caption length
            num_beams: Beam search width
            do_sample: Use sampling instead of beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of captions per image
            
        Returns:
            List of captions
        """
        # Load and process images
        images = [self._open_image(p) for p in image_paths]

        if prompt:
            texts = [prompt] * len(images)
            inputs = self.processor(images=images, text=texts, return_tensors="pt")
        else:
            inputs = self.processor(images=images, return_tensors="pt")

        inputs = self._move_batch_to_device(inputs, self.device)

        # Configure generation
        gen_kwargs = {"max_length": max_length}
        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p)
            })
            gen_kwargs["num_return_sequences"] = num_return_sequences
        else:
            gen_kwargs["num_beams"] = max(1, int(num_beams))
            gen_kwargs["num_return_sequences"] = num_return_sequences

        # Generate with fallback to single images if batch fails
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
        except RuntimeError as e:
            logger.warning("Batch generation failed - falling back to single images: %s", e)
            captions = []
            for p in image_paths:
                captions.append(
                    self.generate_caption(
                        image_path=p,
                        prompt=prompt,
                        max_length=max_length,
                        num_beams=num_beams,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                    )
                )
            return captions

        # Decode outputs
        captions = [self.processor.decode(ids, skip_special_tokens=True).strip() for ids in outputs]
        return captions

    # Visual Question Answering
    def answer_question(
        self,
        image_path: str,
        question: str,
        max_length: int = 40,
    ) -> str:
        """
        Answer a question about an image.
        
        Args:
            image_path: Path to image file
            question: Question text
            max_length: Maximum answer length
            
        Returns:
            Answer string
        """
        image = self._open_image(image_path)
        inputs = self.qa_processor(images=image, text=question, return_tensors="pt")
        inputs = self._move_batch_to_device(inputs, self.device)

        gen_kwargs = {"max_length": max_length, "num_beams": 4}
        try:
            with torch.no_grad():
                outputs = self.qa_model.generate(**inputs, **gen_kwargs)
        except RuntimeError as e:
            logger.exception("VQA generation error: %s", e)
            raise e

        return self.qa_processor.decode(outputs[0], skip_special_tokens=True).strip()

    # Utility methods
    def get_device_info(self) -> str:
        """Get current device and FP16 status."""
        return f"Device: {self.device} | FP16: {self.use_fp16}"

    def warmup(self, image_path: Optional[str] = None) -> None:
        """Warm up the model with a small forward pass."""
        if image_path is not None and os.path.exists(image_path):
            logger.info("Warming up with image: %s", image_path)
            _ = self.generate_caption(image_path, max_length=8, num_beams=1)
        else:
            logger.info("Warming up with synthetic image")
            img = Image.new("RGB", (224, 224), color=(255, 255, 255))
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = self._move_batch_to_device(inputs, self.device)
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_length=8, num_beams=1)

    def get_memory_usage(self) -> Dict[str, Union[str, float]]:
        """Get current GPU memory usage in GB."""
        if self.device.type == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "device": str(self.device)
            }
        return {"device": "cpu", "memory": "N/A"}

    def validate_models(self) -> bool:
        """Test that models are working with a quick synthetic test."""
        try:
            logger.info("Validating models...")
            test_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            inputs = self.processor(images=test_img, return_tensors="pt")
            inputs = self._move_batch_to_device(inputs, self.device)
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_length=5, num_beams=1)
            logger.info("Model validation successful")
            return True
        except Exception as e:
            logger.error("Model validation failed: %s", e)
            return False

    # Device management
    def to(self, device: Union[str, torch.device]) -> None:
        """Move models to specified device."""
        device = torch.device(device) if isinstance(device, str) else device
        logger.info("Moving models to device: %s", device)
        self.model.to(device)
        self.qa_model.to(device)
        self.device = device

    # CLI example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BLIPProcessor demo")
    parser.add_argument("--image", type=str, required=False, help="Path to example image")
    parser.add_argument("--caption", action="store_true", help="Generate caption")
    parser.add_argument("--vqa", type=str, default=None, help="Ask a question about the image")
    parser.add_argument("--batch", action="store_true", help="Run batch mode by scanning images/ dir")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA (if available)")
    args = parser.parse_args()

    # Build the processor
    bp = BLIPProcessor(use_fp16=args.fp16)

    logger.info(bp.get_device_info())

    if args.batch:
        folder = "images"
        if not os.path.isdir(folder):
            logger.error("Create an 'images/' folder with some photos first")
            sys.exit(1)
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not files:
            logger.error("No images found in 'images/'")
            sys.exit(1)
        logger.info("Running batch caption for %d images", len(files))
        caps = bp.generate_captions_batch(files, do_sample=True, num_return_sequences=1, temperature=0.8)
        for f, c in zip(files, caps):
            print(f"FILE: {f} -> {c}")
        sys.exit(0)

    if args.image:
        if args.caption:
            out = bp.generate_caption(args.image, do_sample=False, num_beams=5)
            print("Caption:", out)

        if args.vqa:
            q = args.vqa
            ans = bp.answer_question(args.image, q)
            print("Q:", q)
            print("A:", ans)
    else:
        print("No image provided. Use --image <path> or --batch to run on images/ folder")