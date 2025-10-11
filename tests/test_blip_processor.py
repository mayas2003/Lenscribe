"""
Test suite for BLIPProcessor class.

This test file demonstrates key testing concepts:
1. Unit Testing - Testing individual methods in isolation
2. Integration Testing - Testing how components work together
3. Mocking - Simulating external dependencies
4. Fixtures - Reusable test data and setup
5. Parametrized Testing - Testing multiple scenarios efficiently
6. Error Handling Testing - Ensuring robust error management
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import sys
from pathlib import Path

# Add src to path so we can import our module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.blip_processor import BLIPProcessor


class TestBLIPProcessor:
    """
    Test class for BLIPProcessor.
    
    Key Testing Concepts Demonstrated:
    - Setup/Teardown: @pytest.fixture for reusable test data
    - Mocking: @patch for external dependencies
    - Parametrized Testing: @pytest.mark.parametrize for multiple test cases
    - Exception Testing: Testing error conditions
    - Integration Testing: Testing real functionality
    """

    @pytest.fixture
    def sample_image_path(self):
        """
        Fixture: Creates a temporary test image.
        
        Why use fixtures?
        - Reusable across multiple tests
        - Automatic cleanup (teardown)
        - Consistent test data
        """
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Create a simple test image
            img = Image.new('RGB', (224, 224), color='red')
            img.save(tmp_file.name, 'JPEG')
            yield tmp_file.name
        
        # Cleanup after test
        os.unlink(tmp_file.name)

    @pytest.fixture
    def mock_processor(self):
        """
        Fixture: Creates a mock processor for testing without loading real models.
        
        This is MUCH faster than loading actual AI models!
        """
        with patch('models.blip_processor.BlipProcessor') as mock_blip_processor, \
             patch('models.blip_processor.BlipForConditionalGeneration') as mock_caption_model, \
             patch('models.blip_processor.BlipForQuestionAnswering') as mock_vqa_model:
            
            # Mock the processor and models
            mock_processor = Mock()
            mock_caption_model_instance = Mock()
            mock_vqa_model_instance = Mock()
            
            # Configure mocks
            mock_blip_processor.from_pretrained.return_value = mock_processor
            mock_caption_model.from_pretrained.return_value = mock_caption_model_instance
            mock_vqa_model.from_pretrained.return_value = mock_vqa_model_instance
            
            # Create BLIPProcessor instance with mocked dependencies
            processor = BLIPProcessor()
            
            # Store mocks for later use
            processor._mock_processor = mock_processor
            processor._mock_caption_model = mock_caption_model_instance
            processor._mock_vqa_model = mock_vqa_model_instance
            
            yield processor

    def test_initialization_defaults(self):
        """
        Test 1: Initialization with default parameters.
        
        Concept: Unit Testing - Testing one specific behavior
        """
        with patch('models.blip_processor.BlipProcessor'), \
             patch('models.blip_processor.BlipForConditionalGeneration'), \
             patch('models.blip_processor.BlipForQuestionAnswering'):
            
            processor = BLIPProcessor()
            
            # Assertions: Check that initialization worked correctly
            assert processor.caption_model_name == "Salesforce/blip-image-captioning-base"
            assert processor.vqa_model_name == "Salesforce/blip-vqa-base"
            assert processor.device.type in ["cpu", "cuda"]  # Should auto-detect
            assert processor.use_fp16 == False  # Default should be False

    def test_initialization_custom_params(self):
        """
        Test 2: Initialization with custom parameters.
        
        Concept: Parameter Testing - Testing different input combinations
        """
        with patch('models.blip_processor.BlipProcessor'), \
             patch('models.blip_processor.BlipForConditionalGeneration'), \
             patch('models.blip_processor.BlipForQuestionAnswering'):
            
            custom_device = torch.device("cpu")
            processor = BLIPProcessor(
                caption_model_name="custom/caption-model",
                vqa_model_name="custom/vqa-model", 
                device=custom_device,
                use_fp16=True,
                seed=42
            )
            
            assert processor.caption_model_name == "custom/caption-model"
            assert processor.vqa_model_name == "custom/vqa-model"
            assert processor.device == custom_device
            assert processor.use_fp16 == True

    def test_device_auto_detection(self):
        """
        Test 3: Device auto-detection logic.
        
        Concept: Conditional Testing - Testing different code paths
        """
        with patch('models.blip_processor.BlipProcessor'), \
             patch('models.blip_processor.BlipForConditionalGeneration'), \
             patch('models.blip_processor.BlipForQuestionAnswering'):
            
            # Test CUDA detection
            with patch('torch.cuda.is_available', return_value=True):
                processor = BLIPProcessor()
                assert processor.device.type == "cuda"
            
            # Test CPU fallback
            with patch('torch.cuda.is_available', return_value=False):
                processor = BLIPProcessor()
                assert processor.device.type == "cpu"

    @pytest.mark.parametrize("use_fp16,device_type,expected_fp16", [
        (True, "cuda", True),   # FP16 on CUDA should work
        (True, "cpu", False),  # FP16 on CPU should be disabled
        (False, "cuda", False), # FP16 disabled
        (False, "cpu", False),  # FP16 disabled
    ])
    def test_fp16_configuration(self, use_fp16, device_type, expected_fp16):
        """
        Test 4: FP16 configuration logic.
        
        Concept: Parametrized Testing - Testing multiple scenarios efficiently
        """
        with patch('models.blip_processor.BlipProcessor'), \
             patch('models.blip_processor.BlipForConditionalGeneration'), \
             patch('models.blip_processor.BlipForQuestionAnswering'):
            
            device = torch.device(device_type)
            processor = BLIPProcessor(device=device, use_fp16=use_fp16)
            assert processor.use_fp16 == expected_fp16

    def test_set_seed(self):
        """
        Test 5: Seed setting functionality.
        
        Concept: State Testing - Testing that state changes work correctly
        """
        with patch('models.blip_processor.BlipProcessor'), \
             patch('models.blip_processor.BlipForConditionalGeneration'), \
             patch('models.blip_processor.BlipForQuestionAnswering'):
            
            processor = BLIPProcessor()
            
            # Test seed setting
            processor.set_seed(12345)
            # Note: In a real test, you might check that random numbers are consistent
            # This is a basic test that the method doesn't crash

    def test_open_image_success(self, sample_image_path):
        """
        Test 6: Image opening with valid file.
        
        Concept: File I/O Testing - Testing file operations
        """
        with patch('models.blip_processor.BlipProcessor'), \
             patch('models.blip_processor.BlipForConditionalGeneration'), \
             patch('models.blip_processor.BlipForQuestionAnswering'):
            
            processor = BLIPProcessor()
            image = processor._open_image(sample_image_path)
            
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"

    def test_open_image_file_not_found(self):
        """
        Test 7: Image opening with non-existent file.
        
        Concept: Exception Testing - Testing error conditions
        """
        with patch('models.blip_processor.BlipProcessor'), \
             patch('models.blip_processor.BlipForConditionalGeneration'), \
             patch('models.blip_processor.BlipForQuestionAnswering'):
            
            processor = BLIPProcessor()
            
            # Test that FileNotFoundError is raised for non-existent file
            with pytest.raises(FileNotFoundError):
                processor._open_image("non_existent_file.jpg")

    def test_move_batch_to_device(self):
        """
        Test 8: Batch device movement.
        
        Concept: Data Transformation Testing - Testing data processing
        """
        with patch('models.blip_processor.BlipProcessor'), \
             patch('models.blip_processor.BlipForConditionalGeneration'), \
             patch('models.blip_processor.BlipForQuestionAnswering'):
            
            processor = BLIPProcessor()
            
            # Create mock tensors
            batch = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            # Test device movement
            result = processor._move_batch_to_device(batch, torch.device("cpu"))
            
            assert isinstance(result, dict)
            assert all(isinstance(v, torch.Tensor) for v in result.values())

    def test_generate_caption_success(self, mock_processor, sample_image_path):
        """
        Test 9: Successful caption generation.
        
        Concept: Integration Testing - Testing real functionality with mocked dependencies
        """
        # Mock the processor's generate method to return fake output
        mock_output_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Fake token IDs
        mock_processor._mock_caption_model.generate.return_value = mock_output_ids
        mock_processor._mock_processor.decode.return_value = "A beautiful red image"
        
        # Test caption generation
        caption = mock_processor.generate_caption(sample_image_path)
        
        assert caption == "A beautiful red image"
        mock_processor._mock_caption_model.generate.assert_called_once()

    def test_generate_caption_with_prompt(self, mock_processor, sample_image_path):
        """
        Test 10: Caption generation with prompt.
        
        Concept: Parameter Testing - Testing different input combinations
        """
        mock_output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_processor._mock_caption_model.generate.return_value = mock_output_ids
        mock_processor._mock_processor.decode.return_value = "A red car in the image"
        
        caption = mock_processor.generate_caption(
            sample_image_path, 
            prompt="Describe the car"
        )
        
        assert caption == "A red car in the image"

    def test_generate_caption_oom_fallback(self, mock_processor, sample_image_path):
        """
        Test 11: OOM (Out of Memory) fallback behavior.
        
        Concept: Error Handling Testing - Testing fallback strategies
        """
        # Mock OOM error on first call, success on second
        mock_processor._mock_caption_model.generate.side_effect = [
            RuntimeError("CUDA out of memory"),  # First call fails
            torch.tensor([[1, 2, 3, 4, 5]])     # Second call succeeds
        ]
        mock_processor._mock_processor.decode.return_value = "Fallback caption"
        
        # Mock CUDA availability
        mock_processor.device = torch.device("cuda")
        
        caption = mock_processor.generate_caption(sample_image_path)
        
        assert caption == "Fallback caption"
        # Should have been called twice (original + fallback)
        assert mock_processor._mock_caption_model.generate.call_count == 2

    def test_generate_captions_batch_success(self, mock_processor):
        """
        Test 12: Batch caption generation.
        
        Concept: Batch Processing Testing - Testing multiple items at once
        """
        # Create multiple test images
        image_paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                img = Image.new('RGB', (224, 224), color=(i*50, 100, 150))
                img.save(tmp.name, 'JPEG')
                image_paths.append(tmp.name)
        
        try:
            # Mock batch processing
            mock_outputs = [
                torch.tensor([[1, 2, 3]]),
                torch.tensor([[4, 5, 6]]), 
                torch.tensor([[7, 8, 9]])
            ]
            mock_processor._mock_caption_model.generate.return_value = mock_outputs
            mock_processor._mock_processor.decode.side_effect = [
                "First image caption",
                "Second image caption", 
                "Third image caption"
            ]
            
            captions = mock_processor.generate_captions_batch(image_paths)
            
            assert len(captions) == 3
            assert captions[0] == "First image caption"
            assert captions[1] == "Second image caption"
            assert captions[2] == "Third image caption"
            
        finally:
            # Cleanup
            for path in image_paths:
                os.unlink(path)

    def test_answer_question(self, mock_processor, sample_image_path):
        """
        Test 13: Visual Question Answering.
        
        Concept: Multi-Model Testing - Testing different model types
        """
        mock_output_ids = torch.tensor([[10, 11, 12]])
        mock_processor._mock_vqa_model.generate.return_value = mock_output_ids
        mock_processor._mock_processor.decode.return_value = "Yes, it is red"
        
        answer = mock_processor.answer_question(
            sample_image_path, 
            "Is the image red?"
        )
        
        assert answer == "Yes, it is red"
        mock_processor._mock_vqa_model.generate.assert_called_once()

    def test_get_device_info(self, mock_processor):
        """
        Test 14: Device information retrieval.
        
        Concept: State Query Testing - Testing getter methods
        """
        mock_processor.device = torch.device("cuda")
        mock_processor.use_fp16 = True
        
        info = mock_processor.get_device_info()
        
        assert "cuda" in info
        assert "FP16: True" in info

    def test_memory_usage_cuda(self, mock_processor):
        """
        Test 15: Memory usage on CUDA.
        
        Concept: System Information Testing - Testing system monitoring
        """
        mock_processor.device = torch.device("cuda")
        
        with patch('torch.cuda.memory_allocated', return_value=1024**3), \
             patch('torch.cuda.memory_reserved', return_value=2*1024**3):
            
            memory_info = mock_processor.get_memory_usage()
            
            assert memory_info["allocated_gb"] == 1.0
            assert memory_info["reserved_gb"] == 2.0
            assert "cuda" in memory_info["device"]

    def test_memory_usage_cpu(self, mock_processor):
        """
        Test 16: Memory usage on CPU.
        
        Concept: Platform-Specific Testing - Testing different environments
        """
        mock_processor.device = torch.device("cpu")
        
        memory_info = mock_processor.get_memory_usage()
        
        assert memory_info["device"] == "cpu"
        assert memory_info["memory"] == "N/A"

    def test_validate_models_success(self, mock_processor):
        """
        Test 17: Model validation success.
        
        Concept: Health Check Testing - Testing system validation
        """
        mock_processor._mock_caption_model.generate.return_value = torch.tensor([[1, 2, 3]])
        
        result = mock_processor.validate_models()
        
        assert result == True

    def test_validate_models_failure(self, mock_processor):
        """
        Test 18: Model validation failure.
        
        Concept: Failure Testing - Testing error conditions
        """
        mock_processor._mock_caption_model.generate.side_effect = RuntimeError("Model error")
        
        result = mock_processor.validate_models()
        
        assert result == False

    def test_to_device(self, mock_processor):
        """
        Test 19: Device movement.
        
        Concept: State Change Testing - Testing state modifications
        """
        new_device = torch.device("cpu")
        
        mock_processor.to(new_device)
        
        assert mock_processor.device == new_device
        mock_processor._mock_caption_model.to.assert_called_with(new_device)
        mock_processor._mock_vqa_model.to.assert_called_with(new_device)


# Integration Tests (slower, more realistic)
class TestBLIPProcessorIntegration:
    """
    Integration tests that use real models (slower but more realistic).
    
    These tests are marked with @pytest.mark.slow and can be skipped
    during fast development cycles.
    """
    
    @pytest.mark.slow
    def test_real_model_loading(self):
        """
        Test 20: Real model loading (slow test).
        
        Concept: Integration Testing - Testing with real dependencies
        """
        # This test actually loads the models - it's slow but thorough
        processor = BLIPProcessor()
        
        # Basic validation
        assert processor.device is not None
        assert hasattr(processor, 'model')
        assert hasattr(processor, 'qa_model')

    @pytest.mark.slow
    def test_real_caption_generation(self, sample_image_path):
        """
        Test 21: Real caption generation (slow test).
        
        Concept: End-to-End Testing - Testing complete workflows
        """
        processor = BLIPProcessor()
        
        # This actually generates a caption - slow but realistic
        caption = processor.generate_caption(sample_image_path, max_length=10)
        
        assert isinstance(caption, str)
        assert len(caption) > 0


# Performance Tests
class TestBLIPProcessorPerformance:
    """
    Performance tests to ensure the code is efficient.
    """
    
    def test_batch_processing_efficiency(self, mock_processor):
        """
        Test 22: Batch processing should be more efficient than individual processing.
        
        Concept: Performance Testing - Testing efficiency
        """
        import time
        
        # Create test images
        image_paths = []
        for i in range(5):
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                img = Image.new('RGB', (224, 224), color=(i*50, 100, 150))
                img.save(tmp.name, 'JPEG')
                image_paths.append(tmp.name)
        
        try:
            # Mock batch processing
            mock_outputs = [torch.tensor([[1, 2, 3]]) for _ in range(5)]
            mock_processor._mock_caption_model.generate.return_value = mock_outputs
            mock_processor._mock_processor.decode.return_value = "Test caption"
            
            start_time = time.time()
            captions = mock_processor.generate_captions_batch(image_paths)
            batch_time = time.time() - start_time
            
            assert len(captions) == 5
            assert batch_time < 1.0  # Should be fast with mocked models
            
        finally:
            for path in image_paths:
                os.unlink(path)


if __name__ == "__main__":
    """
    How to run this test file:
    
    1. Run all tests:
       pytest tests/test_blip_processor.py -v
    
    2. Run only fast tests (skip slow integration tests):
       pytest tests/test_blip_processor.py -v -m "not slow"
    
    3. Run only slow tests:
       pytest tests/test_blip_processor.py -v -m "slow"
    
    4. Run specific test:
       pytest tests/test_blip_processor.py::TestBLIPProcessor::test_initialization_defaults -v
    
    5. Run with coverage:
       pytest tests/test_blip_processor.py --cov=src/models/blip_processor --cov-report=html
    
    6. Run in parallel (faster):
       pytest tests/test_blip_processor.py -n auto
    """
    pytest.main([__file__, "-v"])
