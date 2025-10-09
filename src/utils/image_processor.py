import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class ImageProcessor:
    """
    Utility class for image processing operations.
    """
    
    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """
        Load an image from file path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            PIL.Image: Loaded image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        return Image.open(image_path).convert('RGB')
    
    @staticmethod
    def resize_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """
        Resize an image to specified dimensions.
        
        Args:
            image (PIL.Image): Input image
            size (tuple): Target size (width, height)
            
        Returns:
            PIL.Image: Resized image
        """
        return image.resize(size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def enhance_image(image: Image.Image, brightness: float = 1.0, 
                     contrast: float = 1.0, sharpness: float = 1.0) -> Image.Image:
        """
        Enhance image brightness, contrast, and sharpness.
        
        Args:
            image (PIL.Image): Input image
            brightness (float): Brightness factor (1.0 = no change)
            contrast (float): Contrast factor (1.0 = no change)
            sharpness (float): Sharpness factor (1.0 = no change)
            
        Returns:
            PIL.Image: Enhanced image
        """
        enhanced = image
        
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness)
        
        return enhanced
    
    @staticmethod
    def apply_filters(image: Image.Image, filter_type: str = 'blur') -> Image.Image:
        """
        Apply various filters to an image.
        
        Args:
            image (PIL.Image): Input image
            filter_type (str): Type of filter ('blur', 'sharpen', 'edge_enhance')
            
        Returns:
            PIL.Image: Filtered image
        """
        if filter_type == 'blur':
            return image.filter(ImageFilter.BLUR)
        elif filter_type == 'sharpen':
            return image.filter(ImageFilter.SHARPEN)
        elif filter_type == 'edge_enhance':
            return image.filter(ImageFilter.EDGE_ENHANCE)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
    
    @staticmethod
    def convert_to_opencv(image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to OpenCV format.
        
        Args:
            image (PIL.Image): Input PIL image
            
        Returns:
            np.ndarray: OpenCV image array
        """
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def convert_from_opencv(opencv_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image to PIL format.
        
        Args:
            opencv_image (np.ndarray): Input OpenCV image
            
        Returns:
            PIL.Image: PIL image
        """
        return Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    
    @staticmethod
    def display_image(image: Image.Image, title: str = "Image", size: Tuple[int, int] = (10, 8)):
        """
        Display an image using matplotlib.
        
        Args:
            image (PIL.Image): Image to display
            title (str): Title for the plot
            size (tuple): Figure size
        """
        plt.figure(figsize=size)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def save_image(image: Image.Image, output_path: str, quality: int = 95):
        """
        Save an image to file.
        
        Args:
            image (PIL.Image): Image to save
            output_path (str): Output file path
            quality (int): JPEG quality (1-100)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            image.save(output_path, 'JPEG', quality=quality)
        else:
            image.save(output_path)

# Example usage
if __name__ == "__main__":
    # Example image path (replace with actual image)
    image_path = "path/to/your/image.jpg"
    
    try:
        # Load image
        img = ImageProcessor.load_image(image_path)
        print(f"Image loaded successfully. Size: {img.size}")
        
        # Resize image
        resized = ImageProcessor.resize_image(img, (224, 224))
        print(f"Image resized to: {resized.size}")
        
        # Enhance image
        enhanced = ImageProcessor.enhance_image(img, brightness=1.2, contrast=1.1)
        print("Image enhanced")
        
        # Apply filter
        filtered = ImageProcessor.apply_filters(img, 'sharpen')
        print("Filter applied")
        
        # Save processed image
        ImageProcessor.save_image(enhanced, "output/enhanced_image.jpg")
        print("Enhanced image saved")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide a valid image path.")
    except Exception as e:
        print(f"An error occurred: {e}")
