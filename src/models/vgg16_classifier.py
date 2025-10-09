import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import os

class VGG16Classifier:
    """
    VGG16-based image classifier using pre-trained weights.
    """
    
    def __init__(self, weights='imagenet'):
        """
        Initialize the VGG16 classifier.
        
        Args:
            weights (str): Pre-trained weights to use ('imagenet' or None)
        """
        self.model = VGG16(weights=weights, include_top=True)
        self.input_size = (224, 224)
        
    def preprocess_image(self, image_path):
        """
        Preprocess image for VGG16 input.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        img = Image.open(image_path)
        # Convert to RGB to ensure 3 channels (remove alpha channel if present)
        img = img.convert('RGB')
        img = img.resize(self.input_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def predict(self, image_path, top_k=5):
        """
        Predict the class of an image.
        
        Args:
            image_path (str): Path to the image file
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of tuples (class_name, class_description, confidence)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Preprocess the image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array)
        
        # Decode predictions
        decoded_predictions = decode_predictions(predictions, top=top_k)[0]
        
        return decoded_predictions
    
    def extract_features(self, image_path):
        """
        Extract features from an image using VGG16 (without the top classification layer).
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Feature vector
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Create feature extraction model (without top layer)
        feature_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        
        # Preprocess the image
        img_array = self.preprocess_image(image_path)
        
        # Extract features
        features = feature_model.predict(img_array)
        
        return features.flatten()
    
    def get_model_summary(self):
        """
        Get the model architecture summary.
        
        Returns:
            str: Model summary
        """
        return self.model.summary()

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = VGG16Classifier()
    
    # Example image path (replace with actual image)
    image_path = "path/to/your/image.jpg"
    
    try:
        # Make prediction
        predictions = classifier.predict(image_path)
        print("Top 5 predictions:")
        for i, (class_id, class_name, confidence) in enumerate(predictions, 1):
            print(f"{i}. {class_name}: {confidence:.4f}")
        
        # Extract features
        features = classifier.extract_features(image_path)
        print(f"\nFeature vector shape: {features.shape}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide a valid image path.")
