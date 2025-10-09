import torch
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import os

class BLIPProcessor:
    """
    BLIP (Bootstrapping Language-Image Pre-training) processor for vision-language tasks.
    """
    
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """
        Initialize the BLIP processor.
        
        Args:
            model_name (str): Hugging Face model name for BLIP
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        # Initialize QA model separately
        self.qa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)
        self.qa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_fast=True)
    
    def generate_caption(self, image_path, max_length=50, num_beams=5):
        """
        Generate a caption for an image.
        
        Args:
            image_path (str): Path to the image file
            max_length (int): Maximum length of generated caption
            num_beams (int): Number of beams for beam search
            
        Returns:
            str: Generated caption
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=max_length, num_beams=num_beams)
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def answer_question(self, image_path, question):
        """
        Answer a question about an image.
        
        Args:
            image_path (str): Path to the image file
            question (str): Question to ask about the image
            
        Returns:
            str: Answer to the question
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = self.qa_processor(image, question, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            out = self.qa_model.generate(**inputs, max_length=50)
        
        answer = self.qa_processor.decode(out[0], skip_special_tokens=True)
        return answer
    
    def generate_multiple_captions(self, image_path, num_captions=3, max_length=50):
        """
        Generate multiple diverse captions for an image.
        
        Args:
            image_path (str): Path to the image file
            num_captions (int): Number of captions to generate
            max_length (int): Maximum length of each caption
            
        Returns:
            list: List of generated captions
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        captions = []
        for _ in range(num_captions):
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=max_length, 
                                        do_sample=True, temperature=0.7, 
                                        top_p=0.9, num_return_sequences=1)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        
        return captions
    
    def get_device_info(self):
        """
        Get information about the current device.
        
        Returns:
            str: Device information
        """
        return f"Using device: {self.device}"

# Example usage
if __name__ == "__main__":
    # Initialize BLIP processor
    blip = BLIPProcessor()
    print(blip.get_device_info())
    
    # Example image path (replace with actual image)
    image_path = "path/to/your/image.jpg"
    
    try:
        # Generate caption
        caption = blip.generate_caption(image_path)
        print(f"Generated caption: {caption}")
        
        # Answer a question
        question = "What is in this image?"
        answer = blip.answer_question(image_path, question)
        print(f"Answer to '{question}': {answer}")
        
        # Generate multiple captions
        captions = blip.generate_multiple_captions(image_path, num_captions=3)
        print("\nMultiple captions:")
        for i, cap in enumerate(captions, 1):
            print(f"{i}. {cap}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide a valid image path.")
    except Exception as e:
        print(f"An error occurred: {e}")
