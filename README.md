# Lenscribe

A Python project for computer vision and vision-language tasks using VGG16 and BLIP models.

## Overview

Lenscribe is designed to work with pre-trained models for various computer vision, vision-language understanding, and voice synthesis tasks. The project leverages:

- **VGG16**: A deep convolutional neural network for image classification and feature extraction
- **BLIP**: Bootstrapping Language-Image Pre-training for vision-language understanding tasks
- **XTTS-v2**: Advanced text-to-speech with emotion control and voice cloning capabilities

## Project Structure

`
Lenscribe/
â”œâ”€â”€ data/                   # Data storage and processing
â”œâ”€â”€ models/                 # Model checkpoints and saved models
â”œâ”€â”€ notebooks/              # Jupyter notebooks for experimentation
â”œâ”€â”€ src/                    # Source code modules
â”œâ”€â”€ tests/                  # Unit tests
â”œâ”€â”€ config/                 # Configuration files
â”œâ”€â”€ docs/                   # Documentation
â”œâ”€â”€ scripts/                # Utility scripts
â”œâ”€â”€ requirements.txt        # Python dependencies
â”œâ”€â”€ .gitignore             # Git ignore rules
â””â”€â”€ README.md              # This file
`

## Features

- **VGG16 Integration**: Pre-trained VGG16 model for image classification and feature extraction
- **BLIP Integration**: Vision-language model for image captioning, visual question answering, and more
- **Voice Synthesis**: Advanced text-to-speech with emotion control and voice cloning
- **Emotion Control**: Generate speech with different emotions (happy, sad, angry, etc.)
- **Voice Cloning**: Clone voices from reference audio with emotion control
- **Modular Design**: Clean, organized code structure for easy extension
- **Comprehensive Testing**: Unit tests for all major components
- **Documentation**: Detailed documentation and examples

## Installation

1. Clone the repository:
`ash
git clone https://github.com/yourusername/Lenscribe.git
cd Lenscribe
`

2. Create a virtual environment:
`ash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
`

3. Install dependencies:
`ash
pip install -r requirements.txt
`

## Usage

### VGG16 Image Classification

`python
from src.models.vgg16_classifier import VGG16Classifier

# Initialize the classifier
classifier = VGG16Classifier()

# Classify an image
predictions = classifier.predict('path/to/image.jpg')
print(predictions)
`

### BLIP Vision-Language Tasks

`python
from src.models.blip_processor import BLIPProcessor

# Initialize the processor
processor = BLIPProcessor()

# Generate image caption
caption = processor.generate_caption('path/to/image.jpg')
print(caption)

# Answer questions about an image
answer = processor.answer_question('path/to/image.jpg', 'What is in this image?')
print(answer)
`

### Voice Synthesis with Emotion Control

`python
from src.models.voice_processor import VoiceProcessor

# Initialize voice processor
voice = VoiceProcessor()

# Synthesize speech with emotion
voice.synthesize_with_emotion(
    text="Hello, this is a happy voice!",
    emotion="happy",
    output_path="happy_voice.wav"
)

# Clone voice with emotion
voice.clone_voice_with_emotion(
    text="This is my cloned voice.",
    reference_audio="path/to/reference.wav",
    emotion="sad",
    output_path="cloned_sad_voice.wav"
)
`

## Development

### Running Tests

`ash
pytest tests/
`

### Code Formatting

`ash
black src/ tests/
flake8 src/ tests/
`

### Jupyter Notebooks

Start Jupyter Lab for interactive development:

`ash
jupyter lab
`

## Contributing

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add some amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- VGG16 model from the original paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- BLIP model from Salesforce Research: "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation"
- Various open-source libraries and the Python community
