# Lenscribe

**Multimodal portfolio demo:** upload an image → get a natural-language caption → hear it read aloud. Optional visual Q&A and ImageNet tags.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Demo

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements-demo.txt
python examples/assets/create_sample.py
python app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

**CLI (same pipeline):**

```bash
python examples/describe_and_speak.py
python examples/describe_and_speak.py my_photo.jpg --emotion happy
python examples/describe_and_speak.py my_photo.jpg --question "What colors do you see?"
```

> First run downloads BLIP weights (~1 GB). CPU works; GPU is faster.

## What it does

```
┌─────────┐     ┌──────────────┐     ┌─────────────────┐
│  Image  │────▶│ BLIP caption │────▶│ Edge TTS speech │
└─────────┘     └──────┬───────┘     └─────────────────┘
                       │
                       ├────▶ BLIP VQA (ask a question)
                       │
                       └────▶ VGG16 tags (optional)
```

| Feature | Model / tool |
|--------|----------------|
| Image captioning | [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) (Hugging Face) |
| Visual Q&A | [BLIP VQA](https://huggingface.co/Salesforce/blip-vqa-base) |
| Classification tags (optional) | VGG16 / ImageNet |
| Text-to-speech | Microsoft Edge TTS (lightweight demo default) |
| Voice cloning (optional) | Coqui XTTS-v2 — see [full requirements](requirements.txt) |

## Project structure

```
Lenscribe/
├── app.py                      # Gradio portfolio UI
├── requirements-demo.txt       # Recommended install for reviewers
├── requirements.txt            # Full stack (TensorFlow, Coqui TTS, …)
├── src/
│   ├── pipeline.py             # Unified describe + speak flow
│   ├── models/                 # BLIP, VGG16, voice backends
│   └── utils/                  # Image helpers
├── examples/
│   ├── describe_and_speak.py   # CLI demo
│   └── assets/sample.jpg       # Generated sample image
├── notebooks/demo.ipynb
└── tests/
```

## Installation options

| Install | Command | Use when |
|---------|---------|----------|
| **Demo (recommended)** | `pip install -r requirements-demo.txt` | Portfolio / Gradio / Edge TTS |
| **Full** | `pip install -r requirements.txt` | VGG16 + LAVIS + Coqui XTTS + dev tools |

Optional VGG16 panel in the UI:

```bash
pip install tensorflow
```

## Development

```bash
pip install -r requirements.txt
pytest tests/ -m "not slow" -v
black src/ tests/ app.py
```

Slow integration tests (download real BLIP weights):

```bash
pytest tests/ -m slow -v
```

## Limitations (portfolio scope)

- Not production-hardened (no auth, rate limits, or model serving layer).
- BLIP + optional TensorFlow increase install size and first-run time.
- Edge TTS needs network access; XTTS voice cloning is optional and heavier.
- Emotion on Edge TTS uses neural voice styles, not true speaker cloning.

## What I'd add with more time

- Hugging Face Spaces deployment and a recorded demo GIF.
- Single `pyproject.toml` entry point (`lenscribe` CLI).
- Model warm-up endpoint and response caching for repeat images.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- VGG16 — *Very Deep Convolutional Networks for Large-Scale Image Recognition*
- BLIP — Salesforce Research
- Edge TTS / Coqui TTS communities
