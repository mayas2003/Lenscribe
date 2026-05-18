"""Fast tests for the portfolio pipeline (mocked, no model weights)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import LenscribePipeline


@pytest.fixture
def pipe():
    return LenscribePipeline(enable_voice=True, enable_vgg16=False)


def test_describe_missing_image(pipe):
    out = pipe.describe_image("not_a_real_file.jpg", speak=False)
    assert out["error"]
    assert not out["caption"]


def test_describe_happy_path(pipe, tmp_path):
    img = tmp_path / "x.jpg"
    from PIL import Image

    Image.new("RGB", (64, 64), color="blue").save(img)

    mock_blip = MagicMock()
    mock_blip.generate_caption.return_value = "a blue square"
    mock_voice = MagicMock()

    with patch.object(pipe, "blip", mock_blip), patch.object(pipe, "voice", mock_voice):
        out = pipe.describe_image(str(img), speak=True, emotion="happy")

    assert out["caption"] == "a blue square"
    mock_voice.synthesize_with_emotion.assert_called_once()
    assert out["audio_path"]
