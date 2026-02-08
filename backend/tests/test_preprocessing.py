import pytest
from PIL import Image
from pathlib import Path
import sys
import torch

# Add backend to sys.path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from app.services.preprocessing.image import preprocess_image

def test_preprocess_image(tmp_path):
    # Create valid dummy image
    img_path = tmp_path / "test_face.jpg"
    img = Image.new('RGB', (100, 100), color = 'blue')
    img.save(img_path)

    # Run preprocessing
    # Note: MTCNN might fail to detect a face in a pure blue image and fallback to original.
    # We just want to ensure it runs without crashing and returns a path.
    processed_path = preprocess_image(str(img_path))
    
    assert Path(processed_path).exists()
    
    # Check if we can load the result
    result_img = Image.open(processed_path)
    assert result_img is not None
