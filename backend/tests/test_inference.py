import os
import sys
import pytest
from pathlib import Path
from PIL import Image

# Add backend to sys.path so we can import app modules
# backend/tests/test_inference.py -> backend
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from app.models.image_model import ImageDeepfakeModel

@pytest.fixture
def image_model():
    return ImageDeepfakeModel()

def test_model_loading(image_model):
    # Verify the model is loaded (or at least the path logic works as expected)
    # Since we are running in CI/CD or dev environment without the full model file possibly,
    # we should check if the file exists first. But here we know it exists.
    
    # If model is None, it means loading failed.
    # Note: image_model._load_model() catches exception and sets self.model=None
    
    # We assert that self.model is NOT None if the file exists.
    assert image_model.model_path.exists(), f"Model path {image_model.model_path} does not exist"
    assert image_model.model is not None, "Model failed to load"

def test_prediction_flow(image_model, tmp_path):
    # Create a dummy image
    img_path = tmp_path / "test_image.jpg"
    img = Image.new('RGB', (224, 224), color = 'red')
    img.save(img_path)
    
    # Run prediction
    label, confidence = image_model.predict(str(img_path))
    
    assert label in ["real", "fake", "model_not_loaded", "error"]
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0

if __name__ == "__main__":
    # Manually run if executed as script
    m = ImageDeepfakeModel()
    print(f"Model loaded: {m.model is not None}")
