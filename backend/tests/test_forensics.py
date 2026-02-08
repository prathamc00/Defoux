"""
Tests for Forensic Feature Extractor

Tests FFT, color, noise, and compression artifact detection.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import sys

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.forensics.forensic_extractor import ForensicExtractor, get_forensic_extractor
from app.services.rag.evidence_schema import ForensicFeatures


@pytest.fixture
def forensic_extractor():
    return ForensicExtractor()


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image"""
    img_path = tmp_path / "test_image.jpg"
    # Create a simple gradient image
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        arr[i, :, 0] = i  # Red gradient
        arr[:, i, 1] = i  # Green gradient
    arr[:, :, 2] = 128  # Constant blue
    
    img = Image.fromarray(arr)
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def noisy_image(tmp_path):
    """Create a high-noise test image"""
    img_path = tmp_path / "noisy_image.jpg"
    # Random noise
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def smooth_image(tmp_path):
    """Create a very smooth/blurry test image"""
    img_path = tmp_path / "smooth_image.jpg"
    # Solid color with slight variation
    arr = np.full((224, 224, 3), 128, dtype=np.uint8)
    arr += np.random.randint(-5, 5, (224, 224, 3), dtype=np.int8).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(img_path)
    return str(img_path)


class TestForensicExtractor:
    """Tests for ForensicExtractor class"""
    
    def test_extract_all_returns_forensic_features(self, forensic_extractor, sample_image):
        """Test that extract_all returns ForensicFeatures object"""
        features = forensic_extractor.extract_all(sample_image)
        
        assert isinstance(features, ForensicFeatures)
        assert hasattr(features, 'fft_high_freq_ratio')
        assert hasattr(features, 'color_deviation_r')
        assert hasattr(features, 'color_deviation_g')
        assert hasattr(features, 'color_deviation_b')
        assert hasattr(features, 'noise_variance')
        assert hasattr(features, 'compression_artifact_score')
    
    def test_fft_values_in_range(self, forensic_extractor, sample_image):
        """Test that FFT ratio is in valid range"""
        features = forensic_extractor.extract_all(sample_image)
        
        assert 0.0 <= features.fft_high_freq_ratio <= 1.0
    
    def test_color_values_in_range(self, forensic_extractor, sample_image):
        """Test that color deviation values are in valid range"""
        features = forensic_extractor.extract_all(sample_image)
        
        assert 0.0 <= features.color_deviation_r <= 1.0
        assert 0.0 <= features.color_deviation_g <= 1.0
        assert 0.0 <= features.color_deviation_b <= 1.0
    
    def test_noise_values_in_range(self, forensic_extractor, sample_image):
        """Test that noise variance is in valid range"""
        features = forensic_extractor.extract_all(sample_image)
        
        assert 0.0 <= features.noise_variance <= 1.0
    
    def test_compression_values_in_range(self, forensic_extractor, sample_image):
        """Test that compression score is in valid range"""
        features = forensic_extractor.extract_all(sample_image)
        
        assert 0.0 <= features.compression_artifact_score <= 1.0
    
    def test_noisy_image_has_high_variance(self, forensic_extractor, noisy_image):
        """Test that random noise image has high noise variance"""
        features = forensic_extractor.extract_all(noisy_image)
        
        # Random noise should have high variance
        assert features.noise_variance > 0.3
    
    def test_smooth_image_has_low_variance(self, forensic_extractor, smooth_image):
        """Test that smooth image has low noise variance"""
        features = forensic_extractor.extract_all(smooth_image)
        
        # Smooth images should have low variance
        assert features.noise_variance < 0.5
    
    def test_feature_summary_returns_dict(self, forensic_extractor, sample_image):
        """Test that get_feature_summary returns proper dictionary"""
        features = forensic_extractor.extract_all(sample_image)
        summary = forensic_extractor.get_feature_summary(features)
        
        assert isinstance(summary, dict)
        assert "fft_analysis" in summary
        assert "color_analysis" in summary
        assert "noise_analysis" in summary
        assert "compression_analysis" in summary


class TestForensicExtractorSingleton:
    """Tests for singleton pattern"""
    
    def test_get_forensic_extractor_returns_same_instance(self):
        """Test that get_forensic_extractor returns same instance"""
        extractor1 = get_forensic_extractor()
        extractor2 = get_forensic_extractor()
        
        assert extractor1 is extractor2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
