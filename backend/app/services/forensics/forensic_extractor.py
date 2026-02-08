"""
Forensic Feature Extractor for Deepfake Detection

Extracts auxiliary forensic signals from images:
- FFT frequency spectrum analysis
- Color channel inconsistency detection
- Blur & noise statistics
- Compression artifact detection
"""

import numpy as np
from PIL import Image
from scipy import fftpack
from scipy.ndimage import laplace
from typing import Dict, Tuple
import cv2

from app.services.rag.evidence_schema import ForensicFeatures


class ForensicExtractor:
    """Extracts forensic signals from images for deepfake detection"""
    
    def __init__(self):
        """Initialize the forensic extractor"""
        # Thresholds calibrated from research papers
        self.fft_cutoff_ratio = 0.3  # High-freq region starts at 30% from center
    
    def extract_all(self, image_path: str) -> ForensicFeatures:
        """
        Extract all forensic features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ForensicFeatures object with all extracted signals
        """
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # Extract individual features
        fft_ratio = self._extract_fft_features(img_array)
        color_deviations = self._extract_color_features(img_array)
        noise_var = self._extract_noise_features(img_array)
        compression_score = self._extract_compression_features(img_array)
        
        return ForensicFeatures(
            fft_high_freq_ratio=fft_ratio,
            color_deviation_r=color_deviations[0],
            color_deviation_g=color_deviations[1],
            color_deviation_b=color_deviations[2],
            noise_variance=noise_var,
            compression_artifact_score=compression_score
        )
    
    def _extract_fft_features(self, img_array: np.ndarray) -> float:
        """
        Extract FFT frequency spectrum features.
        
        Deepfakes often show unnatural high-frequency patterns due to
        upsampling artifacts from generators.
        
        Returns:
            Ratio of high-frequency energy to total energy (0.0 - 1.0)
        """
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Compute 2D FFT
        fft = fftpack.fft2(gray)
        fft_shifted = fftpack.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # Create mask for high-frequency region
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # High-frequency region (outer ring)
        y, x = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
        max_dist = np.sqrt(crow**2 + ccol**2)
        
        high_freq_mask = dist_from_center > (max_dist * self.fft_cutoff_ratio)
        
        # Calculate energy ratio
        total_energy = np.sum(magnitude**2)
        high_freq_energy = np.sum(magnitude[high_freq_mask]**2)
        
        if total_energy > 0:
            ratio = high_freq_energy / total_energy
        else:
            ratio = 0.0
        
        return float(np.clip(ratio, 0.0, 1.0))
    
    def _extract_color_features(self, img_array: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract color channel inconsistency features.
        
        Deepfakes may show unusual color distributions due to
        imperfect color matching between source and target.
        
        Returns:
            Tuple of (R, G, B) deviation scores (0.0 - 1.0 each)
        """
        # Split channels
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Calculate statistics for each channel
        r_std = np.std(r) / 127.5  # Normalize by half of max value
        g_std = np.std(g) / 127.5
        b_std = np.std(b) / 127.5
        
        # Calculate channel-wise histogram differences
        # Authentic images typically have correlated color channels
        rg_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
        gb_corr = np.corrcoef(g.flatten(), b.flatten())[0, 1]
        rb_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
        
        # Low correlation indicates potential manipulation
        avg_corr = (abs(rg_corr) + abs(gb_corr) + abs(rb_corr)) / 3
        
        # Deviation = inverse of correlation (higher = more suspicious)
        color_deviation_base = 1.0 - avg_corr if not np.isnan(avg_corr) else 0.5
        
        # Combine with std deviation
        r_deviation = float(np.clip(r_std * color_deviation_base, 0.0, 1.0))
        g_deviation = float(np.clip(g_std * color_deviation_base, 0.0, 1.0))
        b_deviation = float(np.clip(b_std * color_deviation_base, 0.0, 1.0))
        
        return (r_deviation, g_deviation, b_deviation)
    
    def _extract_noise_features(self, img_array: np.ndarray) -> float:
        """
        Extract noise/blur statistics using Laplacian variance.
        
        Deepfakes often have inconsistent noise patterns or
        blurred regions where faces are blended.
        
        Returns:
            Normalized noise variance score (0.0 - 1.0)
        """
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Apply Laplacian operator
        laplacian = laplace(gray)
        
        # Calculate variance (higher = more edges/less blur)
        variance = np.var(laplacian)
        
        # Normalize (empirically, variance > 500 is sharp, < 100 is blurry)
        normalized = float(np.clip(variance / 500.0, 0.0, 1.0))
        
        return normalized
    
    def _extract_compression_features(self, img_array: np.ndarray) -> float:
        """
        Detect JPEG compression artifacts.
        
        Multiple compression cycles (source + output) can leave
        detectable 8x8 block boundary artifacts.
        
        Returns:
            Compression artifact density score (0.0 - 1.0)
        """
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = img_array[:, :, 0].astype(np.float32)
        else:
            gray = img_array.astype(np.float32)
        
        height, width = gray.shape
        
        if height < 16 or width < 16:
            return 0.0
        
        # Detect 8x8 block boundaries
        block_size = 8
        
        # Calculate differences at block boundaries vs within blocks
        boundary_diffs = []
        internal_diffs = []
        
        for y in range(0, height - 1):
            for x in range(0, width - 1):
                diff = abs(float(gray[y, x]) - float(gray[y, x + 1]))
                
                # Check if this is a block boundary
                if (x + 1) % block_size == 0:
                    boundary_diffs.append(diff)
                else:
                    internal_diffs.append(diff)
        
        if not boundary_diffs or not internal_diffs:
            return 0.0
        
        # Compare boundary vs internal differences
        avg_boundary = np.mean(boundary_diffs)
        avg_internal = np.mean(internal_diffs)
        
        # Higher ratio = more block artifacts
        if avg_internal > 0:
            artifact_ratio = avg_boundary / avg_internal
            # Normalize (ratio > 1.2 indicates visible artifacts)
            score = float(np.clip((artifact_ratio - 1.0) * 5.0, 0.0, 1.0))
        else:
            score = 0.0
        
        return score
    
    def get_feature_summary(self, features: ForensicFeatures) -> Dict[str, str]:
        """
        Generate human-readable summary of forensic features.
        
        Args:
            features: Extracted ForensicFeatures
            
        Returns:
            Dictionary of feature names to human-readable descriptions
        """
        summary = {}
        
        # FFT analysis
        if features.fft_high_freq_ratio > 0.7:
            summary["fft_analysis"] = "Abnormal high-frequency patterns detected (likely GAN artifacts)"
        elif features.fft_high_freq_ratio > 0.5:
            summary["fft_analysis"] = "Elevated high-frequency content (potential manipulation)"
        else:
            summary["fft_analysis"] = "Normal frequency distribution"
        
        # Color analysis
        avg_color_dev = (features.color_deviation_r + features.color_deviation_g + features.color_deviation_b) / 3
        if avg_color_dev > 0.6:
            summary["color_analysis"] = "Significant color channel inconsistencies"
        elif avg_color_dev > 0.4:
            summary["color_analysis"] = "Minor color anomalies detected"
        else:
            summary["color_analysis"] = "Color channels appear consistent"
        
        # Noise analysis
        if features.noise_variance < 0.2:
            summary["noise_analysis"] = "Unusual blur/smoothing detected (possible face blending)"
        elif features.noise_variance > 0.8:
            summary["noise_analysis"] = "High noise variance (possible post-processing)"
        else:
            summary["noise_analysis"] = "Normal noise levels"
        
        # Compression analysis
        if features.compression_artifact_score > 0.6:
            summary["compression_analysis"] = "Strong JPEG artifacts (multiple compression cycles)"
        elif features.compression_artifact_score > 0.3:
            summary["compression_analysis"] = "Moderate compression artifacts"
        else:
            summary["compression_analysis"] = "Minimal compression artifacts"
        
        return summary


# Singleton instance
_forensic_extractor: ForensicExtractor = None


def get_forensic_extractor() -> ForensicExtractor:
    """Get or create the singleton forensic extractor instance"""
    global _forensic_extractor
    if _forensic_extractor is None:
        _forensic_extractor = ForensicExtractor()
    return _forensic_extractor
