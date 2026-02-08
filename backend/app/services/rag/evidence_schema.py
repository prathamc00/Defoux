"""
Evidence Schema for RAG-based Deepfake Detection

Defines data models for storing and retrieving evidence records
from the vector database.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum


class DeepfakeMethod(str, Enum):
    """Known deepfake generation methods"""
    DEEPFAKES = "Deepfakes"
    FACE2FACE = "Face2Face"
    FACESWAP = "FaceSwap"
    NEURALTEXTURES = "NeuralTextures"
    FACEREENACTMENT = "FaceReenactment"
    STYLEGAN = "StyleGAN"
    PHOTOSHOP = "Photoshop"
    UNKNOWN = "Unknown"
    REAL = "Real"


class SourceDataset(str, Enum):
    """Supported source datasets"""
    FACEFORENSICS = "FaceForensics++"
    DFDC = "DFDC"
    CELEB_DF = "Celeb-DF"
    REAL_VS_FAKE_140K = "140k-Real-vs-Fake"
    REAL_VS_FAKE_UDIT = "RealVsFake-Udit"
    CUSTOM = "Custom"


class ForensicFeatures(BaseModel):
    """Forensic signal features extracted from an image"""
    fft_high_freq_ratio: float = Field(
        description="Ratio of high-frequency energy in FFT spectrum"
    )
    color_deviation_r: float = Field(
        description="Red channel deviation from expected distribution"
    )
    color_deviation_g: float = Field(
        description="Green channel deviation"
    )
    color_deviation_b: float = Field(
        description="Blue channel deviation"
    )
    noise_variance: float = Field(
        description="Laplacian variance indicating blur/noise level"
    )
    compression_artifact_score: float = Field(
        description="JPEG block boundary artifact density"
    )


class EvidenceRecord(BaseModel):
    """A single evidence record stored in the vector database"""
    id: str = Field(description="Unique identifier")
    embedding: List[float] = Field(description="Feature embedding vector")
    source_dataset: SourceDataset = Field(description="Origin dataset")
    label: str = Field(description="Ground truth: 'real' or 'fake'")
    method: DeepfakeMethod = Field(description="Deepfake generation method")
    forensic_features: ForensicFeatures = Field(
        description="Extracted forensic signals"
    )
    image_path: Optional[str] = Field(
        default=None, 
        description="Original image path for reference"
    )
    metadata: Optional[Dict] = Field(
        default=None,
        description="Additional metadata"
    )


class SimilarCase(BaseModel):
    """A similar case retrieved from the vector database"""
    record_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    source_dataset: str
    label: str
    method: str
    forensic_features: Optional[ForensicFeatures] = None


class FusedPrediction(BaseModel):
    """Final prediction after evidence fusion"""
    result: str = Field(description="'real' or 'fake'")
    confidence: float = Field(ge=0.0, le=1.0)
    vision_score: float = Field(description="Raw vision model score")
    rag_adjustment: float = Field(
        description="Confidence adjustment from RAG evidence"
    )
    similar_cases: List[SimilarCase] = Field(
        description="Top-k similar cases from database"
    )
    forensic_signals: ForensicFeatures = Field(
        description="Extracted forensic features"
    )


class ExplanationReport(BaseModel):
    """Human-readable explanation of the prediction"""
    summary: str = Field(description="One-line summary")
    confidence_reasoning: str = Field(
        description="Why this confidence level"
    )
    forensic_evidence: List[str] = Field(
        description="List of forensic findings"
    )
    similar_cases_summary: str = Field(
        description="Summary of retrieved similar cases"
    )
    gradcam_url: Optional[str] = Field(
        default=None,
        description="URL to Grad-CAM visualization"
    )
