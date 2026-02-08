"""
RAG-Enhanced Detection API

New v2 endpoint that provides explainable deepfake detection
with evidence from similar cases and forensic analysis.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
import os
import tempfile
import shutil

from app.models.image_model import ImageDeepfakeModel
from app.services.preprocessing.image import preprocess_image
from app.services.forensics.forensic_extractor import get_forensic_extractor
from app.services.rag.vector_store import get_vector_store
from app.services.rag.fusion_engine import get_fusion_engine
from app.services.explainability.explanation_generator import get_explanation_generator


router = APIRouter(prefix="/api/v2", tags=["RAG Detection"])

# Initialize components
_model = ImageDeepfakeModel()


# Response schemas
class SimilarCaseResponse(BaseModel):
    """A similar case from the evidence database"""
    dataset: str
    method: str
    similarity: float
    label: str


class ForensicEvidenceResponse(BaseModel):
    """Forensic analysis results"""
    fft_anomaly: bool
    fft_score: float
    color_anomaly: bool
    color_score: float
    noise_anomaly: bool
    noise_score: float
    compression_artifacts: str  # "low", "medium", "high"
    compression_score: float


class ExplanationResponse(BaseModel):
    """Explanation of the prediction"""
    summary: str
    confidence_reasoning: str
    forensic_findings: List[str]
    similar_cases_summary: str
    similar_cases: List[SimilarCaseResponse]
    forensic_evidence: ForensicEvidenceResponse
    gradcam_url: Optional[str] = None


class RAGDetectionResponse(BaseModel):
    """Full RAG-enhanced detection response"""
    result: str = Field(description="'real' or 'fake'")
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: ExplanationResponse


class IndexStatsResponse(BaseModel):
    """Statistics about the evidence index"""
    total_records: int
    real_count: int
    fake_count: int
    datasets: List[str]
    methods: List[str]


@router.post("/detect", response_model=RAGDetectionResponse)
async def rag_detect(file: UploadFile = File(...)):
    """
    RAG-enhanced deepfake detection with explainability.
    
    Returns prediction with:
    - Confidence score
    - Similar cases from evidence database
    - Forensic analysis results
    - Human-readable explanation
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Preprocess (face detection)
        processed_path = preprocess_image(temp_path)
        
        # Get vision model prediction with embeddings
        label, confidence, embedding = _model.predict_with_features(processed_path)
        
        if label in ["model_not_loaded", "error"]:
            raise HTTPException(
                status_code=500,
                detail=f"Model error: {label}"
            )
        
        # Extract forensic features
        forensic_extractor = get_forensic_extractor()
        forensic_features = forensic_extractor.extract_all(processed_path)
        
        # Search for similar cases (if embedding available and index has data)
        vector_store = get_vector_store()
        similar_cases = []
        
        if embedding and vector_store.get_record_count() > 0:
            similar_cases = vector_store.search_similar(embedding, k=5)
        
        # Fuse evidence
        fusion_engine = get_fusion_engine()
        fused_prediction = fusion_engine.fuse(
            vision_label=label,
            vision_confidence=confidence,
            similar_cases=similar_cases,
            forensic_signals=forensic_features
        )
        
        # Generate explanation
        explanation_generator = get_explanation_generator()
        explanation_report = explanation_generator.generate(fused_prediction)
        
        # Build response
        similar_cases_response = [
            SimilarCaseResponse(
                dataset=case.source_dataset,
                method=case.method,
                similarity=case.similarity,
                label=case.label
            )
            for case in similar_cases
        ]
        
        # Determine forensic anomaly flags
        forensic_evidence = ForensicEvidenceResponse(
            fft_anomaly=forensic_features.fft_high_freq_ratio > 0.6,
            fft_score=round(forensic_features.fft_high_freq_ratio, 4),
            color_anomaly=(forensic_features.color_deviation_r + forensic_features.color_deviation_g + forensic_features.color_deviation_b) / 3 > 0.5,
            color_score=round((forensic_features.color_deviation_r + forensic_features.color_deviation_g + forensic_features.color_deviation_b) / 3, 4),
            noise_anomaly=forensic_features.noise_variance < 0.2 or forensic_features.noise_variance > 0.85,
            noise_score=round(forensic_features.noise_variance, 4),
            compression_artifacts="high" if forensic_features.compression_artifact_score > 0.6 else "medium" if forensic_features.compression_artifact_score > 0.3 else "low",
            compression_score=round(forensic_features.compression_artifact_score, 4)
        )
        
        explanation_response = ExplanationResponse(
            summary=explanation_report.summary,
            confidence_reasoning=explanation_report.confidence_reasoning,
            forensic_findings=explanation_report.forensic_evidence,
            similar_cases_summary=explanation_report.similar_cases_summary,
            similar_cases=similar_cases_response,
            forensic_evidence=forensic_evidence,
            gradcam_url=explanation_report.gradcam_url
        )
        
        return RAGDetectionResponse(
            result=fused_prediction.result,
            confidence=fused_prediction.confidence,
            explanation=explanation_response
        )
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/index/stats", response_model=IndexStatsResponse)
async def get_index_stats():
    """Get statistics about the evidence index"""
    vector_store = get_vector_store()
    stats = vector_store.get_statistics()
    
    return IndexStatsResponse(
        total_records=stats["total_records"],
        real_count=stats["real_count"],
        fake_count=stats["fake_count"],
        datasets=stats["datasets"],
        methods=stats["methods"]
    )


@router.delete("/index/clear")
async def clear_index():
    """Clear all records from the evidence index (use with caution)"""
    vector_store = get_vector_store()
    vector_store.delete_all()
    
    return {"message": "Index cleared successfully"}
