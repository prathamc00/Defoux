"""
Evidence Fusion Engine for RAG-based Deepfake Detection

Combines vision model predictions with RAG evidence and forensic signals
to produce a final fused prediction with adjusted confidence.
"""

from typing import List, Dict, Optional
import numpy as np

from app.services.rag.evidence_schema import (
    SimilarCase,
    ForensicFeatures,
    FusedPrediction
)


class FusionEngine:
    """
    Fuses vision model output with RAG evidence and forensic signals.
    
    The fusion strategy:
    1. Base prediction comes from the vision model
    2. RAG evidence adjusts confidence based on similar cases
    3. Forensic signals provide additional corroboration
    """
    
    def __init__(
        self,
        vision_weight: float = 0.6,
        rag_weight: float = 0.25,
        forensic_weight: float = 0.15
    ):
        """
        Initialize the fusion engine with component weights.
        
        Args:
            vision_weight: Weight for vision model prediction (default 0.6)
            rag_weight: Weight for RAG evidence (default 0.25)
            forensic_weight: Weight for forensic signals (default 0.15)
        """
        # Normalize weights
        total = vision_weight + rag_weight + forensic_weight
        self.vision_weight = vision_weight / total
        self.rag_weight = rag_weight / total
        self.forensic_weight = forensic_weight / total
        
        # Forensic signal thresholds
        self.fft_suspicious_threshold = 0.6
        self.color_suspicious_threshold = 0.5
        self.noise_suspicious_low = 0.2  # Too blurry
        self.noise_suspicious_high = 0.85  # Unusual noise
        self.compression_suspicious_threshold = 0.5
    
    def fuse(
        self,
        vision_label: str,
        vision_confidence: float,
        similar_cases: List[SimilarCase],
        forensic_signals: ForensicFeatures
    ) -> FusedPrediction:
        """
        Fuse all evidence sources into a final prediction.
        
        Args:
            vision_label: Label from vision model ('real' or 'fake')
            vision_confidence: Confidence from vision model
            similar_cases: List of similar cases from RAG
            forensic_signals: Extracted forensic features
            
        Returns:
            FusedPrediction with combined result and explanation data
        """
        # Calculate RAG score
        rag_score, rag_label = self._calculate_rag_score(similar_cases)
        
        # Calculate forensic score
        forensic_score = self._calculate_forensic_score(forensic_signals)
        
        # Convert vision prediction to score (0 = real, 1 = fake)
        vision_fake_score = vision_confidence if vision_label == "fake" else (1 - vision_confidence)
        
        # Determine effective weights based on RAG availability
        if rag_score < 0:
            # No RAG data available - redistribute RAG weight to vision and forensic
            # Give more weight to vision model since it's the primary detector
            effective_vision_weight = self.vision_weight + self.rag_weight * 0.8
            effective_forensic_weight = self.forensic_weight + self.rag_weight * 0.2
            effective_rag_weight = 0.0
            effective_rag_score = 0.0  # Not used, but set for clarity
            rag_adjustment = 0.0  # No RAG adjustment when no data
        else:
            effective_vision_weight = self.vision_weight
            effective_forensic_weight = self.forensic_weight
            effective_rag_weight = self.rag_weight
            effective_rag_score = rag_score
            rag_adjustment = (rag_score - vision_fake_score) * self.rag_weight
        
        # Weighted fusion
        fused_fake_score = (
            effective_vision_weight * vision_fake_score +
            effective_rag_weight * effective_rag_score +
            effective_forensic_weight * forensic_score
        )
        
        # Determine final label and confidence
        if fused_fake_score > 0.5:
            final_label = "fake"
            final_confidence = fused_fake_score
        else:
            final_label = "real"
            final_confidence = 1.0 - fused_fake_score
        
        return FusedPrediction(
            result=final_label,
            confidence=round(final_confidence, 4),
            vision_score=round(vision_confidence, 4),
            rag_adjustment=round(rag_adjustment, 4),
            similar_cases=similar_cases,
            forensic_signals=forensic_signals
        )
    
    def _calculate_rag_score(self, similar_cases: List[SimilarCase]) -> tuple:
        """
        Calculate fake probability based on similar cases.
        
        Uses weighted voting where weight = similarity score.
        
        Returns:
            Tuple of (fake_score, majority_label)
            Returns (-1.0, "no_data") when no similar cases exist,
            signaling that RAG should not influence the final prediction.
        """
        if not similar_cases:
            return -1.0, "no_data"  # Signal to skip RAG influence
        
        fake_weight = 0.0
        real_weight = 0.0
        
        for case in similar_cases:
            if case.label == "fake":
                fake_weight += case.similarity
            else:
                real_weight += case.similarity
        
        total_weight = fake_weight + real_weight
        
        if total_weight == 0:
            return -1.0, "no_data"  # Signal to skip RAG influence
        
        fake_score = fake_weight / total_weight
        majority_label = "fake" if fake_score > 0.5 else "real"
        
        return fake_score, majority_label
    
    def _calculate_forensic_score(self, features: ForensicFeatures) -> float:
        """
        Calculate suspicion score from forensic signals.
        
        Higher score = more suspicious (likely fake).
        
        Returns:
            Score between 0.0 (likely real) and 1.0 (likely fake)
        """
        suspicion_points = 0.0
        max_points = 4.0  # Four forensic checks
        
        # FFT check
        if features.fft_high_freq_ratio > self.fft_suspicious_threshold:
            suspicion_points += 1.0
        elif features.fft_high_freq_ratio > 0.4:
            suspicion_points += 0.5
        
        # Color check
        avg_color_dev = (
            features.color_deviation_r + 
            features.color_deviation_g + 
            features.color_deviation_b
        ) / 3
        if avg_color_dev > self.color_suspicious_threshold:
            suspicion_points += 1.0
        elif avg_color_dev > 0.35:
            suspicion_points += 0.5
        
        # Noise check (suspicious if too smooth OR too noisy)
        if features.noise_variance < self.noise_suspicious_low:
            suspicion_points += 1.0  # Suspicious smoothing
        elif features.noise_variance > self.noise_suspicious_high:
            suspicion_points += 0.7  # Unusual noise
        
        # Compression check
        if features.compression_artifact_score > self.compression_suspicious_threshold:
            suspicion_points += 1.0
        elif features.compression_artifact_score > 0.3:
            suspicion_points += 0.5
        
        return suspicion_points / max_points


# Singleton instance
_fusion_engine: Optional[FusionEngine] = None


def get_fusion_engine() -> FusionEngine:
    """Get or create the singleton fusion engine instance"""
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = FusionEngine()
    return _fusion_engine
