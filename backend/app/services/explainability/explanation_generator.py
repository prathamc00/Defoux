"""
Explanation Generator for RAG-based Deepfake Detection

Generates human-readable explanations of predictions based on
vision model output, similar cases, and forensic evidence.
"""

from typing import List, Dict, Optional

from app.services.rag.evidence_schema import (
    SimilarCase,
    ForensicFeatures,
    FusedPrediction,
    ExplanationReport
)
from app.services.forensics.forensic_extractor import get_forensic_extractor


class ExplanationGenerator:
    """Generates human-readable explanations for deepfake predictions"""
    
    def __init__(self):
        self.forensic_extractor = get_forensic_extractor()
    
    def generate(
        self,
        prediction: FusedPrediction,
        gradcam_url: Optional[str] = None
    ) -> ExplanationReport:
        """
        Generate a comprehensive explanation report.
        
        Args:
            prediction: The fused prediction result
            gradcam_url: Optional URL to Grad-CAM visualization
            
        Returns:
            ExplanationReport with human-readable explanations
        """
        # Generate summary
        summary = self._generate_summary(prediction)
        
        # Generate confidence reasoning
        confidence_reasoning = self._generate_confidence_reasoning(prediction)
        
        # Generate forensic evidence list
        forensic_evidence = self._generate_forensic_evidence(prediction.forensic_signals)
        
        # Generate similar cases summary
        similar_cases_summary = self._generate_similar_cases_summary(prediction.similar_cases)
        
        return ExplanationReport(
            summary=summary,
            confidence_reasoning=confidence_reasoning,
            forensic_evidence=forensic_evidence,
            similar_cases_summary=similar_cases_summary,
            gradcam_url=gradcam_url
        )
    
    def _generate_summary(self, prediction: FusedPrediction) -> str:
        """Generate one-line summary"""
        confidence_level = self._get_confidence_level(prediction.confidence)
        
        if prediction.result == "fake":
            if confidence_level == "high":
                return "Image shows strong indicators of manipulation"
            elif confidence_level == "medium":
                return "Image likely contains manipulated content"
            else:
                return "Image may contain subtle manipulation"
        else:
            if confidence_level == "high":
                return "Image appears to be authentic"
            elif confidence_level == "medium":
                return "Image is likely authentic"
            else:
                return "No strong manipulation indicators detected"
    
    def _generate_confidence_reasoning(self, prediction: FusedPrediction) -> str:
        """Explain why this confidence level was assigned"""
        reasons = []
        
        # Vision model contribution
        if prediction.vision_score > 0.8:
            reasons.append(f"Vision model is highly confident ({prediction.vision_score:.0%})")
        elif prediction.vision_score > 0.6:
            reasons.append(f"Vision model shows moderate confidence ({prediction.vision_score:.0%})")
        else:
            reasons.append(f"Vision model shows low confidence ({prediction.vision_score:.0%})")
        
        # RAG contribution
        if abs(prediction.rag_adjustment) > 0.1:
            direction = "increased" if prediction.rag_adjustment > 0 else "decreased"
            reasons.append(f"Similar case evidence {direction} fake probability by {abs(prediction.rag_adjustment):.0%}")
        elif prediction.similar_cases:
            reasons.append("Similar cases provide corroborating evidence")
        else:
            reasons.append("No similar cases found in database")
        
        # Forensic contribution
        forensic_score = self._calculate_forensic_suspicion(prediction.forensic_signals)
        if forensic_score > 0.6:
            reasons.append("Forensic analysis detected multiple manipulation indicators")
        elif forensic_score > 0.3:
            reasons.append("Forensic analysis found some suspicious patterns")
        else:
            reasons.append("Forensic analysis found no significant anomalies")
        
        return ". ".join(reasons) + "."
    
    def _generate_forensic_evidence(self, features: ForensicFeatures) -> List[str]:
        """Generate list of forensic findings"""
        evidence = []
        feature_summary = self.forensic_extractor.get_feature_summary(features)
        
        for key, description in feature_summary.items():
            # Only include notable findings
            if any(x in description.lower() for x in ["abnormal", "significant", "unusual", "strong", "detected", "elevated"]):
                evidence.append(description)
        
        if not evidence:
            evidence.append("No significant forensic anomalies detected")
        
        return evidence
    
    def _generate_similar_cases_summary(self, similar_cases: List[SimilarCase]) -> str:
        """Summarize the similar cases from RAG"""
        if not similar_cases:
            return "No similar cases found in the evidence database"
        
        fake_count = sum(1 for c in similar_cases if c.label == "fake")
        real_count = len(similar_cases) - fake_count
        
        # Group by dataset
        datasets = {}
        for case in similar_cases:
            ds = case.source_dataset
            if ds not in datasets:
                datasets[ds] = {"fake": 0, "real": 0}
            datasets[ds][case.label] += 1
        
        # Build summary
        parts = [f"Found {len(similar_cases)} similar cases"]
        
        if fake_count > real_count:
            avg_similarity = sum(c.similarity for c in similar_cases if c.label == "fake") / max(fake_count, 1)
            parts.append(f"({fake_count} fake, {real_count} real)")
            parts.append(f"with avg similarity {avg_similarity:.0%} to known fakes")
        elif real_count > fake_count:
            avg_similarity = sum(c.similarity for c in similar_cases if c.label == "real") / max(real_count, 1)
            parts.append(f"({real_count} real, {fake_count} fake)")
            parts.append(f"with avg similarity {avg_similarity:.0%} to authentic samples")
        else:
            parts.append(f"(evenly split: {fake_count} fake, {real_count} real)")
        
        # Add dataset info
        if datasets:
            ds_info = ", ".join(f"{k}: {v['fake']+v['real']}" for k, v in datasets.items())
            parts.append(f"from: {ds_info}")
        
        return " ".join(parts)
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to level"""
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_forensic_suspicion(self, features: ForensicFeatures) -> float:
        """Calculate overall suspicion score from forensic features"""
        scores = [
            min(features.fft_high_freq_ratio / 0.7, 1.0),
            (features.color_deviation_r + features.color_deviation_g + features.color_deviation_b) / 3,
            1.0 - features.noise_variance if features.noise_variance < 0.3 else 0.0,
            features.compression_artifact_score
        ]
        return sum(scores) / len(scores)


# Singleton instance
_explanation_generator: Optional[ExplanationGenerator] = None


def get_explanation_generator() -> ExplanationGenerator:
    """Get or create the singleton explanation generator instance"""
    global _explanation_generator
    if _explanation_generator is None:
        _explanation_generator = ExplanationGenerator()
    return _explanation_generator
