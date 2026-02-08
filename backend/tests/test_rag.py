"""
Tests for RAG System Components

Tests vector store, fusion engine, and evidence schema.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import sys
import shutil

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.rag.evidence_schema import (
    EvidenceRecord,
    SimilarCase,
    ForensicFeatures,
    FusedPrediction,
    SourceDataset,
    DeepfakeMethod
)
from app.services.rag.vector_store import VectorStore
from app.services.rag.fusion_engine import FusionEngine


@pytest.fixture
def sample_forensic_features():
    """Create sample forensic features"""
    return ForensicFeatures(
        fft_high_freq_ratio=0.45,
        color_deviation_r=0.3,
        color_deviation_g=0.25,
        color_deviation_b=0.28,
        noise_variance=0.6,
        compression_artifact_score=0.2
    )


@pytest.fixture
def temp_vector_store(tmp_path):
    """Create a temporary vector store for testing"""
    store = VectorStore(persist_directory=str(tmp_path / "test_rag_index"))
    yield store
    # Cleanup
    shutil.rmtree(tmp_path / "test_rag_index", ignore_errors=True)


@pytest.fixture
def sample_evidence_record(sample_forensic_features):
    """Create a sample evidence record"""
    return EvidenceRecord(
        id="test-001",
        embedding=[0.1, 0.2, 0.8, 0.9],
        source_dataset=SourceDataset.FACEFORENSICS,
        label="fake",
        method=DeepfakeMethod.DEEPFAKES,
        forensic_features=sample_forensic_features
    )


class TestEvidenceSchema:
    """Tests for evidence schema models"""
    
    def test_forensic_features_creation(self, sample_forensic_features):
        """Test ForensicFeatures model creation"""
        assert sample_forensic_features.fft_high_freq_ratio == 0.45
        assert sample_forensic_features.noise_variance == 0.6
    
    def test_evidence_record_creation(self, sample_evidence_record):
        """Test EvidenceRecord model creation"""
        assert sample_evidence_record.id == "test-001"
        assert sample_evidence_record.label == "fake"
        assert sample_evidence_record.source_dataset == SourceDataset.FACEFORENSICS
    
    def test_similar_case_creation(self):
        """Test SimilarCase model creation"""
        case = SimilarCase(
            record_id="test-001",
            similarity=0.85,
            source_dataset="FaceForensics++",
            label="fake",
            method="Deepfakes"
        )
        assert case.similarity == 0.85
        assert case.label == "fake"


class TestVectorStore:
    """Tests for VectorStore class"""
    
    def test_add_evidence(self, temp_vector_store, sample_evidence_record):
        """Test adding a single evidence record"""
        record_id = temp_vector_store.add_evidence(sample_evidence_record)
        
        assert record_id == "test-001"
        assert temp_vector_store.get_record_count() == 1
    
    def test_add_evidence_batch(self, temp_vector_store, sample_forensic_features):
        """Test adding multiple evidence records in batch"""
        records = [
            EvidenceRecord(
                id=f"batch-{i}",
                embedding=[0.1 * i, 0.2 * i, 0.8, 0.9],
                source_dataset=SourceDataset.DFDC,
                label="real" if i % 2 == 0 else "fake",
                method=DeepfakeMethod.UNKNOWN,
                forensic_features=sample_forensic_features
            )
            for i in range(5)
        ]
        
        ids = temp_vector_store.add_evidence_batch(records)
        
        assert len(ids) == 5
        assert temp_vector_store.get_record_count() == 5
    
    def test_search_similar(self, temp_vector_store, sample_forensic_features):
        """Test similarity search"""
        # Add some records first
        records = [
            EvidenceRecord(
                id=f"search-{i}",
                embedding=[float(i), 0.5, 0.5, 0.5],
                source_dataset=SourceDataset.FACEFORENSICS,
                label="fake",
                method=DeepfakeMethod.FACESWAP,
                forensic_features=sample_forensic_features
            )
            for i in range(3)
        ]
        temp_vector_store.add_evidence_batch(records)
        
        # Search with a query embedding
        query_embedding = [1.0, 0.5, 0.5, 0.5]
        results = temp_vector_store.search_similar(query_embedding, k=2)
        
        assert len(results) <= 2
        for result in results:
            assert isinstance(result, SimilarCase)
            assert 0.0 <= result.similarity <= 1.0
    
    def test_get_statistics(self, temp_vector_store, sample_evidence_record):
        """Test getting store statistics"""
        temp_vector_store.add_evidence(sample_evidence_record)
        stats = temp_vector_store.get_statistics()
        
        assert stats["total_records"] == 1
        assert "datasets" in stats
        assert "methods" in stats
    
    def test_delete_all(self, temp_vector_store, sample_evidence_record):
        """Test clearing the store"""
        temp_vector_store.add_evidence(sample_evidence_record)
        assert temp_vector_store.get_record_count() == 1
        
        temp_vector_store.delete_all()
        assert temp_vector_store.get_record_count() == 0


class TestFusionEngine:
    """Tests for FusionEngine class"""
    
    @pytest.fixture
    def fusion_engine(self):
        return FusionEngine()
    
    def test_fuse_with_no_similar_cases(self, fusion_engine, sample_forensic_features):
        """Test fusion when no similar cases are available"""
        result = fusion_engine.fuse(
            vision_label="fake",
            vision_confidence=0.85,
            similar_cases=[],
            forensic_signals=sample_forensic_features
        )
        
        assert isinstance(result, FusedPrediction)
        assert result.result in ["real", "fake"]
        assert 0.0 <= result.confidence <= 1.0
    
    def test_fuse_with_similar_cases(self, fusion_engine, sample_forensic_features):
        """Test fusion with similar cases"""
        similar_cases = [
            SimilarCase(
                record_id="case-1",
                similarity=0.9,
                source_dataset="FaceForensics++",
                label="fake",
                method="Deepfakes"
            ),
            SimilarCase(
                record_id="case-2",
                similarity=0.75,
                source_dataset="DFDC",
                label="fake",
                method="FaceSwap"
            )
        ]
        
        result = fusion_engine.fuse(
            vision_label="fake",
            vision_confidence=0.7,
            similar_cases=similar_cases,
            forensic_signals=sample_forensic_features
        )
        
        assert result.result == "fake"  # Corroborated by similar cases
        assert len(result.similar_cases) == 2
    
    def test_fuse_weights_sum_to_one(self, fusion_engine, sample_forensic_features):
        """Test that fusion weights are normalized"""
        total = fusion_engine.vision_weight + fusion_engine.rag_weight + fusion_engine.forensic_weight
        assert abs(total - 1.0) < 0.001
    
    def test_fuse_real_prediction(self, fusion_engine, sample_forensic_features):
        """Test fusion for real prediction"""
        result = fusion_engine.fuse(
            vision_label="real",
            vision_confidence=0.95,
            similar_cases=[],
            forensic_signals=sample_forensic_features
        )
        
        # High confidence real prediction with no contrary evidence
        assert result.result == "real"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
