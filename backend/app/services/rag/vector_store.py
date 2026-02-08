"""
Vector Store Service for RAG-based Deepfake Detection

Wraps ChromaDB for storing and retrieving evidence embeddings.
Persists data to ml/rag_index/ for durability across restarts.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import uuid

from app.services.rag.evidence_schema import (
    EvidenceRecord, 
    SimilarCase, 
    ForensicFeatures,
    DeepfakeMethod,
    SourceDataset
)


class VectorStore:
    """ChromaDB-backed vector store for deepfake evidence"""
    
    COLLECTION_NAME = "deepfake_evidence"
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Path to persist ChromaDB data. 
                             Defaults to ml/rag_index/ relative to project root.
        """
        if persist_directory is None:
            # Default: project_root/ml/rag_index/
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[4]  # backend/app/services/rag -> project root
            persist_directory = str(project_root / "ml" / "rag_index")
        
        self.persist_path = Path(persist_directory)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self._client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create the evidence collection
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Deepfake detection evidence embeddings"}
        )
        
        print(f"VectorStore initialized. Path: {self.persist_path}")
        print(f"Collection '{self.COLLECTION_NAME}' has {self._collection.count()} records")
    
    def add_evidence(self, record: EvidenceRecord) -> str:
        """
        Add a single evidence record to the store.
        
        Args:
            record: EvidenceRecord to store
            
        Returns:
            The record ID
        """
        # Prepare metadata (ChromaDB stores metadata as flat dict)
        metadata = {
            "source_dataset": record.source_dataset.value,
            "label": record.label,
            "method": record.method.value,
            "fft_high_freq_ratio": record.forensic_features.fft_high_freq_ratio,
            "color_deviation_r": record.forensic_features.color_deviation_r,
            "color_deviation_g": record.forensic_features.color_deviation_g,
            "color_deviation_b": record.forensic_features.color_deviation_b,
            "noise_variance": record.forensic_features.noise_variance,
            "compression_artifact_score": record.forensic_features.compression_artifact_score,
        }
        
        if record.image_path:
            metadata["image_path"] = record.image_path
        
        self._collection.add(
            ids=[record.id],
            embeddings=[record.embedding],
            metadatas=[metadata]
        )
        
        return record.id
    
    def add_evidence_batch(self, records: List[EvidenceRecord]) -> List[str]:
        """
        Add multiple evidence records in batch.
        
        Args:
            records: List of EvidenceRecords to store
            
        Returns:
            List of record IDs
        """
        ids = []
        embeddings = []
        metadatas = []
        
        for record in records:
            ids.append(record.id)
            embeddings.append(record.embedding)
            metadatas.append({
                "source_dataset": record.source_dataset.value,
                "label": record.label,
                "method": record.method.value,
                "fft_high_freq_ratio": record.forensic_features.fft_high_freq_ratio,
                "color_deviation_r": record.forensic_features.color_deviation_r,
                "color_deviation_g": record.forensic_features.color_deviation_g,
                "color_deviation_b": record.forensic_features.color_deviation_b,
                "noise_variance": record.forensic_features.noise_variance,
                "compression_artifact_score": record.forensic_features.compression_artifact_score,
                **({"image_path": record.image_path} if record.image_path else {})
            })
        
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return ids
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        filter_label: Optional[str] = None
    ) -> List[SimilarCase]:
        """
        Search for similar evidence records.
        
        Args:
            query_embedding: The query embedding vector
            k: Number of results to return
            filter_label: Optional filter by label ('real' or 'fake')
            
        Returns:
            List of SimilarCase objects sorted by similarity
        """
        where_filter = None
        if filter_label:
            where_filter = {"label": filter_label}
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["metadatas", "distances"]
        )
        
        similar_cases = []
        
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, record_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                # ChromaDB returns L2 distance, convert to similarity
                # similarity = 1 / (1 + distance) for L2
                distance = results["distances"][0][i]
                similarity = 1.0 / (1.0 + distance)
                
                forensic_features = ForensicFeatures(
                    fft_high_freq_ratio=metadata.get("fft_high_freq_ratio", 0.0),
                    color_deviation_r=metadata.get("color_deviation_r", 0.0),
                    color_deviation_g=metadata.get("color_deviation_g", 0.0),
                    color_deviation_b=metadata.get("color_deviation_b", 0.0),
                    noise_variance=metadata.get("noise_variance", 0.0),
                    compression_artifact_score=metadata.get("compression_artifact_score", 0.0)
                )
                
                similar_cases.append(SimilarCase(
                    record_id=record_id,
                    similarity=round(similarity, 4),
                    source_dataset=metadata.get("source_dataset", "Unknown"),
                    label=metadata.get("label", "unknown"),
                    method=metadata.get("method", "Unknown"),
                    forensic_features=forensic_features
                ))
        
        return similar_cases
    
    def get_record_count(self) -> int:
        """Get total number of records in the store"""
        return self._collection.count()
    
    def delete_all(self) -> None:
        """Delete all records from the collection (use with caution)"""
        # ChromaDB doesn't have a clear() method, so we delete and recreate
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Deepfake detection evidence embeddings"}
        )
        print(f"All records deleted from '{self.COLLECTION_NAME}'")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the stored evidence"""
        count = self._collection.count()
        
        if count == 0:
            return {
                "total_records": 0,
                "real_count": 0,
                "fake_count": 0,
                "datasets": [],
                "methods": []
            }
        
        # Sample to get distribution (ChromaDB limitation)
        sample = self._collection.get(
            limit=min(count, 1000),
            include=["metadatas"]
        )
        
        real_count = sum(1 for m in sample["metadatas"] if m.get("label") == "real")
        fake_count = sum(1 for m in sample["metadatas"] if m.get("label") == "fake")
        datasets = list(set(m.get("source_dataset", "Unknown") for m in sample["metadatas"]))
        methods = list(set(m.get("method", "Unknown") for m in sample["metadatas"]))
        
        return {
            "total_records": count,
            "real_count": real_count,
            "fake_count": fake_count,
            "datasets": datasets,
            "methods": methods
        }


# Singleton instance for application-wide use
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
