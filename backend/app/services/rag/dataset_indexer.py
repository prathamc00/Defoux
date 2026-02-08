"""
Dataset Indexer for RAG-based Deepfake Detection

Utility for batch indexing images from known deepfake datasets
into the vector store for similarity matching.
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional, Generator
from dataclasses import dataclass
from tqdm import tqdm

from app.models.image_model import ImageDeepfakeModel
from app.services.forensics.forensic_extractor import get_forensic_extractor
from app.services.rag.vector_store import get_vector_store
from app.services.rag.evidence_schema import (
    EvidenceRecord,
    ForensicFeatures,
    SourceDataset,
    DeepfakeMethod
)


@dataclass
class IndexingResult:
    """Result of a batch indexing operation"""
    total_processed: int
    total_indexed: int
    failed_files: List[str]
    datasets_indexed: List[str]


class DatasetIndexer:
    """Indexes images from known deepfake datasets into the vector store"""
    
    # Common image extensions
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    def __init__(self):
        self._model = ImageDeepfakeModel()
        self._forensic_extractor = get_forensic_extractor()
        self._vector_store = get_vector_store()
    
    def index_directory(
        self,
        directory: str,
        label: str,
        dataset: SourceDataset = SourceDataset.CUSTOM,
        method: DeepfakeMethod = DeepfakeMethod.UNKNOWN,
        batch_size: int = 50,
        max_images: Optional[int] = None
    ) -> IndexingResult:
        """
        Index all images in a directory.
        
        Args:
            directory: Path to directory containing images
            label: Ground truth label ('real' or 'fake')
            dataset: Source dataset identifier
            method: Deepfake generation method (for fake images)
            batch_size: Number of images to process before committing
            max_images: Maximum number of images to index (None = all)
            
        Returns:
            IndexingResult with statistics
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        # Find all image files
        image_files = list(self._find_images(dir_path, max_images))
        
        total_processed = 0
        total_indexed = 0
        failed_files = []
        batch_records = []
        
        print(f"Found {len(image_files)} images in {directory}")
        print(f"Indexing as: label={label}, dataset={dataset.value}, method={method.value}")
        
        for image_path in tqdm(image_files, desc="Indexing"):
            total_processed += 1
            
            try:
                # Get embedding from model
                _, _, embedding = self._model.predict_with_features(str(image_path))
                
                if not embedding:
                    failed_files.append(str(image_path))
                    continue
                
                # Extract forensic features
                forensic_features = self._forensic_extractor.extract_all(str(image_path))
                
                # Create evidence record
                record = EvidenceRecord(
                    id=str(uuid.uuid4()),
                    embedding=embedding,
                    source_dataset=dataset,
                    label=label,
                    method=method,
                    forensic_features=forensic_features,
                    image_path=str(image_path)
                )
                
                batch_records.append(record)
                
                # Commit batch
                if len(batch_records) >= batch_size:
                    self._vector_store.add_evidence_batch(batch_records)
                    total_indexed += len(batch_records)
                    batch_records = []
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                failed_files.append(str(image_path))
        
        # Commit remaining records
        if batch_records:
            self._vector_store.add_evidence_batch(batch_records)
            total_indexed += len(batch_records)
        
        return IndexingResult(
            total_processed=total_processed,
            total_indexed=total_indexed,
            failed_files=failed_files,
            datasets_indexed=[dataset.value]
        )
    
    def index_faceforensics(
        self,
        root_dir: str,
        methods: Optional[List[str]] = None,
        max_per_method: int = 1000
    ) -> IndexingResult:
        """
        Index FaceForensics++ dataset with standard structure.
        
        Expected structure:
        root_dir/
          real/
          Deepfakes/
          Face2Face/
          FaceSwap/
          NeuralTextures/
        
        Args:
            root_dir: Root directory of FaceForensics++ dataset
            methods: List of methods to index (None = all)
            max_per_method: Maximum images per method
            
        Returns:
            IndexingResult with combined statistics
        """
        root = Path(root_dir)
        
        method_mapping = {
            "real": (DeepfakeMethod.REAL, "real"),
            "Deepfakes": (DeepfakeMethod.DEEPFAKES, "fake"),
            "Face2Face": (DeepfakeMethod.FACE2FACE, "fake"),
            "FaceSwap": (DeepfakeMethod.FACESWAP, "fake"),
            "NeuralTextures": (DeepfakeMethod.NEURALTEXTURES, "fake"),
        }
        
        if methods is None:
            methods = list(method_mapping.keys())
        
        total_result = IndexingResult(
            total_processed=0,
            total_indexed=0,
            failed_files=[],
            datasets_indexed=[]
        )
        
        for method_name in methods:
            method_dir = root / method_name
            if not method_dir.exists():
                print(f"Skipping {method_name}: directory not found")
                continue
            
            method_enum, label = method_mapping.get(
                method_name, 
                (DeepfakeMethod.UNKNOWN, "fake")
            )
            
            result = self.index_directory(
                directory=str(method_dir),
                label=label,
                dataset=SourceDataset.FACEFORENSICS,
                method=method_enum,
                max_images=max_per_method
            )
            
            total_result.total_processed += result.total_processed
            total_result.total_indexed += result.total_indexed
            total_result.failed_files.extend(result.failed_files)
        
        total_result.datasets_indexed = [SourceDataset.FACEFORENSICS.value]
        
        return total_result
    
    def _find_images(
        self, 
        directory: Path, 
        max_images: Optional[int] = None
    ) -> Generator[Path, None, None]:
        """Find all image files in directory recursively"""
        count = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in self.IMAGE_EXTENSIONS:
                    yield Path(root) / file
                    count += 1
                    if max_images and count >= max_images:
                        return


def get_dataset_indexer() -> DatasetIndexer:
    """Get a dataset indexer instance"""
    return DatasetIndexer()
