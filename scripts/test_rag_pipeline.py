
"""
RAG Pipeline Verification Script

This script simulates the full RAG-enhanced detection pipeline:
1. Preprocessing (face detection)
2. Vision Model Prediction
3. Forensic Feature Extraction
4. RAG Evidence Retrieval
5. Evidence Fusion
6. Explanation Generation

Usage:
    cd scripts
    python test_rag_pipeline.py
"""

import sys
import os
import shutil
from pathlib import Path

# Add backend to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))

from app.models.image_model import ImageDeepfakeModel
from app.services.forensics.forensic_extractor import get_forensic_extractor
from app.services.rag.vector_store import get_vector_store
from app.services.rag.fusion_engine import get_fusion_engine
from app.services.explainability.explanation_generator import get_explanation_generator

# Test images (adjust paths as needed)
REAL_IMAGE_PATH = project_root / "ml" / "datasets" / "rag_udit" / "real" / "real_00001.jpg"
FAKE_IMAGE_PATH = project_root / "ml" / "datasets" / "rag_udit" / "fake" / "easy_100_1111.jpg"

def run_pipeline(image_path, label_type):
    print(f"\nTesting Pipeline on {label_type.upper()} Image...")
    print(f"   Path: {image_path}")
    print("-" * 50)
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    try:
        # 1. Initialize Components
        print("1. Initializing components...")
        model = ImageDeepfakeModel()
        forensic_extractor = get_forensic_extractor()
        vector_store = get_vector_store()
        fusion_engine = get_fusion_engine()
        explanation_generator = get_explanation_generator()
        
        # 2. Vision Model Prediction
        print("2. Running Vision Model...")
        # Note: In a real app, we'd run preprocessing first. 
        # For this test, we assume the image is already a face crop or can be processed as is.
        # But `predict_with_features` expects a path.
        label, confidence, embedding = model.predict_with_features(str(image_path))
        print(f"   Vision Output: {label} ({confidence:.4f})")
        
        if label in ["model_not_loaded", "error"]:
            print("Vision model failed.")
            return

        # 3. Forensic Analysis
        print("3. Extracting Forensic Features...")
        forensic_features = forensic_extractor.extract_all(str(image_path))
        print(f"   FFT High Freq: {forensic_features.fft_high_freq_ratio:.4f}")
        print(f"   Noise Var: {forensic_features.noise_variance:.4f}")
        
        # 4. RAG Retrieval
        print("4. Retrieving Similar Cases...")
        similar_cases = []
        if embedding and vector_store.get_record_count() > 0:
            similar_cases = vector_store.search_similar(embedding, k=3)
            print(f"   found {len(similar_cases)} cases")
            for i, case in enumerate(similar_cases):
                 print(f"   - {i+1}: {case.label} ({case.similarity:.4f}) [{case.source_dataset}]")
        else:
            print("   (Skipped: No embedding or empty DB)")

        # 5. Fusion
        print("5. Fusing Evidence...")
        fused_prediction = fusion_engine.fuse(
            vision_label=label,
            vision_confidence=confidence,
            similar_cases=similar_cases,
            forensic_signals=forensic_features
        )
        print(f"   Result: {fused_prediction.result}")
        print(f"   Confidence: {fused_prediction.confidence:.4f}")
        print(f"   RAG Adjustment: {fused_prediction.rag_adjustment:.4f}")
        
        # 6. Explanation
        print("6. Generating Explanation...")
        report = explanation_generator.generate(fused_prediction)
        print(f"   Summary: {report.summary}")
        print(f"   Reasoning: {report.confidence_reasoning}")
        
        print("\nPipeline Test Passed!")

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Starting RAG Pipeline Verification")
    print("=" * 50)
    
    run_pipeline(REAL_IMAGE_PATH, "real")
    run_pipeline(FAKE_IMAGE_PATH, "fake")

if __name__ == "__main__":
    main()
