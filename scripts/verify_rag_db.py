
"""
RAG Vector Database Verification Script

This script verifies that the ChromaDB vector store is correctly populated
and accessible by performing a test similarity search.

Usage:
    cd scripts
    python verify_rag_db.py
"""

import sys
import os
import random
from pathlib import Path

# Add backend to path
# scripts/verify_rag_db.py -> backend/
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))

from app.services.rag.vector_store import get_vector_store
from app.models.image_model import ImageDeepfakeModel

def verify_database():
    print("\n🔍 Verifying RAG Vector Database...")
    print("=" * 50)
    
    # 1. Connect to Vector Store
    try:
        vector_store = get_vector_store()
        count = vector_store.get_record_count()
        print(f"✅ Connection successful!")
        print(f"📊 Total records in database: {count}")
        
        if count == 0:
            print("⚠️  Warning: Database is empty! Please run populate_rag_db.py first.")
            return
            
        # Get stats
        stats = vector_store.get_statistics()
        print(f"   - Real samples: {stats.get('real_count', 0)}")
        print(f"   - Fake samples: {stats.get('fake_count', 0)}")
        print(f"   - Datasets: {stats.get('datasets', [])}")
        
    except Exception as e:
        print(f"❌ Failed to connect to Vector Store: {e}")
        return

    # 2. Verify Embedding Dimension
    # We need to know what dimension the model produces to query correctly
    try:
        # Check if we can load the model to get dimension
        # (This is optional for DB check but good for integration check)
        model = ImageDeepfakeModel()
        dim = model.get_embedding_dim()
        print(f"✅ Model loaded. Embedding dimension: {dim}")
    except Exception as e:
        print(f"⚠️  Could not load model (using default dim=4): {e}")
        dim = 4
        
    # 3. Test Similarity Search
    print("\n🧪 Testing Similarity Search...")
    
    # Generate random query vector
    # In reality, this would be an embedding from a face image
    query_vector = [random.random() for _ in range(dim)]
    
    try:
        results = vector_store.search_similar(query_vector, k=3)
        print(f"✅ Search completed successfully! Found {len(results)} results.")
        
        print("\n📝 Top 3 Similar Cases (Random Query):")
        for i, case in enumerate(results):
            print(f"   {i+1}. ID: {case.record_id}")
            print(f"      Label: {case.label}")
            print(f"      Similarity: {case.similarity:.4f}")
            print(f"      Dataset: {case.source_dataset}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return

    print("\n✅ Verification passed! The RAG system is ready for use.")

if __name__ == "__main__":
    verify_database()
