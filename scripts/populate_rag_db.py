"""
RAG Vector Database Population Script

This script downloads sample images from the 140k Real and Fake Faces dataset
and indexes them into the ChromaDB vector store for RAG-based detection.

Usage:
    cd backend
    python ../scripts/populate_rag_db.py

Requirements:
    - Kaggle API credentials configured (~/.kaggle/kaggle.json or environment variables)
    - Backend dependencies installed (pip install -r requirements.txt)
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

# Add backend to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))

# Configuration
DATASET_DIR = project_root / "ml" / "datasets" / "rag_samples"
MAX_IMAGES_PER_CLASS = 500  # Limit for faster indexing


def download_dataset():
    """Download the 140k Real and Fake Faces dataset from Kaggle"""
    print("\n📥 Downloading dataset from Kaggle...")
    print("=" * 50)
    
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle package not installed. Installing...")
        os.system(f"{sys.executable} -m pip install kaggle")
        import kaggle
    
    # Create dataset directory
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if (DATASET_DIR / "real").exists() and (DATASET_DIR / "fake").exists():
        real_count = len(list((DATASET_DIR / "real").glob("*.jpg")))
        fake_count = len(list((DATASET_DIR / "fake").glob("*.jpg")))
        if real_count > 0 and fake_count > 0:
            print(f"✅ Dataset already exists: {real_count} real, {fake_count} fake images")
            return True
    
    # Download from Kaggle
    zip_path = DATASET_DIR / "140k-real-and-fake-faces.zip"
    
    try:
        print("Downloading from: xhlulu/140k-real-and-fake-faces")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'xhlulu/140k-real-and-fake-faces',
            path=str(DATASET_DIR),
            unzip=False
        )
        print("✅ Download complete!")
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        print("\n📋 Manual setup instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
        print("2. Download and extract the dataset")
        print(f"3. Place images in: {DATASET_DIR}")
        print("   - {DATASET_DIR}/real/  <- real face images")
        print("   - {DATASET_DIR}/fake/  <- fake face images")
        return False
    
    # Extract zip
    if zip_path.exists():
        print("📦 Extracting dataset...")
        extract_dir = DATASET_DIR / "extracted"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(str(extract_dir))
        
        # Move images to expected structure
        # Dataset structure: real_vs_fake/real-vs-fake/train/real|fake
        source_base = extract_dir / "real_vs_fake" / "real-vs-fake" / "train"
        
        # Copy real images
        real_source = source_base / "real"
        real_dest = DATASET_DIR / "real"
        real_dest.mkdir(exist_ok=True)
        
        if real_source.exists():
            count = 0
            for img in real_source.glob("*.jpg"):
                if count >= MAX_IMAGES_PER_CLASS:
                    break
                shutil.copy(str(img), str(real_dest / img.name))
                count += 1
            print(f"   Copied {count} real images")
        
        # Copy fake images  
        fake_source = source_base / "fake"
        fake_dest = DATASET_DIR / "fake"
        fake_dest.mkdir(exist_ok=True)
        
        if fake_source.exists():
            count = 0
            for img in fake_source.glob("*.jpg"):
                if count >= MAX_IMAGES_PER_CLASS:
                    break
                shutil.copy(str(img), str(fake_dest / img.name))
                count += 1
            print(f"   Copied {count} fake images")
        
        # Cleanup
        shutil.rmtree(str(extract_dir))
        os.remove(str(zip_path))
        
        print("✅ Dataset extracted and organized!")
        return True
    
    return False


def index_dataset():
    """Index the dataset into the vector store"""
    print("\n🗄️  Indexing dataset into ChromaDB...")
    print("=" * 50)
    
    from app.services.rag.dataset_indexer import DatasetIndexer
    from app.services.rag.evidence_schema import SourceDataset, DeepfakeMethod
    from app.services.rag.vector_store import get_vector_store
    
    indexer = DatasetIndexer()
    vector_store = get_vector_store()
    
    # Check current count
    current_count = vector_store.get_record_count()
    print(f"Current records in database: {current_count}")
    
    if current_count > 0:
        response = input("Database already has records. Clear and re-index? (y/N): ")
        if response.lower() == 'y':
            vector_store.delete_all()
            print("Database cleared.")
        else:
            print("Keeping existing records. Adding new ones...")
    
    # Index real images
    real_dir = DATASET_DIR / "real"
    if real_dir.exists():
        print(f"\n📁 Indexing real images from: {real_dir}")
        result = indexer.index_directory(
            directory=str(real_dir),
            label="real",
            dataset=SourceDataset.REAL_VS_FAKE_140K,
            method=DeepfakeMethod.REAL,
            max_images=MAX_IMAGES_PER_CLASS
        )
        print(f"   ✅ Indexed {result.total_indexed}/{result.total_processed} real images")
        if result.failed_files:
            print(f"   ⚠️  Failed: {len(result.failed_files)} files")
    else:
        print(f"⚠️  Real images directory not found: {real_dir}")
    
    # Index fake images
    fake_dir = DATASET_DIR / "fake"
    if fake_dir.exists():
        print(f"\n📁 Indexing fake images from: {fake_dir}")
        result = indexer.index_directory(
            directory=str(fake_dir),
            label="fake",
            dataset=SourceDataset.REAL_VS_FAKE_140K,
            method=DeepfakeMethod.STYLEGAN,  # This dataset uses StyleGAN
            max_images=MAX_IMAGES_PER_CLASS
        )
        print(f"   ✅ Indexed {result.total_indexed}/{result.total_processed} fake images")
        if result.failed_files:
            print(f"   ⚠️  Failed: {len(result.failed_files)} files")
    else:
        print(f"⚠️  Fake images directory not found: {fake_dir}")
    
    # Final stats
    print("\n📊 Final Database Statistics:")
    print("-" * 30)
    stats = vector_store.get_statistics()
    print(f"   Total records: {stats['total_records']}")
    print(f"   Real samples:  {stats['real_count']}")
    print(f"   Fake samples:  {stats['fake_count']}")
    print(f"   Datasets:      {stats['datasets']}")
    print(f"   Methods:       {stats['methods']}")


def main():
    print("🛡️  DeepGuard RAG Database Setup")
    print("=" * 50)
    
    # Step 1: Download dataset
    if not download_dataset():
        print("\n⚠️  Dataset download failed. Please download manually.")
        print("You can still run the indexer after placing images in:")
        print(f"   - {DATASET_DIR}/real/")
        print(f"   - {DATASET_DIR}/fake/")
        return
    
    # Step 2: Index into vector store
    index_dataset()
    
    print("\n" + "=" * 50)
    print("✅ RAG database setup complete!")
    print("   The vector store is now populated and will be used")
    print("   for similarity matching during detection.")


if __name__ == "__main__":
    main()
