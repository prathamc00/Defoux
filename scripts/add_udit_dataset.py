"""
Quick script to download and index the Real vs Fake Faces dataset by Udit Sharma
This adds Photoshop-based fakes to complement the StyleGAN dataset.
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

DATASET_DIR = project_root / "ml" / "datasets" / "rag_udit"
MAX_IMAGES_PER_CLASS = 500

def main():
    print("🛡️  Adding Udit Sharma's Real vs Fake Faces Dataset")
    print("=" * 50)
    
    # Download
    print("\n📥 Downloading from Kaggle...")
    try:
        import kaggle
        kaggle.api.authenticate()
        
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        
        kaggle.api.dataset_download_files(
            'uditsharma72/real-vs-fake-faces',
            path=str(DATASET_DIR),
            unzip=True
        )
        print("✅ Download complete!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return
    
    # Find the real/fake folders
    print("\n📁 Locating image folders...")
    real_dir = None
    fake_dir = None
    
    for root, dirs, files in os.walk(DATASET_DIR):
        root_lower = root.lower()
        if 'real' in root_lower and 'fake' not in root_lower:
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                real_dir = Path(root)
                print(f"   Found real: {real_dir}")
        elif 'fake' in root_lower:
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                fake_dir = Path(root)
                print(f"   Found fake: {fake_dir}")
    
    if not real_dir or not fake_dir:
        print("❌ Could not find real/fake directories")
        # List structure for debugging
        print("Directory structure:")
        for root, dirs, files in os.walk(DATASET_DIR):
            level = root.replace(str(DATASET_DIR), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            if level < 2:
                for f in files[:3]:
                    print(f'{indent}  {f}')
                if len(files) > 3:
                    print(f'{indent}  ... and {len(files)-3} more files')
        return
    
    # Index into vector store
    print("\n🗄️  Indexing into ChromaDB...")
    
    from app.services.rag.dataset_indexer import DatasetIndexer
    from app.services.rag.evidence_schema import SourceDataset, DeepfakeMethod
    from app.services.rag.vector_store import get_vector_store
    
    indexer = DatasetIndexer()
    vector_store = get_vector_store()
    
    print(f"Current records in database: {vector_store.get_record_count()}")
    
    # Index real images
    print(f"\n📁 Indexing real images from: {real_dir}")
    result = indexer.index_directory(
        directory=str(real_dir),
        label="real",
        dataset=SourceDataset.REAL_VS_FAKE_UDIT,
        method=DeepfakeMethod.REAL,
        max_images=MAX_IMAGES_PER_CLASS
    )
    print(f"   ✅ Indexed {result.total_indexed}/{result.total_processed} real images")
    
    # Index fake images (Photoshop-based)
    print(f"\n📁 Indexing fake images from: {fake_dir}")
    result = indexer.index_directory(
        directory=str(fake_dir),
        label="fake",
        dataset=SourceDataset.REAL_VS_FAKE_UDIT,
        method=DeepfakeMethod.PHOTOSHOP,
        max_images=MAX_IMAGES_PER_CLASS
    )
    print(f"   ✅ Indexed {result.total_indexed}/{result.total_processed} fake images")
    
    # Final stats
    print("\n📊 Final Database Statistics:")
    print("-" * 30)
    stats = vector_store.get_statistics()
    print(f"   Total records: {stats['total_records']}")
    print(f"   Real samples:  {stats['real_count']}")
    print(f"   Fake samples:  {stats['fake_count']}")
    print(f"   Datasets:      {stats['datasets']}")
    print(f"   Methods:       {stats['methods']}")
    
    print("\n" + "=" * 50)
    print("✅ Dataset added successfully!")

if __name__ == "__main__":
    main()
