
import zipfile
import os
from pathlib import Path

def unzip_models():
    zip_path = Path("ml/checkpoints/deepfake-models.zip")
    extract_to = Path("ml/checkpoints")
    
    if not zip_path.exists():
        print(f"❌ Zip file not found: {zip_path}")
        return

    print(f"📦 Unzipping {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.testzip() # Check for corruption
            print("✅ Zip file integrity verified.")
            zip_ref.extractall(extract_to)
            print(f"✅ Extracted to {extract_to}")
            print("\nFiles extracted:")
            for f in zip_ref.namelist():
                print(f" - {f}")
    except zipfile.BadZipFile:
        print("❌ Error: Bad Zip File")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    unzip_models()
