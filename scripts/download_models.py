import os
import sys
import shutil
import json
import subprocess
from pathlib import Path

def download_models():
    # Configuration
    dataset_name = "prathamcu823/deepfake-models" # We'll assume this name
    project_root = Path(__file__).parent.parent
    checkpoints_dir = project_root / "ml" / "checkpoints"
    
    print(f"🔍 DeepGuard Model Downloader")
    print(f"==============================")
    
    # 1. Create checkpoints directory
    if not checkpoints_dir.exists():
        print(f"📁 Creating directory: {checkpoints_dir}")
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Configure Kaggle credentials from local kaggle.json
    kaggle_config = project_root / "kaggle.json"
    if kaggle_config.exists():
        print(f"🔑 Configuring Kaggle credentials from {kaggle_config}...")
        try:
            with open(kaggle_config, 'r') as f:
                creds = json.load(f)
            os.environ['KAGGLE_USERNAME'] = creds['username']
            os.environ['KAGGLE_KEY'] = creds['key']
        except Exception as e:
            print(f"⚠️ Warning: Could not read kaggle.json: {e}")
    else:
        print("❌ Error: kaggle.json not found in project root.")
        return

    # 3. Check for Kaggle API and Download
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("📦 Installing kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        from kaggle.api.kaggle_api_extended import KaggleApi

    print(f"📥 Downloading latest models from Kaggle ({dataset_name})...")
    try:
        api = KaggleApi()
        api.authenticate()
        
        # Download files to checkpoints dir
        api.dataset_download_files(dataset_name, path=str(checkpoints_dir), unzip=True)
        
        print(f"🏠 Files downloaded to: {checkpoints_dir}")
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        print("\n💡 Tip: Make sure you've run the UPLOAD script in your Colab notebook first!")
        return

    # 5. Verify files
    model_files = [
        'deepfake_detector_state_dict.pth',
        'deepfake_detector_complete.pth',
        'deepfake_detector_scripted.pt',
        'deepfake_detector.onnx'
    ]
    
    print(f"\n✅ Verification:")
    for f in model_files:
        path = checkpoints_dir / f
        if path.exists():
            size = path.stat().st_size / (1024*1024)
            print(f"   • {f} ({size:.2f} MB)")
        else:
            print(f"   • {f} (Missing)")

if __name__ == "__main__":
    download_models()
