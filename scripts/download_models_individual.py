
import os
import sys
import json
import subprocess
from pathlib import Path

def download_models_individually():
    dataset_name = "prathamcu823/deepfake-models"
    project_root = Path(__file__).parent.parent
    checkpoints_dir = project_root / "ml" / "checkpoints"
    
    print(f"🔍 DeepGuard Individual Model Downloader")
    
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure Kaggle
    kaggle_config = project_root / "kaggle.json"
    if kaggle_config.exists():
        try:
            with open(kaggle_config, 'r') as f:
                creds = json.load(f)
            os.environ['KAGGLE_USERNAME'] = creds['username']
            os.environ['KAGGLE_KEY'] = creds['key']
        except Exception as e:
            print(f"⚠️ Warning: Could not read kaggle.json: {e}")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    
    model_files = [
        'deepfake_detector_state_dict.pth',
        'deepfake_detector_complete.pth',
        'deepfake_detector_scripted.pt',
        'deepfake_detector.onnx'
    ]
    
    print(f"📥 Downloading files individually from {dataset_name}...")
    
    for file_name in model_files:
        print(f"   • Downloading {file_name}...")
        try:
            api.dataset_download_file(dataset_name, file_name, path=str(checkpoints_dir))
            print(f"     ✅ Success")
        except Exception as e:
            print(f"     ❌ Failed: {e}")

    print(f"\n🏠 Files in {checkpoints_dir}:")
    for f in checkpoints_dir.iterdir():
        print(f" - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    download_models_individually()
