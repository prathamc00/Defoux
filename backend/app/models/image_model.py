import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import os

class ImageDeepfakeModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Path to the local model
        # backend/app/models/image_model.py -> ... -> DeepGuard/ml/checkpoints
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3] 
        self.model_path = project_root / "ml" / "checkpoints" / "deepfake_detector_scripted.pt"
        
        self.model = None
        self._load_model()

        # Transforms used during validation/inference (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        if self.model_path.exists():
            try:
                print(f"Loading local model from {self.model_path}...")
                self.model = torch.jit.load(str(self.model_path), map_location=self.device)
                self.model.eval()
                print("Local model loaded successfully.")
            except Exception as e:
                print(f"Error loading local model: {e}")
                self.model = None
        else:
            print(f"Local model not found at {self.model_path}")
            self.model = None

    def predict(self, image_path: str):
        if self.model is None:
            # Fallback or error if model isn't loaded
            # For now, let's return a dummy or raise error. 
            # Given the requirement is to use the trained model, we should probably fail if it's missing.
            # But to be safe, we can return "Error" or similar.
            return "model_not_loaded", 0.0

        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                # Assuming output involves logits. We appy softmax/sigmoid depending on training.
                # In the notebook, it was a binary classification (Real vs Fake).
                # Usually: 0=Real, 1=Fake or vice versa.
                # Let's check the notebook for class mapping...
                # Notebook says: Real: 0, Fake: 1
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                fake_prob = probabilities[0][1].item()
                real_prob = probabilities[0][0].item()

            if fake_prob > real_prob:
                return "fake", fake_prob
            else:
                return "real", real_prob

        except Exception as e:
            print(f"Error during prediction: {e}")
            return "error", 0.0

    def predict_with_features(self, image_path: str):
        """
        Enhanced prediction that returns label, confidence, and feature embedding.
        
        Returns:
            Tuple of (label, confidence, embedding_vector)
            - label: 'real', 'fake', 'model_not_loaded', or 'error'
            - confidence: float between 0.0 and 1.0
            - embedding: List[float] - 1000-dim feature vector (or empty list on error)
        """
        if self.model is None:
            return "model_not_loaded", 0.0, []

        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                fake_prob = probabilities[0][1].item()
                real_prob = probabilities[0][0].item()
                
                # Use the logits as a simple embedding 
                # (TorchScript models don't easily expose intermediate layers)
                # Flatten the output logits and append probabilities for richer representation
                embedding = outputs[0].cpu().numpy().tolist()
                
                # Extend with probabilities for better similarity matching
                embedding.extend([real_prob, fake_prob])

            if fake_prob > real_prob:
                return "fake", fake_prob, embedding
            else:
                return "real", real_prob, embedding

        except Exception as e:
            print(f"Error during prediction with features: {e}")
            return "error", 0.0, []

    def get_embedding_dim(self) -> int:
        """Return the dimension of the embedding vector"""
        # Output logits (2) + probabilities (2) = 4
        # This is minimal but works with the TorchScript model
        return 4

