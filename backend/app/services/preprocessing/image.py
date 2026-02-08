import torch
from PIL import Image
from facenet_pytorch import MTCNN

# Initialize MTCNN ONCE
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_mtcnn = MTCNN(
    image_size=224,
    margin=20,
    post_process=True,
    device=_DEVICE
)

def preprocess_image(image_path: str) -> str:
    """
    Phase 4A preprocessing:
    - Detect face using MTCNN
    - Crop & resize face
    - Fallback to original image if no face found
    """
    img = Image.open(image_path).convert("RGB")

    face = _mtcnn(img)

    if face is None:
        # No face detected — fallback
        img.save(image_path)
        return image_path

    # Convert tensor → PIL image
    face_img = Image.fromarray(
        (face.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    )

    face_img.save(image_path)
    return image_path
