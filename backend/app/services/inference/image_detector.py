from app.models.image_model import ImageDeepfakeModel
from app.services.preprocessing.image import preprocess_image

_model = ImageDeepfakeModel()

def run_image_detection(image_path: str):
    processed_path = preprocess_image(image_path)

    label, confidence = _model.predict(processed_path)

    return {
        "result": label,
        "confidence": round(confidence, 4)
    }
