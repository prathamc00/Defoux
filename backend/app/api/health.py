from fastapi import APIRouter
from app.config import get_settings

router = APIRouter()

@router.get("/health")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "env": settings.ENV
    }
