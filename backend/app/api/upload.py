from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from app.services.job_store import create_job, update_job
from app.services.inference.image_detector import run_image_detection

router = APIRouter()

UPLOAD_DIR = Path("app/storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.content_type.startswith("image"):
        raise HTTPException(
            status_code=400,
            detail="Only image uploads supported in Phase 3"
        )

    job_id = create_job(
        filename=file.filename,
        media_type="image"
    )

    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔥 RUN MODEL (PHASE 3 = synchronous)
    update_job(job_id, status="processing")

    output = run_image_detection(str(file_path))

    update_job(
        job_id,
        status="completed",
        result=output["result"],
        confidence=output["confidence"]
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "result": output["result"],
        "confidence": output["confidence"]
    }
