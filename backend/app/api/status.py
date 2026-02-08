from fastapi import APIRouter, HTTPException
from app.services.job_store import get_job

router = APIRouter()

@router.get("/status/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job["job_id"],
        "status": job["status"]
    }
