import uuid
from datetime import datetime

# DEV-ONLY in-memory store
_JOBS = {}

def create_job(filename: str, media_type: str):
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "job_id": job_id,
        "filename": filename,
        "media_type": media_type,
        "status": "uploaded",
        "result": None,
        "confidence": None,
        "created_at": datetime.utcnow().isoformat()
    }
    return job_id

def update_job(job_id: str, **kwargs):
    if job_id in _JOBS:
        _JOBS[job_id].update(kwargs)

def get_job(job_id: str):
    return _JOBS.get(job_id)
