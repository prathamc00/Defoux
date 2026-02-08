# Start Backend and Frontend in separate windows

Write-Host "Starting DeepGuard locally..."

# Start Backend
Start-Process -FilePath "cmd" -ArgumentList "/k cd backend && echo Installing dependencies... && pip install -r requirements.txt && echo Starting Backend... && uvicorn app.main:app --reload --port 8000" -WorkingDirectory $PSScriptRoot

# Start Frontend
Start-Process -FilePath "cmd" -ArgumentList "/k cd frontend && echo Installing dependencies... && npm install && echo Starting Frontend... && npm run dev -- --port 3000" -WorkingDirectory $PSScriptRoot

Write-Host "Services started! Check the new windows."
Write-Host "Backend: http://localhost:8000/docs"
Write-Host "Frontend: http://localhost:3000"
