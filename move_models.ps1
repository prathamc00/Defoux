# PowerShell script to move trained models to checkpoints directory
# Run this script from the DeepGuard project root directory

Write-Host "Looking for model files in Downloads folder..." -ForegroundColor Cyan
Write-Host ""

# Define source (Downloads) and destination (checkpoints)
$downloadsPath = "$env:USERPROFILE\Downloads"
$checkpointsPath = ".\ml\checkpoints"

# Model files to move
$modelFiles = @(
    "deepfake_detector_state_dict.pth",
    "deepfake_detector_complete.pth",
    "deepfake_detector_scripted.pt",
    "deepfake_detector.onnx"
)

# Create checkpoints directory if it doesn't exist
if (-not (Test-Path $checkpointsPath)) {
    Write-Host "Creating checkpoints directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $checkpointsPath -Force | Out-Null
}

$movedCount = 0
$notFoundCount = 0

foreach ($file in $modelFiles) {
    $sourcePath = Join-Path $downloadsPath $file
    $destPath = Join-Path $checkpointsPath $file
    
    if (Test-Path $sourcePath) {
        $fileSize = (Get-Item $sourcePath).Length / 1MB
        Write-Host "[FOUND] $file ($("{0:N2}" -f $fileSize) MB)" -ForegroundColor Green
        
        try {
            Copy-Item -Path $sourcePath -Destination $destPath -Force
            Write-Host "        Copied to: $checkpointsPath" -ForegroundColor Green
            $movedCount++
        }
        catch {
            Write-Host "        Error moving file: $_" -ForegroundColor Red
        }
        Write-Host ""
    }
    else {
        Write-Host "[NOT FOUND] $file" -ForegroundColor Red
        Write-Host "            Looking in: $sourcePath" -ForegroundColor Gray
        Write-Host ""
        $notFoundCount++
    }
}

Write-Host ("=" * 60) -ForegroundColor Gray
Write-Host "SUMMARY:" -ForegroundColor Cyan
Write-Host "  Moved: $movedCount files" -ForegroundColor Green
Write-Host "  Not found: $notFoundCount files" -ForegroundColor $(if ($notFoundCount -gt 0) { "Yellow" } else { "Green" })
Write-Host ""

if ($movedCount -gt 0) {
    Write-Host "Model files are now in:" -ForegroundColor Green
    Write-Host "  $((Resolve-Path $checkpointsPath).Path)" -ForegroundColor White
    Write-Host ""
    Write-Host "Files in checkpoints directory:" -ForegroundColor Cyan
    Get-ChildItem $checkpointsPath -Filter "*.p*" | ForEach-Object {
        $size = $_.Length / 1MB
        Write-Host "  - $($_.Name) ($("{0:N2}" -f $size) MB)" -ForegroundColor White
    }
}

if ($notFoundCount -gt 0) {
    Write-Host ""
    Write-Host "WARNING: Some files were not found in Downloads folder" -ForegroundColor Yellow
    Write-Host "         Please manually move them to: $checkpointsPath" -ForegroundColor Gray
}
