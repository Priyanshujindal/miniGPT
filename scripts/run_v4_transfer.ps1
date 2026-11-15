$Host.UI.RawUI.WindowTitle = "MiniGPT V4-Transfer Training"
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "         MINIGPT V4-TRANSFER TRAINING (1,500 STEPS)       " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Working Directory: $PSScriptRoot\.."
Set-Location "$PSScriptRoot\.."
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -u training/train_v4_transfer.py --max-steps 1500
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         V4-TRANSFER TRAINING COMPLETE!                   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Read-Host "Press Enter to exit..."
