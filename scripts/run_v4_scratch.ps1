$Host.UI.RawUI.WindowTitle = "MiniGPT V4-Scratch Training"
Write-Host "==========================================================" -ForegroundColor Magenta
Write-Host "         MINIGPT V4-SCRATCH TRAINING (1,500 STEPS)        " -ForegroundColor Magenta
Write-Host "==========================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Working Directory: $PSScriptRoot\.."
Set-Location "$PSScriptRoot\.."
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -u training/train_v4_scratch.py --max-steps 1500
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         V4-SCRATCH TRAINING COMPLETE!                    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Read-Host "Press Enter to exit..."
