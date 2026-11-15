$Host.UI.RawUI.WindowTitle = "MiniGPT V4 Conversational Chat"
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         MINIGPT V4 CONVERSATIONAL CHAT (TRANSFER)        " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Working Directory: $PSScriptRoot\.."
Set-Location "$PSScriptRoot\.."
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe generation/chat_v4.py --model-type transfer
