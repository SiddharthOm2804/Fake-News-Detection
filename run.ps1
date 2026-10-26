$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

if (Test-Path "$ScriptDir\venv\Scripts\python.exe") {
    Set-Location $ScriptDir
} elseif (Test-Path "$ScriptDir\FakeNews\venv\Scripts\python.exe") {
    Set-Location "$ScriptDir\FakeNews"
} else {
    Write-Error "Could not find virtual environment 'venv\Scripts\python.exe'."
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting TruthLens AI - Fake News Detection Web App..." -ForegroundColor Cyan
Write-Host "Working Directory: $(Get-Location)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

& ".\venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats false
