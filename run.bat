@echo off
cd /d "%~dp0"

echo ============================================================
echo Starting TruthLens AI - Fake News Detection Web App...
echo ============================================================

if exist "FakeNews\venv\Scripts\python.exe" goto IN_PARENT
if exist "venv\Scripts\python.exe" goto IN_PROJECT

echo [ERROR] Could not find venv\Scripts\python.exe
echo Please make sure you are in the project folder.
pause
exit /b 1

:IN_PARENT
cd FakeNews
goto RUN_APP

:IN_PROJECT
goto RUN_APP

:RUN_APP
echo Found virtual environment in %cd%\venv
echo Launching Streamlit...
".\venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats false
pause
