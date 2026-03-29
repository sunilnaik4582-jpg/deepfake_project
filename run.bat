@echo off
echo ----------------------------------------------------------
echo  Starting Deepfake Detection App...
echo  Open your browser at: http://127.0.0.1:8080/
echo ----------------------------------------------------------
cd /d "%~dp0"
python app.py
pause
