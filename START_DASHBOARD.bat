@echo off
title Shopper Spectrum — Launching...
color 0A

echo.
echo  ============================================
echo    SHOPPER SPECTRUM - Customer Intelligence
echo  ============================================
echo.
echo  Starting dashboard, please wait...
echo.

cd /d "%~dp0"
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

pause
