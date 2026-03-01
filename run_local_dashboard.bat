@echo off
:: ===================================================
:: LOCAL DASHBOARD ENVIRONMENT
:: ===================================================
:: Purpose: Data Analysis, Dashboard Rendering & Validation

echo ===================================================
echo AquaPredict: Starting Local Analytical Dashboard
echo ===================================================
echo.

echo HANG TIGHT! Starting Streamlit Web Interface...
echo (Note: The web will read the Results files you downloaded from the remote server)

python -m streamlit run dashboard/app.py
pause
