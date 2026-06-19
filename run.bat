@echo off
:: HemaPredict AI - One-Click Execution Pipeline Engine
title HemaPredict AI Activation Core
color 0B
cls

echo ===================================================================
echo               HEMAPREDICT AI ENGINE ACTIVATION CORE
echo ===================================================================
echo Framework Author: Rahma Mahmoud Elsayed Elmahdy
echo Architecture: Tabular Tree Ensembles (XGBoost, LightGBM, CatBoost)
echo Interoperability: Game-Theoretic SHAP Attributions
echo ===================================================================
echo.

:: Checking local system Python installation environment
echo [STEP 1/3] Verifying System Environment Requirements...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python environment was not detected on this system core.
    echo Please install Python 3.8+ and ensure 'Add to PATH' option is checked.
    pause
    exit
)
echo [SUCCESS] Python Environment Verification Completed.
echo.

:: Automatic dependency checks and pipeline installation
echo [STEP 2/3] Validating and Upgrading Core Mathematical Dependencies...
echo This may take a moment depending on internet speed boundaries...
pip install --quiet streamlit pandas numpy xgboost lightgbm catboost shap matplotlib scikit-learn
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Package installation failed inside local workspace. Check network interface.
    pause
    exit
)
echo [SUCCESS] High-Velocity Machine Learning Dependencies Sync Complete.
echo.

:: Localhost Native Cloud-Native Interface Launch
echo [STEP 3/3] Orchestrating Local Streamlit Server Deployment Core...
echo Core is deploying on localized volatile memory. Opening client...
echo.
streamlit run app.py

pause