@echo off
REM ============================================================
REM RUN ALL - Student Feedback Sentiment Analysis
REM ============================================================
REM Script này chạy tất cả các bước: preprocessing, training, dashboard
REM ============================================================

echo.
echo ############################################################
echo # STUDENT FEEDBACK SENTIMENT ANALYSIS
echo # Running Full Pipeline
echo ############################################################
echo.

cd /d "%~dp0"
echo Working directory: %CD%
echo.

REM ============================================================
REM Step 1: Preprocessing
REM ============================================================
echo ============================================================
echo [1/5] PREPROCESSING
echo ============================================================
python preprocessing/build_dataset.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Preprocessing failed!
    pause
    exit /b 1
)
echo.

REM ============================================================
REM Step 2: Train SVM
REM ============================================================
echo ============================================================
echo [2/5] TRAINING SVM
echo ============================================================
python training/train_svm.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: SVM training failed!
    pause
    exit /b 1
)
echo.

REM ============================================================
REM Step 3: Train LSTM
REM ============================================================
echo ============================================================
echo [3/5] TRAINING LSTM
echo ============================================================
python training/train_lstm.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: LSTM training failed!
    pause
    exit /b 1
)
echo.

REM ============================================================
REM Step 4: Train PhoBERT
REM ============================================================
echo ============================================================
echo [4/5] TRAINING PhoBERT
echo ============================================================
python training/train_phobert.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PhoBERT training failed!
    pause
    exit /b 1
)
echo.

REM ============================================================
REM Step 5: Visualization & Summary Report
REM ============================================================
echo ============================================================
echo [5/5] VISUALIZATION ^& SUMMARY REPORT
echo ============================================================
echo.
echo Generating summary report...
python -c "from utils.report_generator import generate_summary_report; generate_summary_report()"
echo.
echo All results saved to:
echo   - Reports: results/*.txt
echo   - Models:  saved_models/*/
echo.

echo.
echo ============================================================
echo COMPLETE!
echo ============================================================
echo.
echo To view charts on Google Colab, use:
echo   from visualization.colab_visualize import show_all_results
echo   show_all_results()
echo.
pause
