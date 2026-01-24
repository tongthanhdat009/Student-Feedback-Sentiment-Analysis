# ============================================================
# RUN ALL - Student Feedback Sentiment Analysis
# ============================================================
# Script này chạy tất cả các bước: preprocessing, training, dashboard
# ============================================================

Write-Host ""
Write-Host "############################################################" -ForegroundColor Cyan
Write-Host "# STUDENT FEEDBACK SENTIMENT ANALYSIS" -ForegroundColor Cyan
Write-Host "# Running Full Pipeline" -ForegroundColor Cyan  
Write-Host "############################################################" -ForegroundColor Cyan
Write-Host ""

# Lưu thư mục hiện tại
$projectDir = $PSScriptRoot
if (-not $projectDir) {
    $projectDir = Get-Location
}

Set-Location $projectDir
Write-Host "Working directory: $projectDir" -ForegroundColor Yellow
Write-Host ""

# ============================================================
# Step 1: Preprocessing
# ============================================================
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[1/5] PREPROCESSING" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
python preprocessing/build_dataset.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Preprocessing failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ============================================================
# Step 2: Train SVM
# ============================================================
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[2/5] TRAINING SVM" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
python training/train_svm.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: SVM training failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ============================================================
# Step 3: Train LSTM
# ============================================================
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[3/5] TRAINING LSTM" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
python training/train_lstm.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: LSTM training failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ============================================================
# Step 4: Train PhoBERT
# ============================================================
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[4/5] TRAINING PhoBERT" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
python training/train_phobert.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PhoBERT training failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ============================================================
# Step 5: Run Dashboard
# ============================================================
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[5/5] LAUNCHING DASHBOARD" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard starting at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

streamlit run dashboard/app.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
