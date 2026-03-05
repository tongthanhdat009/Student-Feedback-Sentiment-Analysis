# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vietnamese Student Feedback Sentiment Analysis using the **UIT-VSFC** dataset. The project implements multiple approaches for 3-class sentiment classification (Negative/Neutral/Positive):

- **PhoBERT Baseline**: Fine-tuned vinai/phobert-base transformer
- **TF-IDF Hybrid**: Combined PhoBERT embeddings + TF-IDF features
- **Improved Hybrid**: Enhanced version with SMOTE, XGBoost, Late Fusion

## Running Notebooks

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter (notebooks are in notebook/ directory)
jupyter notebook
```

**Note**: Notebooks are configured for Google Colab with paths like `/content/drive/MyDrive/Student-Feedback-Sentiment-Analysis`. When running locally, update the `BASE_DIR` in Config classes to use relative paths.

## Data Structure

```
data/
├── processed/
│   ├── train/
│   │   ├── sents.txt          # Vietnamese text samples
│   │   └── sentiments.txt     # Labels: 0=Negative, 1=Neutral, 2=Positive
│   ├── validation/
│   └── test/
└── raw/                        # Original UIT-VSFC data
```

**Dataset Statistics**:
- Train: 11,426 samples (Neutral class only 4% - heavily imbalanced)
- Validation: 1,583 samples
- Test: 3,166 samples

## Model Architecture

### PhoBERT Baseline ([notebook/PhoBERT_Baseline.ipynb](notebook/PhoBERT_Baseline.ipynb))
- Base: `vinai/phobert-base` (135M params)
- Classification head: Linear(768 → 3)
- Loss: CrossEntropyLoss
- Training: AdamW, lr=2e-5, warmup=10%, early stopping patience=5

### TF-IDF Hybrid ([notebook/TF-IDF_Model.ipynb](notebook/TF-IDF_Model.ipynb))
- **Feature extraction**:
  - PhoBERT frozen embeddings (768-dim from [CLS] token)
  - TF-IDF (5000 features, 1-2 grams)
  - Combined: 5768 features
- **Original**: LogisticRegression(C=1.0)
- **Improved versions**:
  - LR + SMOTE (C=0.1)
  - XGBoost + SMOTE (best: F1=0.932)
  - Late Fusion (separate PhoBERT/TF-IDF models, weighted predictions)

## Model Versioning

Models are organized hierarchically by model type and experiment type:

```
results/
├── PhoBERT/
│   ├── baseline/                   # Original PhoBERT (reference)
│   └── improvements/               # Enhanced versions with timestamp
├── PhoBERT_TF-IDF/
│   ├── baseline/                   # Original hybrid
│   └── improvements/               # SMOTE, XGBoost, etc.
├── PhoBERT_Sentiwordnet/
│   ├── baseline/
│   └── improvements/
└── PhoBERT_TF-IDF_Sentiwordnet/
    ├── baseline/
    └── improvements/
```

**Key**: Baseline folders have NO timestamp. Improvement folders use YYYYMMDD timestamps.

## File Naming Conventions

### Result Files (with timestamp)
When creating new result files, **always add a timestamp** to track iterations:

```
Models:         phobert_model_20260305.pt, hybrid_model_20260305.pkl
Summaries:      model_summary_20260305.csv
Visualizations: confusion_matrix_20260305.png, training_history_20260305.png
Artifacts:      tfidf_vectorizer_20260305.pkl, scaler_20260305.pkl
```

### Result Directories
Organize results by model type with baseline/improvements structure:

**Format**: `results/{ModelType}/{baseline|improvements}/{YYYYMMDD}/`

**Example**:
```
results/
├── PhoBERT/
│   ├── baseline/                   # Original baseline (no timestamp - keep as reference)
│   └── improvements/
│       ├── 20260305/              # Improved version with timestamp
│       └── 20260310/              # Another improvement
├── PhoBERT_TF-IDF/
│   ├── baseline/
│   └── improvements/
│       └── 20260305/
├── PhoBERT_Sentiwordnet/
│   ├── baseline/
│   └── improvements/
└── PhoBERT_TF-IDF_Sentiwordnet/
    ├── baseline/
    └── improvements/
```

**Key Rules**:
- **baseline/**: First/baseline version of each model type - NO timestamp
- **improvements/YYYYMMDD/**: All improved versions go here with timestamp
- Each experiment (baseline or improvement) follows the same subdirectory structure

### Result Files Naming Convention
**Inside each experiment directory (baseline or improvements), use organized subdirectories**:

```
results/PhoBERT_TF-IDF/improvements/20260305/
├── models/
│   ├── xgboost_smote_model.pkl     # Main model file
│   ├── lr_smote_model.pkl          # Secondary model
│   └── late_fusion_model.pkl       # Ensemble model
├── summaries/
│   ├── model_comparison.csv        # Comparison table
│   ├── summary.csv                 # Overall summary
│   ├── experiment_log.json         # Hyperparameters & config
│   └── training_results.txt        # Training metrics & logs (REQUIRED)
├── visualizations/
│   ├── confusion_matrix.png        # Standard charts
│   ├── training_history.png
│   ├── per_class_metrics.png
│   └── model_comparison_bar.png
└── artifacts/
    ├── tfidf_vectorizer.pkl        # Preprocessing artifacts
    ├── feature_scaler.pkl
    └── tokenizer.pkl
```

**File naming rules**:
- **Models**: `{algorithm}_model.pkl` (e.g., `xgboost_model.pkl`, `phobert_model.pt`)
- **Summaries**: `{description}.csv` (e.g., `model_comparison.csv`, `summary.csv`)
- **Training Logs**: `training_results.txt` (REQUIRED - see Training Logs section below)
- **Visualizations**: `{chart_type}.png` (e.g., `confusion_matrix.png`, `roc_curve.png`)
- **Artifacts**: `{component}.pkl` (e.g., `tfidf_vectorizer.pkl`, `scaler.pkl`)

**Special cases**:
- Multiple models: Add suffix like `_best`, `_v1`, `_v2`
- Multiple charts: Add qualifier like `_train`, `_val`, `_test`
- Multiple experiments: Add descriptive suffix like `_smote`, `_focal_loss`

### Why This Structure?
- **Timestamp in directory**: Tracks when experiment ran
- **Subdirectories**: Organized by file type
- **Descriptive names**: Self-documenting, no ambiguity
- **Easy comparison**: Side-by-side directories for different runs

## Training Logs (REQUIRED)

**Every training experiment MUST save results to `training_results.txt`** in the `summaries/` directory.

### Content Format

```
========================================
TRAINING RESULTS - {Model Name}
========================================
Date: {YYYY-MM-DD HH:MM:SS}
Model Type: {PhoBERT / PhoBERT_TF-IDF / PhoBERT_Sentiwordnet / ...}
Experiment: {baseline / improvements/}

----------------------------------------
HYPERPARAMETERS
----------------------------------------
Learning Rate: {value}
Batch Size: {value}
Epochs: {value}
Optimizer: {AdamW / Adam / ...}
Loss Function: {CrossEntropy / ...}
{other relevant hyperparameters...}

----------------------------------------
TRAINING RESULTS
----------------------------------------
Train Loss: {value}
Train Accuracy: {value}
Train F1 (macro): {value}

Validation Loss: {value}
Validation Accuracy: {value}
Validation F1 (macro): {value}

----------------------------------------
TEST RESULTS
----------------------------------------
Test Accuracy: {value}
Test F1 (macro): {value}
Test Precision (macro): {value}
Test Recall (macro): {value}

Per-Class Metrics:
  Negative: Precision={value}, Recall={value}, F1={value}
  Neutral:  Precision={value}, Recall={value}, F1={value}
  Positive: Precision={value}, Recall={value}, F1={value}

----------------------------------------
CONFUSION MATRIX
----------------------------------------
[[tn, fn, fp],
 [fn, tn, fp],
 [fp, fn, tp]]

----------------------------------------
TRAINING TIME
----------------------------------------
Total Time: {X minutes Y seconds}
Epochs Completed: {N}
Best Epoch: {N}
```

### Why Training Logs Are Required
- **Reproducibility**: Track exact hyperparameters used
- **Comparison**: Easy to compare different experiments
- **Debugging**: Identify issues by reviewing training history
- **Documentation**: Permanent record without needing to re-run

## Vietnamese Text Preprocessing

```python
def preprocess_vietnamese(text):
    text = text.lower()
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    return ' '.join(text.split())
```

## Common Issues

**Class Imbalance**: Neutral class is only 4% of training data. Solutions:
- SMOTE oversampling (Neutral: 458 → 3000)
- Class-weighted loss
- Focal Loss (see [plan/Baseline_Fine-tune.md](plan/Baseline_Fine-tune.md))

**Memory**: PhoBERT extraction is slow. Extract embeddings once and cache them.

**Model Loading**: Use `load_model_safe()` helper to handle state_dict with/without "module." prefix from DataParallel.

## Results Tracking

Current best model (TF-IDF Hybrid + XGBoost + SMOTE):
- Test Accuracy: 0.933
- Test F1: 0.932
- Best for: Neutral class improvement (main challenge)

## Visualization

[notebook/Model_Visualization.ipynb](notebook/Model_Visualization.ipynb) generates:
- Per-class Precision/Recall/F1 comparisons
- Confusion matrices (normalized and count)
- Confidence distributions
- Model comparison charts
