# 📊 CẤU TRÚC CODE & PHÂN TÍCH CHI TIẾT

## 1. 🏗️ CẤU TRÚC TỔNG THỂ CỦA CODE

```
Student_Feedback_Sentiment_Analysis.ipynb
│
├── PHẦN 1: SETUP & CONFIGURATION
│   ├── Cell 1: Install Required Libraries
│   ├── Cell 2: Import Libraries & Check GPU
│   └── Cell 3: Configuration & Caching Utilities
│
├── PHẦN 2: DATA COLLECTION & EXPLORATION
│   ├── Cell 4: Download Dataset from Huggingface
│   ├── Cell 5: Convert to Pandas DataFrame
│   ├── Cell 6-9: Data Visualization (4 cells)
│   └── Cell 10: Basic Statistics
│
├── PHẦN 3: DATA PREPROCESSING
│   ├── Cell 11: Preprocessing Functions Definition
│   ├── Cell 12: Apply Preprocessing to Dataset
│   └── Cell 13: Prepare Train/Test Data
│
├── PHẦN 4: FEATURE ENGINEERING
│   └── Cell 14: TF-IDF Vectorization
│
├── PHẦN 5: MODEL TRAINING
│   ├── Cell 15: Train SVM Model ✅ (COMPLETED)
│   ├── Cell 16-18: LSTM Model Setup
│   ├── Cell 19: Train LSTM Model ❌ (INTERRUPTED)
│   ├── Cell 20-22: PhoBERT Model Setup
│   └── Cell 23: Train PhoBERT Model ⏳ (NOT STARTED)
│
├── PHẦN 6: MODEL EVALUATION
│   ├── Cell 24: Compile All Results
│   ├── Cell 25: Classification Reports
│   └── Cell 26-30: Visualization (5 cells)
│
├── PHẦN 7: SAVE & INFERENCE
│   ├── Cell 31-32: Save Results & Models
│   └── Cell 33-35: Inference Functions
│
└── PHẦN 8: SUMMARY
    └── Cell 36: Final Summary
```

---

## 2. 🤖 CẤU TRÚC XỬ LÝ CỦA TỪNG MODEL

### 2.1 SVM MODEL (Support Vector Machine) ✅

```
CẤU TRÚC XỬ LÝ:
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Raw Vietnamese Text                                 │
│  "Giảng viên dạy rất hay và tận tình"                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: TEXT PREPROCESSING                                │
│  ────────────────────────────────                           │
│  • Lowercase conversion                                     │
│  • Remove special characters                                │
│  • Keep Vietnamese characters (à, á, ạ, ả, ã, đ...)        │
│  • Remove extra whitespace                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: VIETNAMESE TOKENIZATION (Underthesea)             │
│  ───────────────────────────────────────────────            │
│  Input:  "giảng viên dạy rất hay"                          │
│  Output: "giảng_viên dạy rất hay"                          │
│  (Add "_" between multi-word tokens)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: TF-IDF VECTORIZATION                              │
│  ────────────────────────────────                           │
│  • Create vocabulary (323 words)                            │
│  • Generate n-grams (1-gram, 2-grams)                      │
│  • Calculate TF-IDF scores                                  │
│  • Output: Sparse matrix (700, 323)                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: SVM TRAINING                                      │
│  ──────────────────────                                     │
│  Model: SVC(kernel='rbf', random_state=42)                 │
│  Training:                                                  │
│    • Fit on training data (700 samples)                     │
│    • Validate on validation set (150 samples)              │
│    • Test on test set (150 samples)                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: PREDICTION                                         │
│  ─────────────────────                                       │
│  • Label: 0 (Negative), 1 (Neutral), 2 (Positive)          │
│  Accuracy: 100% on Test Set                                 │
│  F1 Score: 100% on Test Set                                 │
└─────────────────────────────────────────────────────────────┘

CONFIGURATION:
├── Kernel: RBF (Radial Basis Function)
├── Max Features: 10,000
├── N-gram Range: (1, 2)
├── Min DF: 2
└── Max DF: 0.95
```

### 2.2 LSTM MODEL (Long Short-Term Memory) ❌

```
CẤU TRÚC XỬ LÝ:
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Preprocessed & Tokenized Text                       │
│  "giảng_viên dạy rất hay"                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: BUILD VOCABULARY                                   │
│  ────────────────────────                                    │
│  • Count word frequency from training data                  │
│  • Create vocabulary (126 words with min_freq=2)           │
│  • Special tokens: <PAD>=0, <UNK>=1                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: TEXT TO INDICES                                    │
│  ──────────────────────                                      │
│  "giảng_viên dạy rất hay" → [45, 12, 89, 67]               │
│  • Truncate/pad to max_length=100                           │
│  • Replace unknown words with <UNK>                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: CREATE DATALOADERS                                │
│  ────────────────────────────                               │
│  • SimpleTextDataset wrapper                                │
│  • Batch size: 32                                          │
│  • Shuffle training data                                    │
│  • Train: 22 batches | Val: 5 batches | Test: 5 batches   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: LSTM MODEL ARCHITECTURE                            │
│  ───────────────────────────────────────                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Embedding Layer (vocab_size=126, dim=128)           │   │
│  │         ▼                                          │   │
│  │ Bi-LSTM (hidden_dim=256, layers=2, dropout=0.3)     │   │
│  │         ▼                                          │   │
│  │ Concatenate Forward + Backward Hidden States       │   │
│  │         ▼                                          │   │
│  │ FC Layers: 512 → 128 → 3 (output classes)          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Total Parameters: 2,449,667                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: TRAINING LOOP (10 Epochs)                         │
│  ──────────────────────────────                              │
│  For each epoch:                                            │
│    • Forward pass through LSTM                              │
│    • Calculate CrossEntropyLoss                             │
│    • Backward pass with gradient clipping (max_norm=1.0)    │
│    • Adam optimizer (lr=0.001)                              │
│    • ReduceLROnPlateau scheduler                            │
│    • Save best model based on validation accuracy           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: PREDICTION                                         │
│  ─────────────────────                                       │
│  ⚠️ TRAINING INTERRUPTED AT EPOCH 1/10                     │
│  Epoch 1 Results:                                           │
│    • Train Acc: 69.14%                                      │
│    • Val Acc: 97.33%                                        │
└─────────────────────────────────────────────────────────────┘

CONFIGURATION:
├── Vocabulary Size: 126
├── Embedding Dim: 128
├── Hidden Dim: 256
├── Num Layers: 2
├── Dropout: 0.3
├── Bidirectional: True
├── Learning Rate: 0.001
├── Batch Size: 32
└── Max Length: 100
```

### 2.3 PhoBERT MODEL (Vietnamese BERT) ⏳

```
CẤU TRÚC XỬ LÝ:
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Preprocessed Text                                   │
│  "giảng viên dạy rất hay"                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: PhoBERT TOKENIZATION                              │
│  ────────────────────────────────                            │
│  • Use AutoTokenizer from vinai/phobert-base               │
│  • Add special tokens: <s>, </s>                           │
│  • Subword tokenization                                     │
│  • Example: "giảng viên" → "giảng", "_", "viên"            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: ENCODING                                          │
│  ────────────────                                            │
│  • input_ids: Token IDs                                    │
│  • attention_mask: Mask for padding tokens                 │
│  • Max length: 256                                         │
│  • Padding & truncation                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: CREATE DATALOADERS                                │
│  ────────────────────────────                               │
│  • PhoBERTDataset wrapper                                  │
│  • Batch size: 16                                          │
│  • Train: 44 batches | Val: 10 batches | Test: 10 batches │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: PhoBERT MODEL ARCHITECTURE                         │
│  ───────────────────────────────────────                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PhoBERT Base (vinai/phobert-base)                   │   │
│  │ • 12 Transformer Layers                             │   │
│  │ • Hidden Size: 768                                  │   │
│  │ • Attention Heads: 12                               │   │
│  │ • Parameters: ~85M                                  │   │
│  │         ▼                                          │   │
│  │ Classification Head: 768 → 3 (output classes)      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Total Parameters: ~85,000,000                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: TRAINING LOOP (5 Epochs)                          │
│  ──────────────────────────────                              │
│  For each epoch:                                            │
│    • Forward pass through PhoBERT                           │
│    • Calculate loss (CrossEntropy)                          │
│    • Mixed precision training (if GPU)                      │
│    • Gradient clipping (max_norm=1.0)                       │
│    • AdamW optimizer (lr=2e-5, weight_decay=0.01)          │
│    • LinearLR scheduler                                    │
│    • Save best model based on validation accuracy           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: PREDICTION                                         │
│  ─────────────────────                                       │
│  ⏳ TRAINING NOT STARTED                                    │
│  Expected Performance: >95% accuracy on test set           │
└─────────────────────────────────────────────────────────────┘

CONFIGURATION:
├── Model: vinai/phobert-base
├── Num Labels: 3
├── Learning Rate: 2e-5
├── Batch Size: 16
├── Epochs: 5
├── Max Length: 256
├── Optimizer: AdamW
└── Weight Decay: 0.01
```

---

## 3. 📈 TRẠNG THÁI CÁC MODEL

### 3.1 TỔNG QUAN

| Model | Trạng Thái | Accuracy | F1 Score | Training Time | Parameters |
|-------|-----------|----------|----------|---------------|------------|
| **SVM** | ✅ Hoàn thành | 100% | 100% | ~1 giây | N/A |
| **LSTM** | ❌ Bị dừng | 69.14%* | N/A | Epoch 1/10 | 2.45M |
| **PhoBERT** | ⏳ Chưa chạy | N/A | N/A | 30-60 phút | 85M |

*Chỉ accuracy sau epoch 1, training bị người dùng dừng.

### 3.2 CHI TIẾT SVM MODEL

```
✅ SVM MODEL - HOÀN THÀNH TỐT
═════════════════════════════════════════════════════════════

Training Results:
┌──────────────────────────────────────────────────────────┐
│ Metric          │ Train    │ Val      │ Test     │      │
├──────────────────────────────────────────────────────────┤
│ Accuracy        │ 100.00%  │ 100.00%  │ 100.00%  │      │
│ F1 Score        │ 100.00%  │ 100.00%  │ 100.00%  │      │
└──────────────────────────────────────────────────────────┘

Classification Report (Test Set):
                        precision  recall  f1-score  support

Negative               1.00      1.00    1.00       50
Neutral                1.00      1.00    1.00       50
Positive               1.00      1.00    1.00       50

accuracy               1.00                             150
macro avg              1.00      1.00    1.00       150
weighted avg           1.00      1.00    1.00       150

Confusion Matrix:
                Negative  Neutral  Positive
┌───────────────────────────────────────────┐
│ Negative  │      50   │    0    │    0    │
│ Neutral   │       0   │   50    │    0    │
│ Positive  │       0   │    0    │   50    │
└───────────────────────────────────────────┘

✅ Không có lỗi phân loại nào (Perfect Classification)
```

### 3.3 CHI TIẾT LSTM MODEL

```
❌ LSTM MODEL - BỊ NGƯỜI DÙNG DỪNG
═════════════════════════════════════════════════════════════

Training Progress:
┌──────────────────────────────────────────────────────────┐
│ Epoch 1/10:                                              │
│   • Train Loss: 0.7550  │ Train Acc: 69.14%             │
│   • Val Loss:   0.0771  │ Val Acc:   97.33%             │
│   ✅ Best model saved (Val Acc: 97.33%)                 │
├──────────────────────────────────────────────────────────┤
│ Epoch 2/10:                                              │
│   ❌ INTERRUPTED BY USER (KeyboardInterrupt)            │
└──────────────────────────────────────────────────────────┘

Vấn đề:
  • Training quá chậm trên CPU (không có GPU)
  • Người dùng đã huỷ khi train chưa hoàn thành
  • Cần 9 epoch nữa để hoàn thành
  • Model đã lưu tại epoch 1 có thể sử dụng được

Giải pháp:
  ▶️ Chạy lại training từ đầu (hoặc tiếp tục từ epoch 2)
  ▶️ Sử dụng GPU để tăng tốc (nếu có)
  ▶️ Giảm số epochs xuống 5-7 để train nhanh hơn
```

### 3.4 CHI TIẾT PhoBERT MODEL

```
⏳ PhoBERT MODEL - CHƯA BẮT ĐẦU TRAINING
══════────────════════════════════════════════════════════════

Setup Completed:
┌──────────────────────────────────────────────────────────┐
│ Tokenizer: ✅ Loaded from vinai/phobert-base            │
│ Model:     ✅ Loaded with num_labels=3                  │
│ Device:    ❌ CPU (no GPU available)                    │
│ Parameters: 85,000,000                                   │
└──────────────────────────────────────────────────────────┘

Dataloaders: ✅ Created
  • Train: 44 batches (16 samples/batch)
  • Val: 10 batches
  • Test: 10 batches

Training NOT Started:
  ⚠️ Cell training chưa được chạy
  ⏳ Thời gian ước tính: 30-60 phút trên CPU
  ⏳ Thời gian ước tính: 5-10 phút trên GPU

Để bắt đầu training:
  ▶️ Chạy cell "TRAIN PHOBERT MODEL"
  ▶️ Đảm bảo có đủ thời gian (30-60 phút nếu CPU)
  ▶️ Nên sử dụng Google Colab với GPU để train nhanh hơn
```

---

## 4. 📊 DATASET UIT-VSFC

### 4.1 THÔNG TIN DATASET

```
📦 UIT-VSFC (Vietnamese Students' Feedback Corpus)
═════════════════════════════════════════════════════════════

Source: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback

Paper: UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis

Distributions:
┌──────────────────────────────────────────────────────────┐
│ Split      │ Samples │ Percentage │ Status               │
├──────────────────────────────────────────────────────────┤
│ Train      │   700   │    70%     │ ✅ Đã sử dụng        │
│ Validation │   150   │    15%     │ ✅ Đã sử dụng        │
│ Test       │   150   │    15%     │ ✅ Đã sử dụng        │
├──────────────────────────────────────────────────────────┤
│ TOTAL      │  1,000  │   100%    │ ✅ 100% DATASET      │
└──────────────────────────────────────────────────────────┘

✅ CODE ĐÃ SỬ DỤNG TOÀN BỘ DATASET (100%)
```

### 4.2 PHÂN BỔ DỮ LIỆU

```
📊 SENTIMENT DISTRIBUTION
┌──────────────────────────────────────────────────────────┐
│ Sentiment │ Count │ Percentage │ Label                  │
├──────────────────────────────────────────────────────────┤
│ 0 (Neg)   │  333  │   33.3%    │ Negative              │
│ 1 (Neu)   │  334  │   33.4%    │ Neutral               │
│ 2 (Pos)   │  333  │   33.3%    │ Positive              │
└──────────────────────────────────────────────────────────┘
✅ Balanced dataset (mỗi class ~33.3%)

📊 TOPIC DISTRIBUTION
┌──────────────────────────────────────────────────────────┐
│ Topic     │ Count │ Percentage │ Label                  │
├──────────────────────────────────────────────────────────┤
│ 0         │  250  │   25.0%    │ Lecturer              │
│ 1         │  250  │   25.0%    │ Training Program      │
│ 2         │  250  │   25.0%    │ Facility              │
│ 3         │  250  │   25.0%    │ Others                │
└──────────────────────────────────────────────────────────┘
✅ Perfectly balanced (mỗi topic 25%)
```

### 4.3 THỐNG KÊ VĂN BẢN

```
📏 TEXT STATISTICS
┌──────────────────────────────────────────────────────────┐
│ Metric           │ Mean    │ Min  │ Max   │ Std         │
├──────────────────────────────────────────────────────────┤
│ Char Length      │  37.87  │  16  │  52   │  8.00       │
│ Word Count       │   8.65  │   4  │  12   │  1.81       │
└──────────────────────────────────────────────────────────┘

Sample Data:
┌──────────────────────────────────────────────────────────┐
│ Original: "Tài liệu học tập cũ và thiếu cập nhật"        │
│ Preprocessed: "tài liệu học tập cũ và thiếu cập nhật"    │
│ Tokenized: "tài_liệu học_tập cũ và thiếu cập nhật"      │
│ Sentiment: Negative (0)                                  │
│ Topic: Lecturer (0)                                      │
└──────────────────────────────────────────────────────────┘
```

---

## 5. 🔍 KẾT LUẬN & KHUYẾN NGHỊ

### 5.1 TÌNH TRẠNG HIỆN TẠI

```
✅ HOÀN THÀNH:
  • Data collection & exploration (100% dataset)
  • Text preprocessing & tokenization
  • Feature engineering (TF-IDF)
  • SVM model training (100% accuracy)
  • Model evaluation & visualization

❌ CHƯA HOÀN THÀNH:
  • LSTM model training (chỉ 1/10 epochs)
  • PhoBERT model training (chưa bắt đầu)
```

### 5.2 KHUYẾN NGHỊ

```
🎯 ĐỂ HOÀN THÀNH TOÀN BỘ PROJECT:

1️⃣ LSTM MODEL:
   ▶️ Chạy lại training (hoặc tiếp tục từ epoch 2)
   ▶️ Giảm epochs xuống 5-7 để train nhanh hơn
   ▶️ Sử dụng GPU nếu có (Google Colab, Kaggle)

2️⃣ PhoBERT MODEL:
   ▶️ Chạy training cell (đã setup xong)
   ▶️ Nên sử dụng Google Colab với GPU
   ▶️ Thời gian: 5-10 phút với GPU, 30-60 phút với CPU

3️⃣ OPTIONAL:
   ▶️ Thêm Logistic Regression, Naive Bayes
   ▶️ Thêm Random Forest để so sánh
   ▶️ Fine-tune hyperparameters

💡 LƯU Ý:
   • SVM đã đạt 100% accuracy → Dataset có thể quá dễ
   • Cần kiểm tra overfitting (data leakage?)
   • Nên thử với dataset lớn hơn hoặc harder
```

---

## 📚 TÀI LIỆU THAM KHẢO

- **Dataset**: [UIT-VSFC on Huggingface](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)
- **Paper**: [IEEE - Vietnamese Students' Feedback Corpus](https://ieeexplore.ieee.org/document/8573337)
- **PhoBERT**: [VinAI PhoBERT](https://github.com/VinAIResearch/PhoBERT)
- **Underthesea**: [Vietnamese NLP Toolkit](https://github.com/undertheseanlp/underthesea)

---

**Người tạo**: Student Feedback Sentiment Analysis Project
**Cập nhật**: 2025
**Total Cells**: 36 cells
**Models**: 3 models (SVM ✅, LSTM ❌, PhoBERT ⏳)
