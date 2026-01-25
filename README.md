# 📊 Student Feedback Sentiment Analysis

Phân tích cảm xúc phản hồi sinh viên sử dụng dataset **UIT-VSFC** (Vietnamese Students' Feedback Corpus).

## 🚀 Bắt đầu nhanh

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chạy Jupyter Notebook

Mở file `Student_Feedback_Sentiment_Analysis.ipynb` và chạy từng cell theo thứ tự.

```bash
jupyter notebook Student_Feedback_Sentiment_Analysis.ipynb
```

**Hoặc trên Google Colab:**
- Upload file notebook lên Google Colab
- Chạy tất cả cells

### 📓 Notebook bao gồm:
- ✅ Download dataset từ Huggingface API
- ✅ Exploratory Data Analysis với biểu đồ trực quan
- ✅ Preprocessing văn bản tiếng Việt
- ✅ Train các models: SVM, Logistic Regression, Naive Bayes, Random Forest, LSTM, PhoBERT
- ✅ Đánh giá và so sánh models
- ✅ Lưu models và kết quả
- ✅ Inference trên văn bản mới

> 💡 **Lưu ý**: Dataset và PhoBERT models được cache tự động trong thư mục `cache/`. Các lần chạy sau sẽ load từ cache!

## 📁 Cấu trúc thư mục

```
project/
│
├── Student_Feedback_Sentiment_Analysis.ipynb  # 📓 Main Notebook
│
├── cache/                     # 📦 CACHE (tự động tạo)
│   ├── models/               # PhoBERT model cache
│   └── datasets/             # Dataset cache
│
├── saved_models/             # 💾 Models đã train
│   ├── sklearn/              # ML models (SVM, LR, NB, RF)
│   ├── lstm_best.pt          # LSTM model
│   └── phobert_best/         # PhoBERT model
│
├── results/                  # 📊 Kết quả
│   ├── figures/              # Biểu đồ PNG
│   ├── model_comparison.csv  # So sánh models
│   └── experiment_results.json
│
├── requirements.txt          # Thư viện cần cài
└── README.md
```

## 🔧 Cấu hình Cache

Models và Dataset được cache tự động:
- **PhoBERT**: `cache/models/`
- **Dataset**: `cache/datasets/`
- **Trained Models**: `saved_models/`

## 💻 Yêu cầu hệ thống

- Python >= 3.8
- PyTorch >= 1.9.0
- GPU (khuyến nghị) hoặc CPU
- RAM >= 8GB

## 📊 Dataset

**UIT-VSFC** (Vietnamese Students' Feedback Corpus):
- **Source**: [Huggingface API](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)
- **Sentences**: Phản hồi sinh viên bằng tiếng Việt
- **Sentiment**: 0 (Negative), 1 (Neutral), 2 (Positive)
- **Topic**: 0 (Lecturer), 1 (Training Program), 2 (Facility), 3 (Others)

## 🤖 Models

| Model | Type | Mô tả |
|-------|------|-------|
| **Logistic Regression** | ML | Baseline model |
| **SVM** | ML | Support Vector Machine |
| **Naive Bayes** | ML | Multinomial NB |
| **Random Forest** | ML | Ensemble method |
| **LSTM** | Deep Learning | Bidirectional LSTM |
| **PhoBERT** | Transformer | Vietnamese BERT (State-of-the-art) |

## 📈 Visualization

Notebook tạo các biểu đồ sau (lưu trong `results/figures/`):
- **Sentiment Distribution**: Phân bố cảm xúc
- **Topic Distribution**: Phân bố chủ đề
- **Sentiment vs Topic Heatmap**: Ma trận tương quan
- **Text Length Distribution**: Phân bố độ dài văn bản
- **Model Comparison**: So sánh Accuracy/F1 các models
- **Confusion Matrices**: Ma trận nhầm lẫn
- **Training History**: Loss và Accuracy qua từng epoch
- **Radar Chart**: So sánh metrics tổng hợp

## 📚 Tham khảo

- **Dataset**: [UIT-VSFC on Huggingface](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)
- **Paper**: [IEEE - Vietnamese Students' Feedback Corpus](https://ieeexplore.ieee.org/document/8573337)
- **PhoBERT**: [VinAI PhoBERT](https://github.com/VinAIResearch/PhoBERT)

## 📄 License

MIT License
