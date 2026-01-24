# 📊 Student Feedback Sentiment Analysis

Phân tích cảm xúc phản hồi sinh viên sử dụng dataset **UIT-VSFC** (Vietnamese Students' Feedback Corpus).

## 🚀 Bắt đầu nhanh

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Setup Project (Download dataset và cache models)

```bash
python setup_project.py
```

Script này sẽ:
- ✅ Tạo tất cả thư mục cần thiết
- ✅ Download và cache dataset UIT-VSFC
- ✅ Download và cache PhoBERT model

> 💡 **Lưu ý**: Dataset và models chỉ cần download **một lần**. Các lần chạy sau sẽ load từ cache!

### 3. Các bước tiếp theo

```bash
# Preprocessing
python preprocessing/build_dataset.py

# Train models
python training/train_svm.py
python training/train_lstm.py
python training/train_phobert.py

# Run dashboard
streamlit run dashboard/app.py
```

## 📁 Cấu trúc thư mục

```
project/
│
├── data/                      # Dữ liệu
│   ├── raw/uit_vsfc/         # Dataset gốc (CSV)
│   └── processed/acsa/       # Dataset đã xử lý
│
├── cache/                     # 📦 CACHE (không cần download lại)
│   ├── models/               # PhoBERT model cache
│   └── datasets/             # Dataset cache
│
├── saved_models/             # 💾 Models đã train
│   ├── svm/
│   ├── lstm/
│   └── phobert/
│
├── preprocessing/            # Tiền xử lý dữ liệu
├── models/                   # Định nghĩa models
├── training/                 # Scripts huấn luyện
├── evaluation/               # Đánh giá models
├── visualization/            # Biểu đồ (Matplotlib)
├── dashboard/                # Streamlit dashboard
├── inference/                # Suy luận và API
├── configs/                  # Cấu hình
├── utils/                    # Utility functions
│
├── requirements.txt          # Thư viện cần cài
├── setup_project.py          # Script setup ban đầu
└── README.md
```

## 🔧 Cấu hình Cache

Đường dẫn cache được cấu hình trong `configs/config.yaml`:

```yaml
paths:
  model_cache: "./cache/models"      # Cache PhoBERT
  dataset_cache: "./cache/datasets"  # Cache dataset
  saved_models: "./saved_models"     # Models đã train
```

### Kiểm tra trạng thái cache

```bash
python utils/cache_manager.py
```

## 📊 Dataset

**UIT-VSFC** (Vietnamese Students' Feedback Corpus):
- **Sentences**: Phản hồi sinh viên bằng tiếng Việt
- **Sentiment**: 0 (Negative), 1 (Neutral), 2 (Positive)
- **Topic**: 0 (Lecturer), 1 (Training Program), 2 (Facility), 3 (Others)

### Download dataset riêng

```bash
python data/download_dataset.py
```

## 🤖 Models

| Model | Mô tả | Mục đích |
|-------|-------|----------|
| **SVM** | Support Vector Machine | Baseline |
| **LSTM** | Long Short-Term Memory | Deep Learning |
| **PhoBERT** | Vietnamese BERT | State-of-the-art |

## 📈 Dashboard

Dashboard Streamlit với các visualization:

- **Model Comparison**: So sánh Accuracy/F1 các models
- **Sentiment Distribution**: Phân bố cảm xúc
- **Aspect-wise Sentiment**: Cảm xúc theo từng khía cạnh
- **Demo Prediction**: Demo dự đoán trực tiếp

```bash
streamlit run dashboard/app.py
# Truy cập: http://localhost:8501
```

## 📚 Tham khảo

- **Dataset**: [UIT-VSFC on Huggingface](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)
- **Paper**: [IEEE - Vietnamese Students' Feedback Corpus](https://ieeexplore.ieee.org/document/8573337)
- **PhoBERT**: [VinAI PhoBERT](https://github.com/VinAIResearch/PhoBERT)

## 📄 License

MIT License
