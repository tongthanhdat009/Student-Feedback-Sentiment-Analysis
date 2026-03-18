# So sánh các notebook PhoBERT hybrid và kế hoạch xây dựng `PhoBERT_TF-IDF_Sentiwordnet_Baseline.ipynb`

## Phạm vi

Tài liệu này đọc và tổng hợp từ 3 notebook:

- `notebook/PhoBERT_Baseline.ipynb`
- `notebook/PhoBERT_Sentiwordnet_Improved.ipynb`
- `notebook/PhoBERT_TF-IDF_EndToEnd_FineTuning.ipynb`

Để đưa ra kế hoạch xây dựng notebook mới, tài liệu cũng tham chiếu thêm kết quả đã có từ:

- `results/PhoBERT_Sentiwordnet/baseline/summaries/summary.csv`
- `results/PhoBERT_TF-IDF/baseline/20260305/summaries/summary.csv`

## 1. Tóm tắt từng notebook

### 1.1. `PhoBERT_Baseline.ipynb`

Mục tiêu:
- Xây baseline chuẩn bằng `vinai/phobert-base`
- Fine-tune trực tiếp với classifier tuyến tính
- Không dùng feature thủ công
- Không dùng class weights
- Không dùng SMOTE

Thiết lập chính:
- Backbone: `vinai/phobert-base`
- Max length: `256`
- Batch size: `16`
- Learning rate: `2e-5`
- Epochs: `5`
- Loss: `CrossEntropyLoss`

Kết quả chính:
- Best validation F1 (weighted): `0.9455`
- Test Accuracy: `0.9324`
- Test F1 (weighted): `0.9307`

Per-class test F1:
- Negative: `0.9482`
- Neutral: `0.5789`
- Positive: `0.9520`

Nhận xét:
- Đây là mốc tham chiếu mạnh nhất về overall performance trong 3 notebook được yêu cầu đọc.
- Neutral vẫn là lớp khó nhất, recall chỉ `0.5269`.

### 1.2. `PhoBERT_Sentiwordnet_Improved.ipynb`

Mục tiêu:
- Kết hợp PhoBERT trainable với đặc trưng từ VietSentiWordNet
- Fine-tune end-to-end
- Tăng số đặc trưng SentiWordNet từ 8 lên 35
- Tăng chú ý vào lớp Neutral bằng class weights

Thiết lập chính:
- Backbone: `vinai/phobert-base` unfrozen
- SentiWordNet features: `35`
- Projection SentiWordNet: `35 -> 64`
- Combined features: `768 + 64 = 832`
- Classifier: `832 -> 256 -> 3`
- Batch size: `32`
- Learning rate: `2e-5`
- Epochs: `5`
- Loss: `CrossEntropyLoss(weight=class_weights)`
- Class weights: `[1.0, 5.0, 1.0]`

Kết quả chính:
- Best validation F1 (macro): `0.8483`
- Test Accuracy: `0.9274`
- Test F1 (macro): `0.8136`

Per-class test F1:
- Negative: `0.9483`
- Neutral: `0.5466`
- Positive: `0.9459`

Nhận xét:
- Kiến trúc phức tạp hơn baseline nhưng không vượt được baseline PhoBERT thuần.
- Việc thêm 35 đặc trưng SentiWordNet và class weight chưa cải thiện được lớp Neutral so với baseline PhoBERT.

### 1.3. `PhoBERT_TF-IDF_EndToEnd_FineTuning.ipynb`

Mục tiêu:
- Kết hợp embedding từ PhoBERT trainable với TF-IDF 5000 chiều
- Fine-tune end-to-end
- Dùng SMOTE để hỗ trợ lớp Neutral

Thiết lập chính:
- Backbone: `vinai/phobert-base` unfrozen
- TF-IDF: `5000` features, `ngram_range=(1, 2)`
- TF-IDF projection: `5000 -> 256`
- Combined features: `768 + 256 = 1024`
- Batch size: `16`
- Learning rate: `2e-5`
- Epochs trained: `4` (early stopping)
- SMOTE: `True`
- Class weights: `balanced` từ dữ liệu sau resampling

Kết quả chính:
- Best validation F1 (weighted): `0.9443`
- Test Accuracy: `0.9311`
- Test F1 (weighted): `0.9291`

Per-class test F1:
- Negative: `0.9495`
- Neutral: `0.5753`
- Positive: `0.9481`

Nhận xét:
- Overall gần với baseline PhoBERT nhưng vẫn thấp hơn nhẹ.
- Neutral F1 gần như ngang baseline PhoBERT, không có bước nhảy rõ rệt dù đã dùng TF-IDF và SMOTE.

## 2. Bảng so sánh kết quả

### 2.1. So sánh theo metric notebook đang báo cáo

| Notebook | Kiểu mô hình | Accuracy | Metric F1 được báo cáo | Giá trị |
|---|---|---:|---|---:|
| `PhoBERT_Baseline` | PhoBERT fine-tune thuần | `0.9324` | Weighted F1 | `0.9307` |
| `PhoBERT_Sentiwordnet_Improved` | PhoBERT + SentiWordNet end-to-end | `0.9274` | Macro F1 | `0.8136` |
| `PhoBERT_TF-IDF_EndToEnd_FineTuning` | PhoBERT + TF-IDF end-to-end | `0.9311` | Weighted F1 | `0.9291` |

Lưu ý:
- Không nên so sánh trực tiếp `0.9307`, `0.8136`, `0.9291` như cùng một metric.
- `weighted F1` ưu tiên lớp lớn hơn.
- `macro F1` phạt mạnh hơn khi lớp Neutral kém.

### 2.2. So sánh công bằng hơn theo per-class

| Notebook | F1 Negative | F1 Neutral | F1 Positive |
|---|---:|---:|---:|
| `PhoBERT_Baseline` | `0.9482` | `0.5789` | `0.9520` |
| `PhoBERT_Sentiwordnet_Improved` | `0.9483` | `0.5466` | `0.9459` |
| `PhoBERT_TF-IDF_EndToEnd_FineTuning` | `0.9495` | `0.5753` | `0.9481` |

Kết luận từ bảng này:
- `PhoBERT_Baseline` vẫn là mốc mạnh nhất nếu nhìn tổng thể.
- `PhoBERT_TF-IDF_EndToEnd_FineTuning` gần baseline nhất, nhưng chưa vượt.
- `PhoBERT_Sentiwordnet_Improved` không giúp lớp Neutral tốt hơn; ngược lại còn thấp hơn baseline.

## 3. So sánh với các baseline hybrid đã có trong repo

### 3.1. PhoBERT + SentiWordNet baseline hiện có

Nguồn: `results/PhoBERT_Sentiwordnet/baseline/summaries/summary.csv`

Thiết kế:
- Dùng PhoBERT baseline đã fine-tune để trích embedding
- Trích `35` đặc trưng SentiWordNet
- Kết hợp feature theo kiểu hybrid
- Classifier: `LogisticRegression`

Kết quả test:
- Accuracy: `0.9283`
- F1 macro: `0.8194`
- Neutral F1: `0.5641`

Ý nghĩa:
- Hybrid kiểu feature-engineering + classifier nông có cải thiện nhẹ trên validation so với PhoBERT-only theo macro F1.
- Nhưng test vẫn không vượt baseline PhoBERT về overall performance.

### 3.2. PhoBERT + TF-IDF baseline hiện có

Nguồn: `results/PhoBERT_TF-IDF/baseline/20260305/summaries/summary.csv`

Thiết kế:
- Dùng PhoBERT pretrained/fine-tuned làm embedding source
- Kết hợp TF-IDF `5000` features
- Classifier: `LogisticRegression`

Kết quả test:
- Accuracy: `0.9305`
- F1 weighted: `0.9296`

Ý nghĩa:
- TF-IDF hybrid baseline đã tiến khá sát PhoBERT baseline.
- Đây là hướng có tín hiệu tốt hơn SentiWordNet nếu mục tiêu là giữ overall score cao.

## 4. Kết luận

### 4.1. Kết luận về hiệu quả mô hình

Kết luận chính:
- `PhoBERT_Baseline` hiện vẫn là mô hình tốt nhất để làm chuẩn so sánh.
- `PhoBERT_TF-IDF_EndToEnd_FineTuning` là hướng bổ sung feature đáng giữ lại nhất vì hiệu năng gần baseline nhất và vẫn có tiềm năng cải thiện lớp Neutral.
- `PhoBERT_Sentiwordnet_Improved` chưa chứng minh được lợi ích thực sự. Việc thêm lexicon sentiment không tự động tạo ra cải thiện.

### 4.2. Kết luận về đặc trưng bổ sung

TF-IDF:
- Có tín hiệu thực tế tốt hơn SentiWordNet.
- Giúp mô hình giữ được nhiều tín hiệu từ từ/cụm từ cụ thể trong feedback.

SentiWordNet:
- Có thể hữu ích như feature phụ nhẹ.
- Nhưng đứng một mình chưa đủ mạnh để kéo hiệu năng lên.
- Nếu ghép với TF-IDF, SentiWordNet nên được xem là nhánh nhỏ hỗ trợ, không phải nhánh chính.

### 4.3. Kết luận cho notebook mới

Notebook `PhoBERT_TF-IDF_Sentiwordnet_Baseline.ipynb` nên được xây như một baseline hybrid theo kiểu:
- Dùng **PhoBERT embedding đã có**
- Cộng thêm **TF-IDF**
- Cộng thêm **SentiWordNet features**
- Huấn luyện bằng **classifier nông** như `LogisticRegression`

Không nên xây notebook mới theo kiểu end-to-end ngay từ đầu, vì:
- Chi phí chạy cao hơn
- Khó kiểm soát tác động riêng của từng feature
- Các thử nghiệm end-to-end hiện tại chưa chứng minh được lợi ích vượt baseline

## 5. Kế hoạch xây dựng `PhoBERT_TF-IDF_Sentiwordnet_Baseline.ipynb`

## Mục tiêu

Xây một baseline rõ ràng để trả lời câu hỏi:

> Khi ghép cả `PhoBERT embedding + TF-IDF + SentiWordNet`, feature nào thực sự mang lại thêm tín hiệu trên tập student feedback?

## Thiết kế đề xuất

Pipeline đề xuất:

1. Load dữ liệu `train/validation/test` từ `data/processed`
2. Load PhoBERT tokenizer và mô hình baseline đã fine-tune
3. Trích embedding PhoBERT cho từng câu
4. Tiền xử lý văn bản và tạo TF-IDF features
5. Load VietSentiWordNet và trích `35` extended features
6. Scale từng nhóm feature riêng:
   - PhoBERT embeddings
   - TF-IDF
   - SentiWordNet
7. Tạo nhiều tổ hợp feature để so sánh:
   - `PhoBERT`
   - `PhoBERT + TF-IDF`
   - `PhoBERT + SentiWordNet`
   - `PhoBERT + TF-IDF + SentiWordNet`
8. Train `LogisticRegression` cho từng tổ hợp
9. Chọn mô hình theo validation
10. Đánh giá trên test
11. Lưu model, scaler, vectorizer, summary, confusion matrix

## Các biến thể cần chạy

Để notebook mới có giá trị phân tích, nên chạy tối thiểu các cấu hình sau:

- `PhoBERT-only`
- `PhoBERT + TF-IDF`
- `PhoBERT + SentiWordNet`
- `PhoBERT + TF-IDF + SentiWordNet`

Nếu còn thời gian, thêm:
- `PhoBERT + TF-IDF + SentiWordNet` với `class_weight='balanced'`
- `PhoBERT + TF-IDF + SentiWordNet` với vài hệ số scale cho SentiWordNet, ví dụ `alpha in [0.5, 1.0, 2.0]`

Lý do:
- SentiWordNet chỉ có 35 features, rất dễ bị TF-IDF 5000 chiều lấn át.
- Cần kiểm tra có nên nhân trọng số cho nhánh SentiWordNet hay không.

## Hyperparameter baseline đề xuất

PhoBERT:
- Dùng embedding từ mô hình đã fine-tune ở `results/PhoBERT/baseline/models/phobert_model.pt`
- Không fine-tune lại trong notebook này

TF-IDF:
- `max_features=5000`
- `ngram_range=(1, 2)`
- `min_df=2` hoặc `3`
- `max_df=0.90` hoặc `0.95`
- `sublinear_tf=True`

SentiWordNet:
- Dùng `35` extended features
- Scale bằng `StandardScaler`

Classifier:
- `LogisticRegression`
- `multi_class='multinomial'`
- `max_iter=2000`
- Grid nhỏ cho `C`: `[0.01, 0.1, 1.0]`
- `class_weight`: `[None, 'balanced']`

Selection metric:
- Ưu tiên `F1 macro` trên validation
- Báo cáo thêm `Accuracy`, `F1 weighted`, `F1 neutral`

Lý do chọn `F1 macro`:
- Bài toán đang mất cân bằng lớp rõ rệt
- Nếu chỉ tối ưu `weighted F1`, mô hình rất dễ đẹp ở lớp Negative/Positive nhưng bỏ sót Neutral

## Cấu trúc notebook đề xuất

Các section nên có:

1. `Setup và cấu hình`
2. `Load dữ liệu`
3. `Load PhoBERT baseline và extract embedding`
4. `TF-IDF feature extraction`
5. `SentiWordNet feature extraction`
6. `Feature scaling và feature combinations`
7. `Training + model selection`
8. `Evaluation trên validation/test`
9. `Visualization`
10. `Save artifacts và summary`
11. `Final comparison với các notebook trước`

## Output cần lưu

Thư mục đề xuất:
- `results/PhoBERT_TF-IDF_Sentiwordnet/baseline/<timestamp>/`

Artifacts cần lưu:
- `models/best_model.pkl`
- `artifacts/tfidf_vectorizer.pkl`
- `artifacts/phobert_scaler.pkl`
- `artifacts/swn_scaler.pkl`
- `summaries/summary.csv`
- `summaries/model_comparison.csv`
- `summaries/experiment_summary.json`
- `visualizations/confusion_matrix.png`
- `visualizations/per_class_metrics.png`
- `visualizations/model_comparison_bar.png`

## Tiêu chí thành công

Notebook mới được xem là có giá trị nếu đạt ít nhất một trong các mục tiêu sau:

- Test `F1 macro` cao hơn `PhoBERT + SentiWordNet baseline`
- Test `Neutral F1` cao hơn `PhoBERT_Baseline`
- Test `weighted F1` không giảm quá nhiều so với `PhoBERT_Baseline`

Ngưỡng thực tế nên kỳ vọng:
- `Neutral F1 >= 0.59`
- `Weighted F1 >= 0.928`
- `Macro F1 >= 0.82`

## Rủi ro kỹ thuật cần lưu ý

- TF-IDF có chiều cao, dễ lấn át SentiWordNet nếu không scale hoặc không re-weight.
- Nếu dùng quá nhiều biến thể trong cùng notebook, thời gian chạy sẽ dài và khó đọc.
- Nếu chỉ nhìn accuracy, có thể chọn nhầm mô hình kém ở lớp Neutral.
- `PhoBERT_TF-IDF_Sentiwordnet_Baseline.ipynb` hiện chưa là notebook hợp lệ, nên nên tạo mới sạch thay vì sửa tiếp file cũ.

## Khuyến nghị thực thi

Thứ tự triển khai nên là:

1. Tạo notebook baseline theo kiểu feature-based, không end-to-end
2. Dùng cùng cách extract embedding như notebook hybrid baseline hiện có
3. So sánh 4 tổ hợp feature trong cùng một notebook
4. Chọn metric chính là `validation macro F1`
5. Chỉ khi triple-hybrid baseline có tín hiệu tốt, mới làm bước tiếp theo là end-to-end triple-hybrid

## Kết luận cuối cùng

Nếu mục tiêu hiện tại là xây một notebook mới có giá trị nghiên cứu và dễ so sánh, hướng đúng là:

- **Không** lao ngay vào mô hình end-to-end phức tạp
- **Nên** xây `PhoBERT_TF-IDF_Sentiwordnet_Baseline.ipynb` như một baseline feature-based có kiểm soát
- **Ưu tiên kiểm tra xem SentiWordNet có đóng góp thêm gì khi TF-IDF đã có mặt hay không**

Nói ngắn gọn:
- PhoBERT thuần đang là chuẩn mạnh nhất
- TF-IDF là feature bổ sung có tín hiệu tốt nhất
- SentiWordNet chỉ nên được kiểm chứng như feature hỗ trợ nhỏ trong baseline mới
