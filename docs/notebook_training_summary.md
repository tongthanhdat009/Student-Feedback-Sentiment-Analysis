# Notebook Training Summary

Nguồn đọc:
- `notebook/PhoBERT_Baseline.ipynb`
- `notebook/PhoBERT_Sentiwordnet_Refactored_LightFusion_TopicAnalysis.ipynb`
- `notebook/PhoBERT_TF-IDF_Sentiwordnet_Baseline_Positional.ipynb`
- `notebook/PhoBERT_TFIDF_Refactored_LightFusion.ipynb`

Dataset chung: UIT-VSFC processed splits: train 11,426 / validation 1,583 / test 3,166. Labels: Negative=0, Neutral=1, Positive=2. Model nền: `vinai/phobert-base`, `MAX_LENGTH=256`.

## 1. `PhoBERT_Baseline.ipynb`

### Cấu trúc notebook
1. Setup Colab/Drive, cài thư viện, import.
2. Config đường dẫn + hyperparams.
3. Load `data/processed/{train,validation,test}/sents.txt` + `sentiments.txt`.
4. `SentimentDataset` tokenize text bằng PhoBERT tokenizer.
5. `PhoBERTClassifier`: PhoBERT encoder + classifier 3 lớp.
6. Training setup: `CrossEntropyLoss`, AdamW, linear warmup scheduler.
7. `train_epoch`, `evaluate`.
8. Training loop + early stopping + save best model.
9. Plot training history.
10. Test evaluation + confusion matrix + classification report.
11. Save summaries/visualizations/model.

### Phương pháp train
- Input: raw Vietnamese sentence.
- Feature/model: CLS/pooled PhoBERT representation → dropout/classifier → 3 logits.
- Loss: `nn.CrossEntropyLoss()`; không class weights; không augmentation.
- Optimizer: `AdamW(lr=2e-5, weight_decay=0.01)`.
- Scheduler: `get_linear_schedule_with_warmup`, warmup ratio `0.1`.
- Batch size: `16`.
- Epoch config trong notebook: `NUM_EPOCHS=5`; result file ghi `Epochs Trained=7` (có thể từ run cũ/lần chạy khác).
- Early stopping patience: `5`.
- Selection: best validation F1.

### Kết quả lưu trong repo
Từ `results/PhoBERT/baseline/summaries/summary.csv`:

| Metric | Value |
|---|---:|
| Best Val F1 | 0.9444968582 |
| Test Accuracy | 0.9320909665 |
| Test Precision weighted | 0.9284720006 |
| Test Recall weighted | 0.9320909665 |
| Test F1 weighted | 0.9290737995 |
| Test Loss | 0.3290720447 |

Artifacts chính:
- `results/PhoBERT/baseline/models/phobert_model.pt`
- `results/PhoBERT/baseline/visualizations/confusion_matrix.png`
- `results/PhoBERT/baseline/visualizations/training_history.png`

## 2. `PhoBERT_Sentiwordnet_Refactored_LightFusion_TopicAnalysis.ipynb`

### Cấu trúc notebook
1. Setup Colab/local root detection.
2. Import libs + `src.data_utils`.
3. Config experiment.
4. Load train/val/test.
5. Load VietSentiWordNet + extract extended SWN features.
6. Scale SWN features with `StandardScaler`.
7. `HybridDataset` chứa text + SWN features + label.
8. `PhoBERTSentiWordNetLightHybrid`.
9. DataLoaders.
10. Evaluation fn.
11. Load pretrained checkpoint + topic-wise test analysis.
12. Topic visualizations.

### Phương pháp train/eval
Notebook này chủ yếu phục vụ topic-wise evaluation từ pretrained checkpoint, nhưng cấu hình/model tương ứng training Light Fusion:
- Input: text + compact SentiWordNet features.
- SWN features: `extract_swn_features_extended_batch`, feature names `SWN_EXTENDED_FEATURE_NAMES`.
- Preprocess text: `preprocess_vietnamese`.
- Scale SWN: `StandardScaler` fit train, transform val/test.
- Model: PhoBERT CLS token trực tiếp + SWN projection.
- Fusion: `FUSION_MODE='concat'` (có option gated trong thiết kế).
- Dimensions: `SWN_PROJ_DIM=128`, `CLASSIFIER_HIDDEN_DIM=256`.
- Dropout: `0.3`.
- Batch size: `32`.
- Selection metric: `f1_macro`.
- Class weights: `CLASS_WEIGHT_MODE='sqrt_balanced'`.
- Checkpoint trong notebook: `results/PhoBERT_Sentiwordnet_Refactored_LightFusion/improvements/20260322_045916/models/best_model.pt`.

### Kết quả lưu trong repo
Best matching run: `results/PhoBERT_Sentiwordnet_Refactored_LightFusion/improvements/20260322_045916/summaries/summary.csv`.

| Split | Best Epoch | Accuracy | F1 Macro | F1 Weighted | F1 Negative | F1 Neutral | F1 Positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 7 | 0.9437776374 | 0.8536019556 | 0.9428280660 | 0.9530483532 | 0.6470588235 | 0.9606986900 |
| Test | 7 | 0.9317751105 | 0.8281559452 | 0.9298391608 | 0.9481017067 | 0.5866666667 | 0.9496994622 |

Earlier run `20260321_060451`:
- Test Accuracy `0.9301958307`
- Test F1 Macro `0.8300916037`
- Test F1 Weighted `0.9304225394`
- Test F1 Neutral `0.5917159763`

Topic-analysis outputs exist in later folders, e.g.:
- `summaries/topic_metrics_from_pretrained.csv`
- `summaries/topic_predictions_test.csv`
- `visualizations/topic_confusion_matrices.png`
- `visualizations/topic_f1_scores.png`

## 3. `PhoBERT_TF-IDF_Sentiwordnet_Baseline_Positional.ipynb`

### Cấu trúc notebook
Notebook compact JSON nhưng nội dung chính:
1. Setup Colab/local root detection.
2. Import libs + sklearn + `src.data_utils`.
3. Config experiment.
4. Load data + SentiWordNet.
5. Build TF-IDF features + SWN features.
6. `HybridDataset` chứa text + TF-IDF + SWN + label.
7. `PositionalEncoding`.
8. `PhoBERTTFIDFSentiWordNetHybrid`.
9. Gradual unfreezing training.
10. Eval + save summary/model/artifacts.

### Phương pháp train
- Input: text + TF-IDF vector + SentiWordNet features.
- TF-IDF config:
  - `TFIDF_MAX_FEATURES=5000`
  - `TFIDF_NGRAM_RANGE=(1, 2)`
  - `TFIDF_MIN_DF=3`
  - `TFIDF_MAX_DF=0.90`
  - `TFIDF_SUBLINEAR_TF=True`
- SWN: extended SentiWordNet features scaled with `StandardScaler`.
- Model:
  - PhoBERT encoder.
  - Positional encoding over PhoBERT sequence output.
  - TF-IDF projection: Linear → SiLU → Dropout, dim `256`.
  - SWN projection: Linear → SiLU → Dropout, dim `64`.
  - Concatenate PhoBERT + TF-IDF + SWN.
  - Fusion projection + residual block + classifier.
- Gradual unfreezing:
  - Epoch 1 frozen PhoBERT (`PHOBERT_LR_FROZEN=0.0`).
  - Epoch 2 partial unfreeze last 4 layers (`PHOBERT_LR_PARTIAL=1e-5`).
  - Epoch 4 full unfreeze (`PHOBERT_LR_FULL=2e-5`).
- Head LR: `1e-4`.
- Batch size: `16`.
- Epochs: `10`.
- Early stop patience: `2`.
- Warmup ratio: `0.1`.
- Gradient clip: `1.0`.
- Weight decay: `0.01`.
- Dropout: `0.3`.
- Selection metric: `f1_macro`.

### Kết quả lưu trong repo
Best run among matching saved runs appears `20260329_050139` by test F1 macro.

From `results/PhoBERT_TF-IDF_Sentiwordnet/end_to_end_unfreezing/20260329_050139/summaries/summary.csv`:

| Split | Best Epoch | Accuracy | F1 Macro | F1 Weighted | F1 Negative | F1 Neutral | F1 Positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 6 | 0.9431459255 | 0.8436731084 | 0.9417841608 | 0.9531795947 | 0.6165413534 | 0.9612983770 |
| Test | 6 | 0.9346178143 | 0.8349466907 | 0.9328696287 | 0.9484029484 | 0.6026490066 | 0.9537881169 |

Second run `20260329_074056`:
- Test Accuracy `0.9317751105`
- Test F1 Macro `0.8297551049`
- Test F1 Weighted `0.9296907837`
- Test F1 Neutral `0.5925925926`

## 4. `PhoBERT_TFIDF_Refactored_LightFusion.ipynb`

### Cấu trúc notebook
1. Setup Colab/local root detection.
2. Import libs + sklearn + data utils.
3. Config experiment.
4. Load data.
5. TF-IDF feature extraction.
6. Optional TF-IDF dimensionality reduction with LSA/SVD.
7. `HybridDataset` text + TF-IDF + label.
8. `PhoBERTTFIDFLightHybrid`.
9. DataLoaders.
10. Training setup with gradual unfreezing.
11. Training loop.
12. Validation/test evaluation.
13. Save results.
14. Visualizations.
15. Neutral-class error analysis.
16. Final summary + recommended next experiments.

### Phương pháp train
- Input: text + TF-IDF lexical features.
- TF-IDF config:
  - `TFIDF_MAX_FEATURES=5000`
  - `TFIDF_NGRAM_RANGE=(1, 2)`
  - `TFIDF_MIN_DF=3`
  - `TFIDF_MAX_DF=0.90`
  - `TFIDF_SUBLINEAR_TF=True`
- Reduction: `TFIDF_REDUCTION='lsa'`, `TFIDF_LSA_COMPONENTS=256`, `n_iter=10`, random state 42.
- Model: PhoBERT CLS token + TF-IDF projection + light fusion classifier.
- Fusion: `FUSION_MODE='concat'` (option gated).
- TF-IDF projection dim: `96`.
- Classifier hidden dim: `256`.
- Dropout: `0.3`.
- Class weights: `CLASS_WEIGHT_MODE='sqrt_balanced'`.
- Gradual unfreezing:
  - frozen first epoch (`PHOBERT_LR_FROZEN=0.0`)
  - partial unfreeze from epoch 2, last 4 layers (`PHOBERT_LR_PARTIAL=1e-5`)
  - full unfreeze from epoch 4 (`PHOBERT_LR_FULL=2e-5`)
- Head LR:
 `5e-5`.
- Batch size: `16`.
- Epochs: `15`.
- Early stop patience: `3`.
- Warmup ratio: `0.1`.
- Gradient clip: `1.0`.
- Weight decay: `0.01`.
- Selection metric: `f1_macro`.

### Kết quả lưu trong repo
Best matching run by test F1 macro: `results/PhoBERT_TFIDF_Refactored_LightFusion/improvements/20260320_132355/summaries/summary.csv`.

| Split | Best Epoch | Accuracy | F1 Macro | F1 Weighted | F1 Negative | F1 Neutral | F1 Positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 10 | 0.9450410613 | 0.8558930235 | 0.9440103439 | 0.9545772187 | 0.6518518519 | 0.9612500000 |
| Test | 10 | 0.9324068225 | 0.8266144211 | 0.9304225227 | 0.9496855346 | 0.5800000000 | 0.9501577287 |

Second run `20260320_173649`:
- Test Accuracy `0.9327226785`
- Test F1 Macro `0.8225827915`
- Test F1 Weighted `0.9291263086`
- Test F1 Neutral `0.5693430657`

Optimized variant exists outside notebook name: `results/PhoBERT_TFIDF_Refactored_LightFusion_Optimized/optimized/20260322_081746`; uses neutral bias tuning but test F1 macro lower (`0.8158653178`).

## So sánh nhanh

| Notebook | Signals | Training style | Best test Acc | Best test F1 macro | Best test F1 weighted | Neutral F1 |
|---|---|---|---:|---:|---:|---:|
| PhoBERT Baseline | PhoBERT only | CE, AdamW, warmup | 0.9321 | N/A in summary | 0.9291 | N/A |
| PhoBERT + SWN LightFusion | PhoBERT + SWN | light concat, sqrt class weights | 0.9318 | 0.8282 | 0.9298 | 0.5867 |
| PhoBERT + TF-IDF + SWN Positional | PhoBERT + TF-IDF + SWN | residual hybrid, gradual unfreezing | 0.9346 | 0.8349 | 0.9329 | 0.6026 |
| PhoBERT + TF-IDF LightFusion | PhoBERT + TF-IDF | light concat, LSA, gradual unfreezing | 0.9324 | 0.8266 | 0.9304 | 0.5800 |

## Nhận xét chính
- Baseline PhoBERT rất mạnh về weighted metrics; hybrid không vượt rõ rệt toàn cục.
- Hybrid TF-IDF+SWN positional cho test accuracy/F1 macro tốt nhất trong 4 notebook có kết quả lưu.
- Neutral class yếu nhất do imbalance (~4–5% samples). Hybrid/weighting cải thiện neutral F1 tới khoảng `0.58–0.60` test, cao hơn trên validation ở vài run.
- Light fusion giữ PhoBERT gần baseline, giảm nguy cơ auxiliary features phá representation; positional TF-IDF+SWN model phức tạp hơn, kết quả test tốt hơn trong run `20260329_050139`.
