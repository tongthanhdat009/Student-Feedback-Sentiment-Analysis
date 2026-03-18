# Notebook Analysis Report

## Summary of Current Results

| Notebook | Test Acc | Test F1 Macro | Neutral F1 | Val F1 | Selection Metric |
|----------|----------|---------------|------------|--------|------------------|
| PhoBERT_Sentiwordnet_Improved | 0.9274 | 0.8136 | 0.5466 | 0.8483 (macro) | macro-F1 |
| PhoBERT_TF-IDF_EndToEnd | 0.9311 | ~0.82 | 0.5753 | 0.9443 (weighted!) | **WEIGHTED-F1** |
| PhoBERT_TF-IDF_Sentiwordnet | 0.9274 | 0.8183 | 0.5631 | 0.8607 (macro) | macro-F1 |

---

## Notebook 1: PhoBERT_Sentiwordnet_Improved.ipynb

### Architecture
- PhoBERT (768-dim, UNFROZEN immediately)
- SentiWordNet 35 extended features → 64-dim projection
- Simple concatenation: 768 + 64 = 832
- Classifier: 832 → 256 → 3

### Strengths
- Uses extended SentiWordNet features (35)
- Has class weights [1.0, 5.0, 1.0]
- Uses macro-F1 for model selection (correct)

### Critical Issues
1. **No gradual unfreezing**: PhoBERT is immediately trainable from epoch 1, risking catastrophic forgetting
2. **Fixed class weights**: [1.0, 5.0, 1.0] is arbitrary; should use computed class weights
3. **No discriminative learning rates**: Same LR for PhoBERT and classifier
4. **Simple concatenation fusion**: No learned interaction between modalities
5. **Missing validation checks**: No comparison of train vs val for overfitting detection

### Optimization Plan
1. Add gradual unfreezing: frozen → partial (last 4 layers) → full
2. Use `compute_class_weight('balanced')` instead of fixed weights
3. Add discriminative learning rates (lower for PhoBERT, higher for classifier)
4. Add LayerNorm after concatenation for better fusion
5. Add per-class F1 tracking during training

---

## Notebook 2: PhoBERT_TF-IDF_EndToEnd_FineTuning.ipynb

### Architecture
- PhoBERT (768-dim, UNFROZEN immediately)
- TF-IDF 5000 features → 256-dim projection
- Simple concatenation: 768 + 256 = 1024
- Classifier: 1024 → 256 → 3

### Strengths
- Uses TF-IDF for n-gram information
- Has warmup scheduler
- Uses class weights

### Critical Issues
1. **CRITICAL BUG - SMOTE modality mismatch**:
   - SMOTE is applied to TF-IDF features creating synthetic samples
   - Text resampling is done RANDOMLY and INDEPENDENT
   - This creates a mismatch: TF-IDF features don't correspond to the text
   - Example: SMOTE creates TF-IDF vector that is blend of sample A and B's TF-IDF, but text is randomly copied from either A or B

2. **Wrong selection metric**: Uses `val_f1` (weighted) instead of macro-F1
   - Weighted-F1 is dominated by majority classes
   - Model optimizes for accuracy, not Neutral class performance

3. **No gradual unfreezing**: Immediate full fine-tuning

4. **No discriminative learning rates**: 10x LR difference but same for all PhoBERT layers

### Optimization Plan
1. **Remove SMOTE or fix modality alignment**:
   - Option A: Remove SMOTE, use WeightedRandomSampler instead
   - Option B: Apply synchronized oversampling (duplicate entire samples)
   - Option C: Use Focal Loss for class imbalance

2. Change selection metric to macro-F1

3. Add gradual unfreezing

4. Add per-class metrics tracking

---

## Notebook 3: PhoBERT_TF-IDF_Sentiwordnet_Baseline.ipynb

### Architecture
- PhoBERT (768-dim) with gradual unfreezing
- TF-IDF 5000 features → 256-dim projection
- SentiWordNet 35 features → 64-dim projection
- Simple concatenation: 768 + 256 + 64 = 1088
- Classifier: 1088 → 256 → 3

### Strengths
- **Best architecture**: Combines all three feature types
- **Gradual unfreezing**: frozen → partial → full
- **Discriminative learning rates**: Different LR for PhoBERT vs head
- Uses macro-F1 for model selection
- Has class weights computed from data
- Good warmup and scheduler

### Critical Issues
1. **Simple concatenation fusion**: No learned weighting between modalities
2. **Missing error analysis**: No analysis of Neutral class failures
3. **Could benefit from gated fusion**: Weight features dynamically

### Optimization Plan
1. Add gated fusion mechanism to weight modalities
2. Add LayerNorm after projection layers
3. Add detailed error analysis for Neutral class
4. Consider mean pooling instead of just CLS token
5. Add threshold tuning for Neutral class if helpful

---

## Common Issues Across All Notebooks

1. **No error analysis**: Missing analysis of what's being misclassified
2. **No threshold optimization**: Could improve Neutral recall by adjusting thresholds
3. **Simple fusion**: All use simple concatenation, could benefit from learned fusion
4. **CLS token only**: Not exploring mean pooling or other representations

---

## Priority Fixes by Impact

### High Priority (Critical Bugs)
1. Fix SMOTE modality mismatch in Notebook 2
2. Change to macro-F1 selection in Notebook 2
3. Add gradual unfreezing to Notebooks 1 and 2

### Medium Priority (Quality Improvements)
4. Add discriminative learning rates
5. Improve fusion architecture (LayerNorm, gating)
6. Add error analysis

### Low Priority (Optional Enhancements)
7. Try mean pooling instead of CLS
8. Add threshold tuning
9. Compare Focal Loss vs class weights