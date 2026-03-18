# Notebook Optimization Report

## Executive Summary

This report documents the analysis and optimization of three Vietnamese sentiment classification notebooks. All notebooks have been optimized with critical fixes and architectural improvements.

---

## 1. Notebook-by-Notebook Summary

### Notebook A: PhoBERT_Sentiwordnet_Improved.ipynb

**Architecture:**
- PhoBERT (768-dim, previously immediately unfrozen)
- SentiWordNet 35 extended features → 64-dim projection
- Simple concatenation fusion: 768 + 64 = 832
- Classifier: 832 → 256 → 3

**Original Issues:**
1. ❌ No gradual unfreezing - PhoBERT immediately trainable from epoch 1
2. ❌ Fixed class weights [1.0, 5.0, 1.0] - arbitrary, not computed from data
3. ❌ No discriminative learning rates
4. ❌ Simple concatenation fusion without normalization
5. ❌ Missing Neutral class F1 tracking during training

**Changes Applied (→ PhoBERT_Sentiwordnet_Optimized.ipynb):**
1. ✅ Added gradual unfreezing: frozen (epoch 1) → partial (epoch 2-3) → full (epoch 4+)
2. ✅ Use `compute_class_weight('balanced')` from sklearn
3. ✅ Discriminative learning rates: PhoBERT LR varies by stage, head LR = 5e-5
4. ✅ Added LayerNorm after projection and fusion layers
5. ✅ Track per-class F1 including Neutral class

**Original Results:**
- Test Accuracy: 0.9274
- Test F1 Macro: 0.8136
- Neutral F1: 0.5466

---

### Notebook B: PhoBERT_TF-IDF_EndToEnd_FineTuning.ipynb

**Architecture:**
- PhoBERT (768-dim, immediately unfrozen)
- TF-IDF 5000 features → 256-dim projection
- Simple concatenation: 768 + 256 = 1024
- Classifier: 1024 → 256 → 3

**Critical Issues:**
1. ❌ **CRITICAL BUG: SMOTE modality mismatch**
   - SMOTE applied to TF-IDF features creates synthetic feature vectors
   - Text resampling is done RANDOMLY and INDEPENDENTLY
   - Result: TF-IDF features don't correspond to the actual text
   - Example: A synthetic TF-IDF vector (blend of samples A and B) paired with text from sample A

2. ❌ Wrong selection metric: Uses weighted-F1 instead of macro-F1
   - Weighted-F1 is dominated by majority classes (Negative/Positive)
   - Model optimizes for overall accuracy, ignores Neutral class

3. ❌ No gradual unfreezing

4. ❌ No discriminative learning rates

**Changes Applied (→ PhoBERT_TF-IDF_EndToEnd_FineTuning_Optimized.ipynb):**
1. ✅ **Replaced SMOTE with WeightedRandomSampler**
   - Each sample stays intact (text + TF-IDF from same sample)
   - Oversampling happens at the sample level, not feature level

2. ✅ Changed selection metric to macro-F1

3. ✅ Added gradual unfreezing schedule

4. ✅ Added discriminative learning rates

5. ✅ Added LayerNorm after fusion

6. ✅ Added per-class F1 tracking and error analysis

**Original Results:**
- Test Accuracy: 0.9311
- Test F1 Weighted: 0.9291 (wrong metric!)
- Neutral F1: 0.5753

---

### Notebook C: PhoBERT_TF-IDF_Sentiwordnet_Baseline.ipynb

**Architecture:**
- PhoBERT (768-dim) with gradual unfreezing ✅
- TF-IDF 5000 features → 256-dim projection
- SentiWordNet 35 features → 64-dim projection
- Simple concatenation: 768 + 256 + 64 = 1088
- Classifier: 1088 → 256 → 3

**Strengths (already correct):**
- ✅ Uses macro-F1 for model selection
- ✅ Has gradual unfreezing
- ✅ Has discriminative learning rates
- ✅ Has computed class weights

**Issues:**
1. ❌ Simple concatenation fusion - no learned interaction between modalities
2. ❌ Missing detailed error analysis
3. ❌ Could benefit from better fusion architecture

**Changes Applied (→ PhoBERT_TF-IDF_Sentiwordnet_Optimized.ipynb):**
1. ✅ Added Gated Fusion mechanism
   - Learnable weights for each modality
   - Dynamic weighting based on input

2. ✅ Added LayerNorm after projections

3. ✅ Added comprehensive error analysis for Neutral class

4. ✅ Added gate weight visualization during training

**Original Results:**
- Test Accuracy: 0.9274
- Test F1 Macro: 0.8183
- Neutral F1: 0.5631

---

## 2. Critical Issues Found

### Issue 1: SMOTE Modality Mismatch (Notebook B)
**Severity: CRITICAL**

The original notebook applies SMOTE to TF-IDF features, creating synthetic feature vectors by interpolating between real samples. However, the text resampling is done independently and randomly.

**Impact:**
- Training samples have TF-IDF features that don't correspond to the text
- Model learns incorrect associations
- May explain poor Neutral class performance

**Fix:** Use WeightedRandomSampler instead, which keeps each sample intact.

### Issue 2: Wrong Selection Metric (Notebook B)
**Severity: HIGH**

Using weighted-F1 for model selection optimizes for majority classes. With Neutral being only 4% of data, weighted-F1 essentially ignores Neutral performance.

**Fix:** Changed to macro-F1 which gives equal weight to all classes.

### Issue 3: Missing Gradual Unfreezing (Notebooks A, B)
**Severity: MEDIUM**

Immediately unfreezing all PhoBERT layers risks catastrophic forgetting of pre-trained representations.

**Fix:** Added three-stage unfreezing:
- Stage 1 (frozen): Only train projection + classifier
- Stage 2 (partial): Unfreeze last 4 encoder layers
- Stage 3 (full): Unfreeze all PhoBERT layers

### Issue 4: Simple Concatenation Fusion (All Notebooks)
**Severity: MEDIUM**

All notebooks use simple concatenation, which doesn't capture interactions between modalities.

**Fix:** Added LayerNorm after concatenation, and for Notebook C, added Gated Fusion mechanism.

### Issue 5: Fixed Class Weights (Notebook A)
**Severity: MEDIUM**

Arbitrary weights [1.0, 5.0, 1.0] instead of computed from data distribution.

**Fix:** Use `compute_class_weight('balanced')` for data-driven weights.

---

## 3. Changes Applied Summary

| Change | Notebook A | Notebook B | Notebook C |
|--------|-----------|-----------|-----------|
| Fixed SMOTE mismatch | N/A | ✅ | N/A |
| Macro-F1 selection | Already correct | ✅ Fixed | Already correct |
| Gradual unfreezing | ✅ Added | ✅ Added | Already correct |
| Discriminative LR | ✅ Added | ✅ Added | Already correct |
| Computed class weights | ✅ Fixed | Already correct | Already correct |
| LayerNorm fusion | ✅ Added | ✅ Added | ✅ Added |
| Gated Fusion | N/A | N/A | ✅ Added |
| Error Analysis | ✅ Added | ✅ Added | ✅ Enhanced |

---

## 4. Expected Results Before vs After

### Notebook A: PhoBERT_Sentiwordnet

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Test Accuracy | 0.9274 | ~0.93-0.94 |
| F1 Macro | 0.8136 | ~0.82-0.84 |
| Neutral F1 | 0.5466 | ~0.56-0.60 |

### Notebook B: PhoBERT_TF-IDF End-to-End

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Test Accuracy | 0.9311 | ~0.92-0.93 |
| F1 Macro | ~0.82* | ~0.82-0.84 |
| Neutral F1 | 0.5753 | ~0.58-0.62 |

*Estimated from per-class metrics; original used wrong metric

### Notebook C: PhoBERT_TF-IDF-SentiWordNet

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Test Accuracy | 0.9274 | ~0.93-0.94 |
| F1 Macro | 0.8183 | ~0.83-0.85 |
| Neutral F1 | 0.5631 | ~0.58-0.63 |

---

## 5. Best Notebook After Optimization

### Recommendation: PhoBERT_TF-IDF_Sentiwordnet_Optimized.ipynb

**Why this should be the primary base model:**

1. **Most comprehensive features**: Combines all three modalities (PhoBERT, TF-IDF, SentiWordNet)

2. **Best architecture**: Gated Fusion allows the model to learn optimal weighting of each modality

3. **Correct training setup**:
   - Macro-F1 selection
   - Gradual unfreezing
   - Discriminative learning rates
   - Computed class weights

4. **Best extensibility**:
   - Easy to add more features
   - Gated fusion can be extended to more modalities
   - Architecture supports ablation studies

5. **Comprehensive evaluation**:
   - Error analysis for Neutral class
   - Gate weight visualization
   - Per-class metrics tracking

**Alternative**: PhoBERT_Sentiwordnet_Optimized.ipynb for simpler, faster training with fewer features but still strong performance.

---

## 6. Recommended Next Steps

### Priority 1: Data-Level Improvements
1. **Augment Neutral class data**: Collect more Neutral samples or use careful oversampling
2. **Analyze Neutral class errors**: Identify linguistic patterns that confuse the model
3. **Improve teencode normalization**: Add more mappings to the TEENCODE_DICT

### Priority 2: Model-Level Improvements
1. **Try Focal Loss**: Compare with class-weighted CrossEntropy
2. **Add threshold tuning**: Optimize prediction thresholds for each class separately
3. **Experiment with mean pooling**: Compare with CLS token extraction

### Priority 3: Feature Engineering
1. **Expand SentiWordNet**: Add more Vietnamese sentiment lexicons
2. **Add negation handling**: Better detection of negation patterns
3. **Add punctuation features**: Exclamation marks, question marks can indicate sentiment

### Priority 4: Ensemble Methods
1. **Late fusion ensemble**: Combine predictions from multiple model configurations
2. **Cross-validation**: More robust evaluation with stratified k-fold

### Priority 5: Deployment Considerations
1. **Model quantization**: Reduce model size for faster inference
2. **Export to ONNX**: For cross-platform deployment
3. **Build prediction API**: Create REST API for inference

---

## Files Created

1. `notebook/PhoBERT_Sentiwordnet_Optimized.ipynb` - Optimized version of Notebook A
2. `notebook/PhoBERT_TF-IDF_EndToEnd_FineTuning_Optimized.ipynb` - Fixed version of Notebook B
3. `notebook/PhoBERT_TF-IDF_Sentiwordnet_Optimized.ipynb` - Enhanced version of Notebook C with Gated Fusion
4. `docs/notebook_analysis.md` - Detailed analysis document
5. `docs/optimization_report.md` - This report

---

## Conclusion

The optimizations address critical bugs (especially the SMOTE modality mismatch) and improve the overall architecture with gradual unfreezing, discriminative learning rates, and better fusion mechanisms. The recommended primary model is the PhoBERT_TF-IDF_SentiWordNet with Gated Fusion, which offers the best combination of feature richness, architectural sophistication, and extensibility.

Run the optimized notebooks on GPU (Google Colab recommended) to validate the expected improvements.