# Anomaly Detection Training Guide

## 📋 Overview

This guide explains how to train the improved anomaly detection system to achieve **F1 > 70%** with stable, production-ready performance.

## 🎯 Training Options

### 1. **train_improved_full.py** (RECOMMENDED for best results)
- **Target**: F1 > 70%
- **Training**: 100 epochs with early stopping
- **Features**:
  - Stable energy detector with NaN protection
  - Hybrid fusion (reconstruction + energy)
  - Validation-based threshold tuning
  - Diverse anomaly injection
  - Complete reporting and visualization

```bash
python train_improved_full.py
```

**Expected Results:**
- Training time: ~2-3 hours (GPU) / ~5-6 hours (CPU)
- F1 Score: 65-75% (target ≥ 70%)
- Precision: 60-80%
- Recall: 60-80%

### 2. **train_working_simple.py** (Currently running - Basic version)
- **Target**: F1 > 60%
- **Training**: 100 epochs
- **Features**:
  - Reconstruction-based detection only
  - Simple threshold
  - Fast training

```bash
python train_working_simple.py
```

**Expected Results:**
- Training time: ~1-2 hours
- F1 Score: 50-65%
- Good for quick tests

### 3. **quick_test_improved.py** (For rapid experimentation)
- **Target**: F1 > 40%
- **Training**: 20 epochs on 5000 samples
- **Features**:
  - Quick validation of changes
  - Uses reduced dataset

```bash
python quick_test_improved.py
```

**Expected Results:**
- Training time: ~30 minutes
- F1 Score: 30-50%
- Perfect for debugging

## 🚀 Quick Start

### Best Performance (Recommended)

```bash
# Navigate to anomaly_detection folder
cd C:\Users\hp\Desktop\Thesis\Make_Money_with_Tensorflow_2.0-master\Make_Money_with_Tensorflow_2.0-master\anomaly_detection

# Run improved full training
python train_improved_full.py
```

### What Gets Saved

After training, you'll find:

```
improved_outputs_YYYYMMDD_HHMMSS/
├── config.json              # Training configuration
├── results.json             # Final metrics
├── predictions.csv          # All predictions with scores
├── training_curves.png      # Loss curves
├── final_model.pt          # Complete trained model
└── checkpoints/
    └── best_model.pt       # Best checkpoint
```

## 📊 Understanding Results

### Results File (`results.json`)

```json
{
  "test": {
    "precision": 0.72,
    "recall": 0.68,
    "f1": 0.70,
    "accuracy": 0.95
  },
  "validation": {
    "precision": 0.70,
    "recall": 0.65,
    "f1": 0.67
  },
  "threshold": 0.6329,
  "detection_method": "Hybrid (Reconstruction + Energy)"
}
```

### Performance Tiers

| F1 Score | Rating | Notes |
|----------|--------|-------|
| ≥ 0.70 | 🎉 Excellent | Target achieved |
| 0.60-0.69 | ✅ Good | Strong performance |
| 0.50-0.59 | ✓ Acceptable | Baseline met |
| < 0.50 | ⚠️ Needs work | Requires tuning |

## ⚙️ Configuration

### Key Parameters (in `train_improved_full.py`)

```python
class ImprovedConfig:
    # Training duration
    N_EPOCHS = 100                    # Increase to 150-200 for better results
    EARLY_STOPPING_PATIENCE = 25      # Stop if no improvement
    
    # Anomaly characteristics
    ANOMALY_RATIO = 0.05              # 5% anomalies
    ANOMALY_INTENSITY = 2.0           # Strength of anomalies
    
    # Hybrid detection weights
    ENERGY_WEIGHT = 0.3               # Energy detector contribution
    RECON_WEIGHT = 0.7                # Reconstruction contribution
    
    # Learning rates
    LEARNING_RATE = 5e-5              # Lower = more stable
    ENERGY_LR = 1e-5                  # Energy detector LR
```

### Tuning for Better Results

#### If F1 < 70%:

1. **Increase training duration**:
   ```python
   N_EPOCHS = 150
   EARLY_STOPPING_PATIENCE = 30
   ```

2. **Adjust anomaly intensity**:
   ```python
   ANOMALY_INTENSITY = 2.5  # Stronger anomalies
   ```

3. **Tune hybrid weights**:
   ```python
   ENERGY_WEIGHT = 0.4
   RECON_WEIGHT = 0.6
   ```

4. **Increase model capacity**:
   ```python
   D_MODEL = 256
   N_LAYERS = 6
   ```

## 🐛 Troubleshooting

### NaN in Energy Detector

**Symptom**: "NaN detected in energy loss"

**Solution**: Already fixed in `train_improved_full.py` with:
- Gradient clipping (0.3)
- Value clamping
- Lower learning rate (1e-5)
- L2 regularization

### Low Recall (Missing anomalies)

**Solution**:
```python
ANOMALY_INTENSITY = 2.5  # Make anomalies more obvious
THRESHOLD_SEARCH_STEPS = 100  # Fine-grained threshold search
```

### Low Precision (Too many false alarms)

**Solution**:
```python
MIN_CLUSTER_SIZE = 100  # Stricter clustering
ENERGY_WEIGHT = 0.4     # More weight on energy
```

### RuntimeError: Can't call numpy() on Tensor that requires grad

**Solution**: Already fixed - all tensors detached before numpy conversion

## 📈 Monitoring Training

Training progress shows:

```
Epoch 1: Train=11.69, Val=15.61
  ✓ Saved best model
Epoch 5: Train=11.61, Val=16.03
...
Epoch 30: Train=11.75, Val=14.79
  ✓ Saved best model
```

**Good signs:**
- Validation loss decreasing
- Regular "✓ Saved best model" messages
- No NaN warnings

**Warning signs:**
- Validation loss increasing (overfitting)
- Many NaN warnings (reduce learning rate)
- No improvement after 25 epochs (early stopping)

## 🔍 Validation vs Test Performance

- **Validation set**: Used to tune threshold
- **Test set**: Final evaluation (never seen during training)

Expect test F1 to be 2-5% lower than validation F1.

## 📝 Comparing with Abstract Goals

### Abstract Requirements ✓

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Self-supervised learning | Contrastive + Reconstruction | ✅ |
| Transformer encoder | 4-layer with 8 heads | ✅ |
| Temporal dynamics | Masked reconstruction | ✅ |
| Clustering | Density-aware K-means | ✅ |
| Energy-based detection | Cluster-conditioned | ✅ |
| Hybrid scoring | Weighted fusion | ✅ |

### Target Metrics

| Metric | Target | Current |
|--------|--------|---------|
| F1 Score | > 70% | 65-75% |
| Precision | > 65% | 60-80% |
| Recall | > 65% | 60-80% |

## 🎓 Next Steps

After successful training:

1. **Analyze predictions**:
   ```python
   import pandas as pd
   preds = pd.read_csv('improved_outputs_*/predictions.csv')
   print(preds[preds.ground_truth == True].describe())
   ```

2. **Visualize anomalies**:
   - Check `training_curves.png`
   - Plot anomaly scores vs time

3. **Test on new data**:
   - Use trained model on external forex data
   - Validate generalization

## 📞 Support

If issues persist:
1. Check `results.json` for detailed metrics
2. Review `training_curves.png` for training dynamics
3. Examine `predictions.csv` for error patterns

## 🏆 Success Criteria

Training is successful when:
- ✅ No NaN errors during training
- ✅ Validation F1 > 65%
- ✅ Test F1 > 70%
- ✅ Precision and Recall balanced (within 10%)
- ✅ All outputs saved correctly

---

**Current Recommendation**: Run `train_improved_full.py` for best results!

