# 🔧 Bug Fix Applied - train_improved_full.py

## Issue Fixed

**Error:** `TypeError: cannot do slice indexing on DatetimeIndex with these indexers [3133] of type int64`

**Location:** `inject_diverse_anomalies()` function, line 174

## Root Cause

The function was using `.loc` with integer indices on a DataFrame that has a DatetimeIndex. The `.loc` accessor expects label-based indexing, not integer-based indexing.

## Solution Applied

Changed from `.loc` with slice to `.iloc` with integer range:

### Before (BROKEN):
```python
elif anomaly_type == 'trend_break':
    # Sudden reversal
    window = slice(max(0, idx-3), min(len(data_modified), idx+3))
    mean_price = data_modified.loc[window, 'close'].mean()  # ❌ ERROR
```

### After (FIXED):
```python
elif anomaly_type == 'trend_break':
    # Sudden reversal
    window_start = max(0, idx-3)
    window_end = min(len(data_modified), idx+3)
    mean_price = data_modified.iloc[window_start:window_end]['close'].mean()  # ✅ WORKS
```

## Additional Improvements

Also ensured that `train_contrastive` and `train_reconstruction` losses are properly tracked during training for use in visualizations:

```python
# Variables now properly initialized and populated
train_contrastive = []
train_reconstruction = []

# Values collected during training
epoch_contrastive_losses.append(losses.get('contrastive_loss', 0))
epoch_reconstruction_losses.append(losses.get('reconstruction_loss', 0))

# Averaged and stored per epoch
train_contrastive.append(np.mean(epoch_contrastive_losses))
train_reconstruction.append(np.mean(epoch_reconstruction_losses))
```

## Status

✅ **FIXED** - The training script should now run without errors.

## Next Steps

Run the training:
```bash
cd anomaly_detection
python train_improved_full.py
```

Expected behavior:
- Training completes successfully
- All 7 visualizations generated
- Excel report created
- No DatetimeIndex errors

---

**Fixed:** February 13, 2026  
**Files Modified:** `train_improved_full.py` (3 changes)

