# 🚀 Getting Started with Anomaly Detection

## Complete Step-by-Step Guide

This guide will walk you through using the self-supervised anomaly detection framework on your forex data.

---

## ⚡ Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
cd anomaly_detection
pip install -r requirements.txt
```

**Required packages**: PyTorch, NumPy, Pandas, scikit-learn, matplotlib, ta (technical analysis)

### Step 2: Run the Example

```bash
python example.py
```

This will:
- Load sample H4 EUR/USD data
- Train a small model (10 epochs)
- Detect anomalies
- Print results

**Expected output**: Detection of ~2-5% anomalies in test set

---

## 📊 Using Your Own Data

### Data Format

Your CSV should have these columns:
- `open` - Opening price
- `high` - High price
- `low` - Low price
- `close` - Closing price
- `volume` (optional) - Trading volume
- `time` or `date` (optional) - Timestamp

**Example**:
```csv
time,open,high,low,close,volume
1420070400,1.21045,1.21320,1.20890,1.21150,12345
1420074000,1.21150,1.21400,1.21100,1.21280,15678
...
```

### Training on Your Data

```bash
python main.py \
    --data_path /path/to/your/data.csv \
    --window_size 60 \
    --n_epochs 100 \
    --batch_size 32 \
    --n_clusters 10 \
    --mode train
```

**Parameters explained**:
- `--data_path`: Path to your CSV file
- `--window_size`: How many time steps in each sequence (60 = 60 hours for H1 data)
- `--n_epochs`: Number of training epochs (100 is good for most datasets)
- `--batch_size`: Samples per batch (32 works for most GPUs)
- `--n_clusters`: Number of market regimes to discover (10 is default)
- `--mode`: train/test/full

---

## 🎯 Complete Workflow

### Phase 1: Training

```bash
python main.py \
    --data_path ../forexPredictor/H4_EURUSD_2015.csv \
    --window_size 60 \
    --n_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --n_clusters 10 \
    --mode train \
    --save_dir my_experiment
```

**What happens**:
1. Loads and preprocesses data
2. Generates 30+ technical indicators
3. Creates sliding windows
4. Trains transformer encoder (self-supervised)
5. Performs clustering to find market regimes
6. Trains anomaly detectors
7. Saves model to `my_experiment/checkpoints/`

**Training time**: 
- CPU: ~2-3 hours for 100K samples, 100 epochs
- GPU (CUDA): ~20-30 minutes

**Output files**:
- `my_experiment/checkpoints/best_model.pt` - Best model during training
- `my_experiment/checkpoints/final_model.pt` - Final trained model
- `my_experiment/visualizations/training_history.png` - Loss curves

---

### Phase 2: Testing

```bash
python main.py \
    --data_path ../forexPredictor/H4_GBPUSD_2015.csv \
    --checkpoint my_experiment/checkpoints/final_model.pt \
    --mode test \
    --save_dir my_experiment
```

**What happens**:
1. Loads pre-trained model
2. Processes test data
3. Extracts embeddings
4. Assigns cluster labels
5. Computes anomaly scores
6. Detects anomalies
7. Generates visualizations

**Output files**:
- `my_experiment/anomaly_results.csv` - Detected anomalies with scores
- `my_experiment/visualizations/anomaly_scores.png` - Score distribution
- `my_experiment/visualizations/anomalies_on_prices.png` - Anomalies on price chart

---

## 📈 Understanding Results

### Output CSV Format

`anomaly_results.csv`:
```csv
anomaly_score,is_anomaly,cluster_label
0.234,False,2
0.156,False,5
0.892,True,7
0.945,True,-1
```

**Columns**:
- `anomaly_score`: Higher = more anomalous (normalized 0-1)
- `is_anomaly`: True/False based on threshold
- `cluster_label`: Which market regime (-1 = noise in DBSCAN)

### Interpreting Scores

- **Score < 0.5**: Normal behavior (low risk)
- **Score 0.5-0.7**: Borderline (monitor)
- **Score > 0.7**: Anomaly (high risk)

**Typical anomaly rate**: 2-5% of data points

---

## 🔧 Tuning Parameters

### For More Sensitive Detection (detect more anomalies)

```bash
--anomaly_threshold_percentile 90  # Lower threshold (default: 95)
--n_clusters 15                     # More granular regimes
```

### For Less False Positives (stricter detection)

```bash
--anomaly_threshold_percentile 98  # Higher threshold
--n_clusters 5                      # Broader regimes
```

### For Better Accuracy (longer training)

```bash
--n_epochs 200                      # More training
--learning_rate 0.00005             # Lower LR
--d_model 256                       # Larger model
--n_layers 6                        # Deeper transformer
```

### For Faster Training (prototyping)

```bash
--n_epochs 20                       # Fewer epochs
--batch_size 64                     # Larger batches
--window_size 30                    # Shorter sequences
--d_model 64                        # Smaller model
--n_layers 2                        # Shallower transformer
```

---

## 🎓 Advanced Usage

### Full Pipeline (Train + Test)

```bash
python main.py \
    --data_path data.csv \
    --mode full \
    --train_ratio 0.8
```

Automatically splits data into 80% train / 20% test

### Custom Model Architecture

```bash
python main.py \
    --data_path data.csv \
    --d_model 256 \
    --n_heads 16 \
    --n_layers 6 \
    --dim_feedforward 1024 \
    --dropout 0.2 \
    --mode train
```

### Different Clustering Methods

**K-Means (fast, default)**:
```bash
--clustering_method kmeans --n_clusters 10
```

**Gaussian Mixture Models (probabilistic)**:
```bash
--clustering_method gmm --n_clusters 10
```

**DBSCAN (automatic cluster count)**:
```bash
--clustering_method dbscan
```

### Score Fusion Strategies

**Weighted sum (balanced)**:
```bash
--fusion_method weighted_sum
```

**Maximum (conservative)**:
```bash
--fusion_method max
```

**Product (strict)**:
```bash
--fusion_method product
```

---

## 📊 Working with Results

### Load and Analyze Results in Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('outputs/anomaly_results.csv')

# Get anomalies only
anomalies = results[results['is_anomaly'] == True]

print(f"Detected {len(anomalies)} anomalies")
print(f"Anomaly rate: {len(anomalies)/len(results)*100:.2f}%")

# Plot score distribution
plt.hist(results['anomaly_score'], bins=50)
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Score Distribution')
plt.show()
```

### Combine with Original Data

```python
# Load original data
df = pd.read_csv('your_data.csv')

# Assume results correspond to last N rows
test_size = len(results)
df_test = df.iloc[-test_size:].reset_index(drop=True)

# Merge
df_test['anomaly_score'] = results['anomaly_score'].values
df_test['is_anomaly'] = results['is_anomaly'].values

# Get anomalous timestamps
anomalous_times = df_test[df_test['is_anomaly']][['time', 'close', 'anomaly_score']]
print(anomalous_times)
```

### Create Trading Signals

```python
# Example: Buy when anomaly detected (mean reversion)
df_test['signal'] = 0
df_test.loc[df_test['is_anomaly'], 'signal'] = 1  # Buy signal

# Or: Sell when anomaly detected (risk-off)
df_test.loc[df_test['is_anomaly'], 'signal'] = -1  # Sell signal

# High-score anomalies only
high_score_threshold = 0.8
df_test.loc[df_test['anomaly_score'] > high_score_threshold, 'signal'] = 1
```

---

## 🐛 Troubleshooting

### Error: "CUDA out of memory"

**Solution**: Reduce batch size
```bash
--batch_size 16  # or even 8
```

### Error: "No module named 'torch'"

**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

### Error: "File not found"

**Solution**: Use absolute path
```bash
--data_path C:/full/path/to/data.csv
```

### Warning: "Too few anomalies detected"

**Solution**: Lower threshold
```bash
--anomaly_threshold_percentile 90
```

### Warning: "Too many anomalies detected"

**Solution**: Increase threshold or train longer
```bash
--anomaly_threshold_percentile 98
--n_epochs 150
```

---

## 💡 Tips & Best Practices

### Data Preparation

1. **Remove outliers first**: Extreme outliers can affect scaling
2. **Ensure chronological order**: Time series should be sorted by time
3. **Handle missing values**: Fill gaps or remove incomplete periods
4. **Consistent timeframe**: Don't mix H1 and H4 data

### Model Training

1. **Start small**: Test with 20 epochs first, then increase
2. **Monitor loss**: Should decrease steadily (check visualizations)
3. **Use validation**: Split data chronologically, not randomly
4. **Save checkpoints**: Training can be resumed if interrupted

### Anomaly Detection

1. **Context matters**: Anomaly in one regime might be normal in another
2. **Combine with fundamentals**: Verify detected anomalies with news
3. **Backtesting**: Test how anomaly signals would have performed historically
4. **Threshold tuning**: Adjust based on your risk tolerance

### Performance

1. **Use GPU**: 10-20x faster training with CUDA
2. **Batch size**: Larger = faster, but needs more memory
3. **Sequence length**: Shorter = faster, but less context
4. **Feature selection**: Can remove less important indicators

---

## 📚 Next Steps

### 1. Experiment with Different Assets

Try different currency pairs, timeframes:
- H1 EUR/USD (high frequency)
- H4 GBP/USD (medium frequency)
- D1 USD/JPY (daily)

### 2. Ablation Studies

Test individual components:
- Contrastive learning only: `--reconstruction_weight 0`
- Reconstruction only: `--contrastive_weight 0`
- Energy-based only
- Reconstruction-based only

### 3. Hyperparameter Search

Systematically test different:
- Window sizes: 30, 60, 120
- Cluster counts: 5, 10, 15, 20
- Model dimensions: 64, 128, 256

### 4. Integration with Trading System

- Connect to MT5 for real-time detection
- Build alerting system for high-score anomalies
- Backtest anomaly-based strategies

### 5. Research Extensions

- Multi-asset anomaly detection
- Cross-market anomaly propagation
- Anomaly severity classification
- Anomaly cause attribution

---

## 📞 Getting Help

### Check Documentation

1. `README.md` - Comprehensive framework overview
2. `IMPLEMENTATION_SUMMARY.md` - Technical details
3. Code comments - Inline documentation

### Common Issues

See "Troubleshooting" section above

### Run Interactive Guide

```bash
python quickstart.py
```

---

## 🎯 Quick Reference Card

### Training
```bash
python main.py --data_path DATA.csv --mode train
```

### Testing
```bash
python main.py --data_path DATA.csv --checkpoint MODEL.pt --mode test
```

### Full Pipeline
```bash
python main.py --data_path DATA.csv --mode full
```

### Example
```bash
python example.py
```

### Help
```bash
python main.py --help
python quickstart.py
```

---

**Happy anomaly detecting! 🚀📈**

For questions or issues, refer to the README.md or examine the example.py code.

