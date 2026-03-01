# Self-Supervised Temporal Anomaly Detection Framework

## 📋 Overview

A comprehensive self-supervised learning framework for detecting anomalies in financial time-series data. This implementation is based on the research paper abstract focusing on **temporal representation learning, contrastive learning, masked reconstruction, and energy-based anomaly scoring**.

## 🎯 Key Features

### 1. **Transformer-Based Temporal Encoder**
- Multi-head self-attention mechanism
- Positional encoding for temporal sequences
- Captures long-range dependencies in market behavior
- Context-aware embeddings for regime transitions

### 2. **Self-Supervised Learning Objectives**
- **Temporal Contrastive Learning**: NT-Xent loss with temporal augmentations
- **Masked Time-Series Reconstruction**: Autoencoder-style reconstruction loss
- Combined multi-task learning approach

### 3. **Density-Aware Clustering**
- Discovers normal market regimes automatically
- Supports K-Means, GMM, and DBSCAN
- Density-based filtering for regime identification
- Latent space regularization for better separation

### 4. **Hybrid Anomaly Detection**
- **Energy-Based Scoring**: Cluster-conditioned energy functions
- **Reconstruction-Based Detection**: Mahalanobis distance measures
- Score fusion (weighted sum, max, product)
- Adaptive threshold determination

### 5. **Financial Time-Series Processing**
- Automatic technical indicator generation (30+ features)
- Robust scaling and preprocessing
- Sliding window sequence generation
- Compatible with OHLC forex data

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Input Time-Series (OHLC)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Technical Indicator Generation & Scaling             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Transformer Temporal Encoder                    │
│  • Positional Encoding                                       │
│  • Multi-Head Self-Attention (4 layers)                      │
│  • Feed-Forward Networks                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
┌───────────────────────┐  ┌──────────────────────────┐
│  Contrastive Learning │  │ Masked Reconstruction    │
│  • Temporal Augment   │  │ • Random Masking         │
│  • NT-Xent Loss       │  │ • MSE Reconstruction     │
└───────────────────────┘  └──────────────────────────┘
            │                         │
            └────────────┬────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Learned Embeddings                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Density-Aware Clustering                          │
│  • K-Means / GMM / DBSCAN                                    │
│  • Normal Regime Identification                              │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
┌───────────────────────┐  ┌──────────────────────────┐
│  Energy-Based Scoring │  │ Reconstruction Error     │
│  • Cluster-Conditioned│  │ • Mahalanobis Distance   │
│  • Learned Energy Fn  │  │ • Deviation Measures     │
└───────────────────────┘  └──────────────────────────┘
            │                         │
            └────────────┬────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Hybrid Anomaly Scores                           │
│  • Score Fusion                                              │
│  • Adaptive Thresholding                                     │
└────────────────────────┬────────────────────────────────────┘
                         ▼
            🚨 Detected Anomalies 🚨
```

## 📁 Project Structure

```
anomaly_detection/
├── models/
│   ├── temporal_transformer.py    # Transformer encoder & SSL objectives
│   ├── clustering.py               # Density-aware clustering
│   └── anomaly_detector.py         # Energy & reconstruction detectors
├── training/
│   └── trainer.py                  # Training pipeline
├── data/
│   └── preprocessing.py            # Data loading & preprocessing
├── utils/
│   ├── visualization.py            # Plotting functions
│   └── evaluation.py               # Metrics & evaluation
├── main.py                         # Main execution script
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
cd anomaly_detection
pip install -r requirements.txt
```

### Option 1: Quick Training & Testing (Recommended for First Time)

**Train the anomaly detector:**
```bash
python train_anomaly_detector.py
```

This simplified script will:
- Load EURUSD H4 data
- Train for 10 epochs (fast demo)
- Test on validation set
- Save model to `example_outputs/checkpoints/`

**Validate with synthetic anomalies:**
```bash
python validate_anomaly_detector.py
```

This will:
- Inject synthetic anomalies
- Test detection performance
- Show precision, recall, F1 score
- Save report to `validation_outputs/`

**Test reconstruction-only detection:**
```bash
python test_reconstruction_only.py
```

### Option 2: Full Training with Custom Parameters

**1. Train the Model**

```bash
python main.py \
    --data_path ../forexPredictor/H4_EURUSD_2015.csv \
    --window_size 60 \
    --n_epochs 100 \
    --batch_size 32 \
    --n_clusters 10 \
    --mode train
```

**2. Test on New Data**

```bash
python main.py \
    --data_path ../forexPredictor/H4_EURUSD_2015.csv \
    --checkpoint outputs/checkpoints/final_model.pt \
    --mode test
```

**3. Full Pipeline (Train + Test)**

```bash
python main.py \
    --data_path ../forexPredictor/H4_EURUSD_2015.csv \
    --window_size 60 \
    --n_epochs 100 \
    --mode full
```

## 📊 Command Line Arguments

### Data Parameters
- `--data_path`: Path to CSV file with OHLC data (required)
- `--window_size`: Sequence length (default: 60)
- `--stride`: Sliding window stride (default: 1)
- `--train_ratio`: Train/test split ratio (default: 0.8)

### Model Parameters
- `--d_model`: Transformer dimension (default: 128)
- `--n_heads`: Attention heads (default: 8)
- `--n_layers`: Transformer layers (default: 4)
- `--dropout`: Dropout rate (default: 0.1)

### Training Parameters
- `--n_epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--contrastive_weight`: Contrastive loss weight (default: 1.0)
- `--reconstruction_weight`: Reconstruction loss weight (default: 1.0)

### Clustering Parameters
- `--n_clusters`: Number of clusters (default: 10)
- `--clustering_method`: kmeans/gmm/dbscan (default: kmeans)

### Anomaly Detection Parameters
- `--anomaly_threshold_percentile`: Threshold percentile (default: 95)
- `--fusion_method`: weighted_sum/max/product (default: weighted_sum)

## 📈 Output

The framework generates:

1. **Checkpoints**: Trained model weights
   - `outputs/checkpoints/best_model.pt`
   - `outputs/checkpoints/final_model.pt`

2. **Visualizations**:
   - Training history curves
   - Embedding visualizations (t-SNE)
   - Anomaly score distributions
   - Detected anomalies on price charts

3. **Results**:
   - `outputs/anomaly_results.csv`: Anomaly scores and labels
   - Statistics and metrics

## 🔬 Technical Details

### Temporal Contrastive Learning
- Creates augmented views through temporal masking and noise injection
- Uses InfoNCE loss (NT-Xent) with temperature scaling
- Learns invariant representations across augmentations

### Masked Reconstruction
- Randomly masks 15% of time steps
- Reconstructs masked segments using transformer encoder
- Learns to capture temporal dependencies

### Energy-Based Detection
- Trains energy function to assign low energy to normal samples
- Cluster-conditioned normalization for better calibration
- High energy indicates out-of-distribution samples

### Reconstruction-Based Detection
- Computes reconstruction error using Mahalanobis distance
- Fits distribution on normal training data
- Large deviations indicate anomalies

## 📊 Evaluation Metrics

- **Anomaly Statistics**: Count, rate, score distribution
- **Clustering Quality**: Silhouette score, density distribution
- **Detection Performance**: Precision, recall, F1, AUROC (if labels available)
- **Financial Impact**: Returns, Sharpe ratio, win rate

## 🎯 Use Cases

1. **Abnormal Price Movements**: Flash crashes, sudden spikes
2. **Volatility Spikes**: Market regime changes
3. **Market Microstructure Anomalies**: Unusual trading patterns
4. **Risk Management**: Early warning system
5. **Trading Strategy**: Anomaly-based signal generation

## 🔧 Customization

### Add Custom Technical Indicators

Edit `data/preprocessing.py`:

```python
def add_features(self, df):
    # Add your custom indicators
    df['custom_indicator'] = ...
    return df
```

### Modify Transformer Architecture

Edit `models/temporal_transformer.py`:

```python
class TemporalTransformerEncoder(nn.Module):
    def __init__(self, ...):
        # Customize layers, dimensions, etc.
        ...
```

### Implement New Anomaly Scoring

Edit `models/anomaly_detector.py`:

```python
class CustomAnomalyDetector:
    def compute_score(self, embeddings):
        # Your scoring logic
        ...
```

## 📚 References

This implementation is based on concepts from:

- **Transformers**: "Attention is All You Need" (Vaswani et al., 2017)
- **Contrastive Learning**: "A Simple Framework for Contrastive Learning" (Chen et al., 2020)
- **Masked Autoencoders**: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)
- **Energy-Based Models**: "Your Classifier is Secretly an Energy Based Model" (Grathwohl et al., 2019)

## ⚠️ Disclaimer

This framework is for **research and educational purposes only**. Detected anomalies should be validated before making trading decisions. No financial advice is provided.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional SSL objectives (MoCo, BYOL, SwAV)
- More sophisticated clustering algorithms
- Online/streaming anomaly detection
- Multi-scale temporal modeling
- Integration with existing trading systems

## 📝 License

Same as parent project - Educational use only.

---

**Author**: Implementation based on research abstract for thesis project
**Framework**: PyTorch-based self-supervised learning for financial anomaly detection

