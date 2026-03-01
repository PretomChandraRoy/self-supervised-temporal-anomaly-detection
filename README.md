# Self-Supervised Temporal Anomaly Detection in Financial Time-Series

A PyTorch implementation of self-supervised temporal representation learning for anomaly detection in financial time-series data. Developed as part of a thesis project on detecting abnormal market behavior using transformer-based encoders, contrastive learning, and energy-based scoring.

---

## Overview

This framework learns normal market behavior patterns from unlabeled forex data using self-supervised objectives, then detects anomalies as deviations from those learned patterns. The system combines three complementary detection strategies in a hybrid fusion approach.

### Key Components

| Component | Description |
|-----------|-------------|
| **Temporal Transformer Encoder** | Multi-head self-attention with positional encoding for capturing long-range dependencies |
| **Self-Supervised Learning** | Temporal contrastive learning (NT-Xent) + masked time-series reconstruction |
| **Density-Aware Clustering** | Automatic discovery of normal market regimes via K-Means with density filtering |
| **Energy-Based Scoring** | Cluster-conditioned energy functions that assign high energy to out-of-distribution samples |
| **Reconstruction-Based Detection** | Reconstruction error measurement for identifying anomalous patterns |
| **Hybrid Fusion** | Weighted combination of reconstruction, cluster, and energy scores with adaptive thresholding |

---

## Architecture

```
Input Time-Series (OHLC + Technical Indicators)
        │
        ▼
┌──────────────────────────────────────┐
│  Financial Data Preprocessing        │
│  • Technical indicators (25 features)│
│  • Robust scaling & outlier clipping │
│  • Sliding window sequences (60-step)│
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  Transformer Temporal Encoder        │
│  • Positional encoding               │
│  • Multi-head self-attention (5 layers, 8 heads) │
│  • Feed-forward networks             │
└──────────────┬───────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐ ┌──────────────┐
│ Contrastive │ │   Masked     │
│  Learning   │ │Reconstruction│
│ (NT-Xent)   │ │   (MSE)      │
└──────┬──────┘ └──────┬───────┘
       └───────┬───────┘
               ▼
┌──────────────────────────────────────┐
│  Learned Temporal Embeddings         │
└──────────────┬───────────────────────┘
               │
       ┌───────┼───────────┐
       ▼       ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│Density-  │ │Energy-   │ │Reconstruction│
│Aware     │ │Based     │ │Error         │
│Clustering│ │Scoring   │ │Detection     │
└────┬─────┘ └────┬─────┘ └──────┬───────┘
     └─────────┬──┴──────────────┘
               ▼
┌──────────────────────────────────────┐
│  Hybrid Anomaly Score Fusion         │
│  • Weighted sum + max aggregation    │
│  • Validation-based threshold tuning │
└──────────────┬───────────────────────┘
               ▼
        Detected Anomalies
```

---

## Project Structure

```
anomaly_detection/
├── train_improved_full.py          # Main training & evaluation pipeline
├── generate_detailed_excel.py      # Excel report generator (9-sheet report)
├── requirements.txt                # Python dependencies
├── __init__.py
│
├── models/
│   ├── temporal_transformer.py     # Transformer encoder & SSL objectives
│   ├── clustering.py               # Density-aware clustering
│   └── anomaly_detector.py         # Energy & reconstruction detectors
│
├── data/
│   ├── preprocessing.py            # Data loading, feature engineering, scaling
│   └── H4_EURUSD_2015.csv         # EUR/USD H4 forex dataset (2015-2024)
│
├── training/
│   └── trainer.py                  # Training pipeline utilities
│
├── validation/
│   └── synthetic_anomalies.py      # Synthetic anomaly injection for evaluation
│
└── utils/
    ├── evaluation.py               # Metrics & evaluation functions
    └── visualization.py            # Plotting utilities
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd anomaly_detection
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.6.0, NumPy, pandas, scikit-learn, matplotlib, seaborn

### 2. Run Training

```bash
python train_improved_full.py
```

This single command runs the complete pipeline:
1. Loads and preprocesses EUR/USD H4 forex data
2. Injects synthetic anomalies for supervised evaluation
3. Trains the transformer encoder with contrastive + reconstruction objectives
4. Performs density-aware clustering on learned embeddings
5. Trains the energy-based anomaly detector
6. Tunes detection threshold on validation set
7. Evaluates on held-out test set
8. Generates thesis-ready visualizations and Excel report

### 3. View Results

After training completes, results are saved in a timestamped folder:

```
improved_outputs_YYYYMMDD_HHMMSS/
├── checkpoints/
│   └── best_model.pt              # Best model weights
├── config.json                     # Training configuration
├── results.json                    # All metrics (JSON)
├── predictions.csv                 # Per-sample predictions
├── training_curves.png             # Loss curves
├── final_model.pt                  # Final model + detectors
├── DETAILED_RESULTS.xlsx           # 9-sheet Excel report
└── thesis_figures/                 # Publication-ready figures (300 DPI)
    ├── 1_training_curves.png
    ├── 2_confusion_matrix.png
    ├── 3_performance_metrics.png
    ├── 4_anomaly_score_distribution.png
    ├── 5_precision_recall_curve.png
    ├── 6_detection_timeline.png
    └── 7_results_dashboard.png
```

---

## Configuration

Key hyperparameters can be modified in the `ImprovedConfig` class inside `train_improved_full.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `D_MODEL` | 192 | Transformer embedding dimension |
| `N_HEADS` | 8 | Number of attention heads |
| `N_LAYERS` | 5 | Number of transformer layers |
| `N_EPOCHS` | 150 | Maximum training epochs |
| `BATCH_SIZE` | 64 | Training batch size |
| `LEARNING_RATE` | 1e-4 | AdamW learning rate |
| `WINDOW_SIZE` | 60 | Input sequence length (time steps) |
| `N_CLUSTERS` | 8 | Number of market regime clusters |
| `ANOMALY_RATIO` | 0.07 | Fraction of synthetic anomalies injected |
| `RECON_WEIGHT` | 0.45 | Reconstruction score weight in hybrid fusion |
| `ENERGY_WEIGHT` | 0.30 | Energy score weight in hybrid fusion |
| `CLUSTER_WEIGHT` | 0.25 | Cluster score weight in hybrid fusion |

---

## Dataset

**EUR/USD H4 (4-hour) candlestick data** from 2015 to 2024, containing 14,566 rows with OHLC prices, tick volume, spread, and real volume. The preprocessing pipeline automatically generates 25 features including returns, moving averages, Bollinger Bands, RSI, MACD, and other technical indicators.

---

## Technical Details

### Self-Supervised Learning Objectives

- **Temporal Contrastive Learning**: Creates augmented views through temporal masking and noise injection. Uses NT-Xent (InfoNCE) loss with temperature scaling to learn invariant representations.
- **Masked Reconstruction**: Randomly masks time steps and reconstructs them using the transformer encoder. MSE loss captures temporal dependencies.

### Anomaly Detection Pipeline

- **Energy-Based Detection**: Trains a neural energy function conditioned on cluster assignments. Normal samples receive low energy; anomalies receive high energy via contrastive margin loss.
- **Reconstruction-Based Detection**: Computes per-sample reconstruction error. Anomalies that deviate from learned normal patterns produce higher reconstruction loss.
- **Hybrid Fusion**: Combines normalized scores from all three components (reconstruction, cluster distance, energy) using a weighted sum + max aggregation strategy.

### Threshold Tuning

Multi-stage search on the validation set:
1. Wide search over percentile-based thresholds (70th–99.9th)
2. Fine-grained refinement around best candidate
3. Precision-constrained optimization (minimum 30% precision)

---

## References

- Vaswani et al., "Attention is All You Need" (2017)
- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (2020)
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (2021)
- Grathwohl et al., "Your Classifier is Secretly an Energy Based Model" (2019)

---

## License

Educational and research use only.
