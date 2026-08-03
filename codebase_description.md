# Self-Supervised Temporal Anomaly Detection — Full Codebase Description

## Overview

This is a **thesis-level research project** for detecting anomalies in financial time-series data (specifically EUR/USD 4-hour OHLC forex data). It uses a **self-supervised deep learning framework** that requires **no labeled anomalies during model training** — instead, anomaly detection emerges from learned representations.

The high-level pipeline is:

```
Raw OHLC CSV → Feature Engineering → Sliding Windows → Self-Supervised Encoder Training → Clustering → Energy Detector Training → Threshold Tuning → Anomaly Scoring & Evaluation
```

The target performance is F1 > 70%.

---

## Project Structure

```
anomaly_detection/
├── data/
│   ├── H4_EURUSD_2015.csv         # Raw forex data (EUR/USD, 4h bars)
│   └── preprocessing.py           # Feature engineering, sliding windows, scaling
├── models/
│   ├── temporal_transformer.py    # Core model: encoder, contrastive learning, masking
│   ├── anomaly_detector.py        # Energy-based & reconstruction-based detectors
│   └── clustering.py              # Density-aware clustering for market regime discovery
├── utils/
│   ├── evaluation.py              # Metrics: P/R/F1/AUC, financial impact
│   └── visualization.py           # (Support visualization utilities)
├── figs_style.py                  # Publication-quality matplotlib style
├── train.py                       # Main training & evaluation orchestration (3770 lines)
└── report_generator.py            # Excel report generation
```

---

## 1. Data Pipeline — `data/preprocessing.py`

### Input
A CSV file of 4-hour EUR/USD OHLC bars (`open`, `high`, `low`, `close`, `tick_volume`).

### Steps

#### 1.1 Feature Engineering — [`add_features()`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/data/preprocessing.py#L77-L136)
Enriches raw OHLC with ~25 technical indicators:
- **Price features**: returns, log-returns
- **Moving averages**: SMA(20), SMA(50), EMA(12), EMA(26)
- **MACD**: line, signal, histogram
- **RSI** (14-period)
- **Stochastic Oscillator** (%K, %D)
- **ATR** (volatility proxy)
- **ADX** (trend strength)
- **Bollinger Bands** (width, %B position)
- **Volume ratio** (current / 20-MA)
- **Price range** metrics (high-low, close-open)

#### 1.2 Outlier Clipping — [`clip_outliers_by_percentile()`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/data/preprocessing.py#L138-L162)
Clips each feature at the 1st / 99th percentile to prevent extreme values from dominating scaling.

#### 1.3 Scaling — [`prepare_data()`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/data/preprocessing.py#L164-L258)
Uses **RobustScaler** by default (median and IQR-based), which is resistant to outliers. Automatically drops columns that are timestamp-like, unnamed, have extreme values (>100,000), or near-zero variance.

#### 1.4 Sliding Windows — [`create_sequences()`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/data/preprocessing.py#L260-L274)
Creates overlapping windows of `window_size=60` time steps with `stride=1`, producing tensors of shape `(N, 60, n_features)`. Each window represents approximately 10 days of 4-hour bars.

#### 1.5 Data Split
Chronological 70% / 15% / 15% train/validation/test split (no shuffling, to respect time order).

---

## 2. Core Model — `models/temporal_transformer.py`

The main model is [`SelfSupervisedTemporalModel`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/temporal_transformer.py#L580-L676), which wraps a shared encoder with two self-supervised learning objectives.

### 2.1 Architecture — [`TemporalTransformerEncoder`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/temporal_transformer.py#L267-L390)

The encoder backbone, with shape `(B, T, n_features) → (B, T, d_model)`.

**Configuration** (from `ImprovedConfig`):
| Param | Value |
|---|---|
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_layers` | 2 |
| `dim_feedforward` | 4×64 = 256 |
| `dropout` | 0.25 |
| `max_seq_len` | 512 |

**Data flow:**
1. **MSTC Front-end** (optional) — applies multi-scale convolutions before projection
2. **Linear Projection** — `n_features → d_model`
3. **Positional Encoding** — sinusoidal, shape `(1, T, d_model)`
4. **Transformer Encoder Layers** — `n_layers` of Pre-LN transformer blocks (norm_first=True)
5. **LayerNorm** — final normalization
6. **HRG Post-encoder** (optional) — hierarchical regime guidance refinement

To get a **global embedding** (for clustering/anomaly scoring), [`get_sequence_embedding()`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/temporal_transformer.py#L372-L390) applies **mean + max pooling** over the time dimension and projects `(d_model×2) → d_model`. Max-pooling preserves anomaly peaks that mean-pooling would dilute.

---

### 2.2 Novel Block — Multi-Scale Temporal Context (MSTC)
> [`MultiScaleTemporalContext`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/temporal_transformer.py#L112-L157)

A **convolutional front-end** that runs before the linear projection, operating on raw features in `(B, C, T)` format.

- Splits channels into **3 groups** (roughly equal thirds)
- Each group uses a **learnable-dilation depthwise Conv1d** ([`_DepthwiseConv1dContinuousDilation`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/temporal_transformer.py#L70-L109)) with initial dilations of 1, 2, 3
- The dilation `alpha` is a **learnable scalar** — it interpolates between floor and ceil integer dilations, so the model learns what temporal scale each feature group cares about
- Groups are concatenated, then fused with a pointwise conv + Squeeze-and-Excitation (SE) gate + residual

**Purpose**: Captures short-, medium-, and long-range temporal patterns before feeding into the transformer.

---

### 2.3 Novel Block — Hierarchical Regime Guidance (HRG)
> [`HierarchicalRegimeGuidance`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/temporal_transformer.py#L184-L260)

A **post-encoder refinement block** applied after the transformer.

- Takes encoded `H: (B, C, T)` and runs it through **4 parallel level branches** with dilations 1, 2, 3, 4 (multi-scale depthwise convolutions)
- Fuses the 4 branches `(4C → 2C → C)` into a reference tensor `f`
- A **level predictor** (global avg pool → softmax) computes per-sample weights `π` over the 4 levels, producing a **soft mixture** `h_bar`
- A **guidance head** produces `g: (B, 1, T)` in `[0, 1]` — a temporal gate indicating which timesteps are "interesting"
- Output: `H' = H + g * h_bar` (gated residual)

**Purpose**: Hierarchically re-weights the encoder output based on market regime complexity, giving more attention to temporally relevant positions.

---

### 2.4 Self-Supervised Objective 1 — Masked Reconstruction
> [`MaskedTimeSeriesReconstructor`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/temporal_transformer.py#L393-L478)

Inspired by **Masked Autoencoders (MAE)**:
1. Randomly masks `mask_ratio=40%` of the time steps
2. Encodes the full sequence (with masking applied via a learned `mask_token`)
3. A **reconstruction head** (`d_model → dim_ff → n_features`) predicts the original values at masked positions
4. Loss: **MSE only at masked positions**

**During inference**, deterministic masking of the **last 4 time steps** is used ([`_apply_inference_masking()`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/anomaly_detector.py#L310-L325)), because anomaly labels use "last-point labeling" — the label of a 60-step window reflects whether its last timesteps contain an anomaly.

---

### 2.5 Self-Supervised Objective 2 — Temporal Contrastive Learning
> [`TemporalContrastiveLearning`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/temporal_transformer.py#L481-L577)

Uses **NT-Xent loss** (SimCLR-style) with temporal augmentations:
- **View 1**: Random time masking (10% dropout)
- **View 2**: Small Gaussian noise addition (σ=0.01)
- Both views are encoded, projected (`d_model → 128`), L2-normalized
- NT-Xent loss pushes representations of the **same sequence to be similar** while pushing different sequences apart

**Combined loss**: `total = contrastive_weight × L_contrastive + L_reconstruction`
With `contrastive_weight = 0.05` (reconstruction is dominant).

---

## 3. Clustering — `models/clustering.py`

### [`DensityAwareClustering`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/clustering.py#L14-L355)

After the encoder is trained, its embeddings are clustered to discover **normal market regimes** (bull, bear, sideways, high-vol, etc.).

- **Method**: KMeans with `n_clusters=8`, `n_init=10`
- **Normal cluster identification**: clusters are ranked by density (size / avg distance to center); high-density clusters are labeled "normal"
- **Cluster anomaly scores** combine:
  1. Distance to assigned cluster center (z-score normalized per cluster)
  2. Membership score (abnormal cluster = 0.8 baseline)
  3. Outlier score within the cluster (distance > 90th percentile of cluster)
- **Regime transition scores**: counts rapid cluster-switching in a local window — instability signals anomalies

### [`LatentSpaceRegularizer`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/clustering.py#L358-L432)

An optional fine-tuning step after clustering:
- **Center loss**: pulls each embedding toward its cluster center
- **Separation loss**: pushes cluster centers apart (negative average pairwise distance)
- Centers are updated with EMA (`α=0.1`)

---

## 4. Anomaly Detection — `models/anomaly_detector.py`

### 4.1 [`EnergyBasedAnomalyDetector`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/anomaly_detector.py#L15-L196)

Assigns a scalar **energy** to each embedding — higher energy = more anomalous.

**Standard architecture** (non-gated):
```
embedding (d) → Linear(d, 256) → residual block → Linear(256, 128) → Linear(128, 1)
```

**Gated Dual-Detector Head** (optional, `USE_GATED_HEAD=True`):
- **Shared trunk**: Linear → BN → GeLU → Dropout × 3, producing hidden `h`
- **Gating network**: `h → Linear → ReLU → Linear → Sigmoid` (element-wise gate)
- **Energy branch**: separate path with residual + Softplus activation (ensures positive energy)
- **Reconstruction proxy head**: `h * gate → Linear(1)` — a proxy score for use in the cascade detector

**Training** ([`train_energy_detector_stable()`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/train.py#L301-L437)):
Uses **injected ground-truth labels** (not unsupervised), with a sophisticated compound loss:
1. **Per-sample hinge loss**: each anomaly must exceed `normal_mean + margin`; each normal must be below `anomaly_mean - margin` (margin ramps 2.0→5.0 over 30 epochs)
2. **Variance penalty**: compresses tails of both distributions
3. **Push targets**: normal → energy < -2; anomaly → energy > 5
4. **Focal loss** (α=0.75 for anomalies, γ=2): focuses on hard examples near the boundary
5. **L2 regularization**

**Cluster normalization**: after training, per-cluster energy statistics (mean, std) are computed and used to normalize scores at inference time.

---

### 4.2 [`ReconstructionBasedDetector`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/anomaly_detector.py#L202-L411)

Measures how well the encoder can reconstruct **the last few timesteps** from context.

**Error computation** ([`compute_reconstruction_error()`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/anomaly_detector.py#L231-L275)):
- Focuses on **last 4 timesteps** (where the anomaly label lives)
- Per-timestep: `0.8 × max(feature errors) + 0.2 × mean(feature errors)` — a single-feature spike is not diluted
- Over time: `0.7 × max(timestep error) + 0.3 × top-2 mean` — sharper peak detection

**Mahalanobis distance** (optional): instead of Euclidean, fits full covariance on training residuals for a statistically-grounded distance metric.

**Threshold**: set at the 95th percentile of training scores.

---

### 4.3 [`HybridAnomalyDetector`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/models/anomaly_detector.py#L414-L540)

Fuses energy and reconstruction scores. Supports three fusion methods:
- `weighted_sum` (default): normalizes both to [0,1] then combines with weights
- `max`: takes the maximum of the two scores
- `product`: multiplicative combination

---

## 5. Threshold Tuning — [`tune_threshold_on_validation()`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/train.py#L460-L770)

This is a critical and sophisticated step. Several strategies compete and the best F1 wins:

| Strategy | Description |
|---|---|
| **Per-component** | Find the best threshold for each component independently |
| **OR-ensemble** | A sample is anomalous if ANY discriminative component (d' > 0.5) exceeds its threshold |
| **Weighted sum** | Normalize + combine all components with config weights; grid search threshold |
| **Recon-only** | Use reconstruction score alone |
| **Cascade** | Flag if recon > threshold, OR if recon passes but energy is very high (energy "rescues" recon's misses) |

**d'-prime** (signal detectability) is used to decide which components are discriminative. Only components with d' > 0.5 are included in the OR-ensemble.

Thresholds are converted to **percentiles** before application to the test set, making them distribution-invariant.

---

## 6. Anomaly Injection — [`inject_diverse_anomalies()`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/train.py#L147-L298)

Since no real labeled anomalies are available, synthetic anomalies are injected into the training data **before the encoder is trained**, at a 5% rate (`ANOMALY_RATIO=0.05`), each affecting **3 consecutive timesteps** (`ANOMALY_WINDOW=3`).

**Anomaly types** (with probabilities):
| Type | Prob. | Description |
|---|---|---|
| `price_spike` | 20% | Sudden directional price jump scaled by local volatility |
| `volatility_spike` | 20% | Extreme range expansion (high-low widens dramatically) |
| `flash_crash` | 20% | Severe drop with partial recovery |
| `volume_spike` | 15% | Extreme volume + correlated price impact |
| `trend_break` | 15% | Sharp deviation from local mean |
| `gap_anomaly` | 10% | Open/close gap from prior close |

**Intensity**: `ANOMALY_INTENSITY=12.0` (12× local rolling std), strong enough to survive `RobustScaler` clipping.

> [!IMPORTANT]
> The encoder is trained on the **anomaly-injected** data but with **no knowledge of the labels** (self-supervised). The labels are only used to train the energy detector and to evaluate performance.

---

## 7. Training Pipeline — `train.py` (Main Orchestration)

### Configuration — [`ImprovedConfig`](file:///d:/Documents/11th%20semester/Thesis/Main%20Files/anomaly_detection/train.py#L62-L145)

Key hyperparameters:

| Category | Param | Value |
|---|---|---|
| Model | D_MODEL | 64 |
| Model | N_HEADS / N_LAYERS | 4 / 2 |
| Model | DROPOUT / MASK_RATIO | 0.25 / 0.40 |
| Training | N_EPOCHS / BATCH_SIZE | 150 / 64 |
| Training | LEARNING_RATE | 3e-4 (AdamW) |
| Training | WEIGHT_DECAY | 1e-3 |
| Energy | ENERGY_EPOCHS | 150 |
| Clustering | N_CLUSTERS | 8 |
| Detection | ENERGY_WEIGHT / RECON_WEIGHT | 0.50 / 0.50 |
| Anomalies | ANOMALY_RATIO / INTENSITY | 0.05 / 12.0 |
| Toggle | USE_MSTC / USE_HRG / USE_GATED_HEAD | All True |

### Training Flow

```
1. Load & preprocess OHLC data
2. Inject synthetic anomalies (5%, 6 types)
3. Engineer features → sliding windows (60 steps, stride 1)
4. Split: 70% train / 15% val / 15% test
    ↓
5. Train SelfSupervisedTemporalModel (150 epochs):
   - AdamW + CosineAnnealing LR scheduler
   - Loss = reconstruction MSE + 0.05 × contrastive NT-Xent
   - Early stopping (patience=40)
    ↓
6. Extract embeddings for training set
7. Fit DensityAwareClustering (KMeans, k=8)
8. Optional latent space regularization (10 epochs)
    ↓
9. Fit ReconstructionBasedDetector (threshold at 95th pctl)
10. Train EnergyBasedAnomalyDetector (150 epochs, supervised with GT labels)
11. Update cluster energy statistics
    ↓
12. Tune thresholds on validation set:
    - Compare OR-ensemble, weighted-sum, cascade, recon-only
    - Pick best F1 strategy
    ↓
13. Evaluate on test set:
    - Compute all component scores
    - Apply winning threshold strategy
    - Report P / R / F1 / AUROC / AUPRC
    ↓
14. Run ablation study (if RUN_ABLATION=True):
    - Baseline (no MSTC, no HRG, no gated head)
    - +MSTC only
    - +HRG only
    - Full model
    ↓
15. Save figures, Excel report, predictions
```

---

## 8. Evaluation Metrics — `utils/evaluation.py`

| Metric | Description |
|---|---|
| **Precision** | Of all flagged anomalies, what fraction is real |
| **Recall** | Of all real anomalies, what fraction is caught |
| **F1** | Harmonic mean of precision and recall |
| **AUROC** | Area under ROC curve (threshold-independent) |
| **AUPRC** | Area under Precision-Recall curve (more informative for imbalanced data) |
| **Specificity** | True-negative rate |
| **d' (d-prime)** | Signal detectability measure from signal-detection theory |
| **Financial Impact** | Simulated P&L if anomaly signals are traded |

---

## 9. Visualization — `train.py` (Thesis Figures)

The code generates 15+ publication-ready figures saved as both `.pdf` (for LaTeX) and `.png`:

| Figure | Content |
|---|---|
| `1_training_curves` | Total/contrastive/reconstruction loss + improvement % |
| `2_confusion_matrix` | Breakdown by anomaly type (7×2 heatmap) |
| `2b_quadrant_dashboard` | TP/FP/FN/TN infographic |
| `2c_multi_threshold_cm` | 4×4 grid of confusion matrices at 16 thresholds |
| `3_performance_metrics` | Precision/Recall/F1/Accuracy bar chart |
| `4_score_distribution` | Histogram + boxplot of anomaly scores by class |
| `5_precision_recall_curve` | PR curve with AUC and operating point |
| `6_detection_timeline` | Anomaly score over time with detection markers |
| `7_results_dashboard` | Multi-panel summary |
| `8_tsne_embeddings` | t-SNE of learned embeddings colored by class |
| `9_roc_curve` | ROC curve with AUROC |
| `10_score_components` | Individual score distributions for each detector |
| `11_cluster_analysis` | Cluster assignments in embedding space |
| `12_ablation_study` | F1 comparison across architectural variants |
| HRG guidance maps | Per-sample temporal guidance weights |

---

## 10. Publication Style — `figs_style.py`

A centralized style module applied to all figures:
- **Color palette**: teal-blue (`#2A6F97`) for normal class, amber (`#E1812C`) for anomalies/accent, green/red for good/bad outcomes
- **Typography**: DejaVu Sans/Arial, 10.5pt
- **Layout**: No top/right spines, horizontal grid only
- **Output**: 300 DPI PNG + PDF
- **d' function**: pooled-variance d-prime metric for signal detectability reporting

---

## 11. Key Design Decisions

| Decision | Rationale |
|---|---|
| Self-supervised pretraining | No labeled anomalies available in real forex data |
| Mask ratio 40% (vs. typical 15%) | Forces model to learn structure, not memorize sequences |
| Focus on last 4 timesteps for reconstruction | Anomaly labels use last-point labeling (`ANOMALY_WINDOW=3`) |
| Low contrastive weight (0.05) | Reconstruction is a stronger anomaly signal for this task |
| Energy detector trained with GT labels | Bridges the gap from self-supervised to supervised |
| Cascade / OR-ensemble thresholding | Energy catches anomalies that reconstruction misses |
| RobustScaler + outlier clipping | Financial data has fat tails; prevents normalization collapse |
| Anomaly intensity 12.0× local std | Must survive RobustScaler, otherwise anomalies are invisible |
| Max+mean pooling for embeddings | Max-pool preserves single-timestep anomaly peaks |

---

## 12. Environment

- **Language**: Python 3.x
- **Framework**: PyTorch
- **Key dependencies**: `torch`, `scikit-learn`, `pandas`, `numpy`, `ta` (technical analysis), `matplotlib`, `seaborn`
- **Hardware**: CUDA GPU if available, CPU fallback
- **Data**: `data/H4_EURUSD_2015.csv` (~886 KB, approximately 3+ years of 4h EUR/USD bars)
- **Output**: timestamped `improved_outputs_YYYYMMDD_HHMMSS/` directory with plots, predictions JSON, and Excel report
