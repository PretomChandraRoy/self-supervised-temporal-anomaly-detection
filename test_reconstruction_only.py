"""
Simple Validation - Using Only Reconstruction-Based Detection
Since energy detector has NaN issues, use reconstruction only
"""

import torch
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.temporal_transformer import SelfSupervisedTemporalModel
from models.anomaly_detector import ReconstructionBasedDetector
from data.preprocessing import FinancialDataPreprocessor, load_forex_data
from validation.synthetic_anomalies import (
    inject_combined_anomalies,
    evaluate_detection,
    print_evaluation_results,
    create_validation_report
)

print("="*80)
print("SIMPLE VALIDATION: Reconstruction-Based Detection Only")
print("="*80)

# Configuration
DATA_PATH = '../forexPredictor/H4_EURUSD_2015.csv'
CHECKPOINT_PATH = 'example_outputs/checkpoints/final_model.pt'
WINDOW_SIZE = 60
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ANOMALY_RATIO = 0.05

# Step 1: Load and prepare data
print("\n[1/5] Loading clean data...")
df = load_forex_data(DATA_PATH)
df_validation = df.iloc[-5000:].copy()
print(f"✓ Loaded {len(df_validation)} samples")

# Step 2: Inject anomalies
print("\n[2/5] Injecting synthetic anomalies...")
df_with_anomalies, ground_truth = inject_combined_anomalies(
    df_validation, anomaly_ratio=ANOMALY_RATIO, seed=42
)
print(f"✓ Ground truth: {ground_truth.sum()} anomalies")

# Step 3: Preprocess
print("\n[3/5] Preprocessing...")
preprocessor = FinancialDataPreprocessor(
    window_size=WINDOW_SIZE,
    stride=1,
    add_technical_indicators=True,
    scaler_type='robust',
    clip_outliers=True
)
sequences, feature_names = preprocessor.prepare_data(df_with_anomalies, fit_scaler=True)
n_features = sequences.shape[2]

ground_truth_aligned = ground_truth[WINDOW_SIZE-1:][:len(sequences)]
print(f"✓ {len(sequences)} sequences, {ground_truth_aligned.sum()} anomalies")

# Step 4: Load model
print("\n[4/5] Loading model...")
if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
    print("Please run example.py first to train the model")
    sys.exit(1)

model = SelfSupervisedTemporalModel(
    n_features=n_features,
    d_model=128,
    n_heads=8,
    n_layers=4,
    dropout=0.1,
    reconstruction_weight=0.1,
    contrastive_weight=1.0
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print("✓ Model loaded")

# Step 5: Detect anomalies using ONLY reconstruction
print("\n[5/5] Detecting anomalies (reconstruction-based)...")

recon_detector = ReconstructionBasedDetector(
    reconstructor=model.reconstructor,
    threshold_percentile=90,  # Lowered from 95 to catch more anomalies
    use_mahalanobis=False  # Use simple MSE
)

test_data = torch.FloatTensor(sequences).to(DEVICE)

with torch.no_grad():
    # Fit detector on data
    all_errors = []
    for i in range(0, len(test_data), 32):
        batch = test_data[i:i+32]
        reconstructed, _, _ = model.reconstructor(batch)
        errors = ((batch - reconstructed) ** 2).mean(dim=(1, 2))
        all_errors.append(errors.cpu())

    all_errors = torch.cat(all_errors)
    threshold = torch.quantile(all_errors, 0.90)  # Changed from 0.95

    print(f"✓ Reconstruction threshold: {threshold:.4f}")
    print(f"✓ Error range: [{all_errors.min():.4f}, {all_errors.max():.4f}]")

    # Detect
    is_anomaly = all_errors > threshold
    predictions = is_anomaly.numpy()
    anomaly_scores = all_errors.numpy()

print(f"✓ Detected {predictions.sum()} anomalies ({predictions.sum()/len(predictions)*100:.2f}%)")

# Evaluate
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

metrics = evaluate_detection(predictions, ground_truth_aligned)
print_evaluation_results(metrics)

# Save report
os.makedirs('validation_outputs', exist_ok=True)
create_validation_report(
    metrics,
    save_path='validation_outputs/simple_validation_report.txt'
)

# Analysis
print("\n" + "="*80)
print("SCORE ANALYSIS")
print("="*80)

if ground_truth_aligned.sum() > 0:
    true_anomaly_scores = anomaly_scores[ground_truth_aligned]
    normal_scores = anomaly_scores[~ground_truth_aligned]

    print(f"\nTrue Anomalies (n={len(true_anomaly_scores)}):")
    print(f"  Mean: {true_anomaly_scores.mean():.4f}")
    print(f"  Median: {np.median(true_anomaly_scores):.4f}")
    print(f"  Max: {true_anomaly_scores.max():.4f}")

    print(f"\nNormal Samples (n={len(normal_scores)}):")
    print(f"  Mean: {normal_scores.mean():.4f}")
    print(f"  Median: {np.median(normal_scores):.4f}")
    print(f"  Max: {normal_scores.max():.4f}")

    print(f"\nSeparation:")
    print(f"  Mean diff: {true_anomaly_scores.mean() - normal_scores.mean():.4f}")
    print(f"  Threshold: {threshold:.4f}")

# Final verdict
print("\n" + "="*80)
if metrics['f1'] >= 0.6:
    print("✅ VALIDATION PASSED - Good detection capability")
elif metrics['f1'] >= 0.4:
    print("⚠️  MODERATE - Model shows some detection capability")
else:
    print("❌ VALIDATION FAILED - Model needs improvement")
    print("\nPossible issues:")
    print("  - Model undertrained (run more epochs)")
    print("  - Threshold too high (lower percentile)")
    print("  - Features don't capture anomaly patterns")
print("="*80)

