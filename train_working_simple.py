"""
WORKING VERSION - With Better Default Parameters
This version has been tested and works with the current codebase
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.models.temporal_transformer import SelfSupervisedTemporalModel
from anomaly_detection.models.clustering import DensityAwareClustering
from anomaly_detection.models.anomaly_detector import (
    EnergyBasedAnomalyDetector,
    ReconstructionBasedDetector,
    HybridAnomalyDetector
)
from anomaly_detection.data.preprocessing import FinancialDataPreprocessor, load_forex_data


# ============================================================================
# CONFIGURATION - WORKING DEFAULTS
# ============================================================================

class WorkingConfig:
    """Configuration that actually works with current codebase"""

    # Data
    DATA_PATH = '../forexPredictor/H4_EURUSD_2015.csv'
    WINDOW_SIZE = 60
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Model - Use defaults that work
    D_MODEL = 128
    N_HEADS = 8
    N_LAYERS = 4
    DROPOUT = 0.1

    # Training
    N_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0

    # Loss weights - CRITICAL: Balance the losses
    CONTRASTIVE_WEIGHT = 0.1  # Lower weight for contrastive
    RECONSTRUCTION_WEIGHT = 1.0

    # Energy detector
    ENERGY_EPOCHS = 20
    ENERGY_LR = 5e-5
    ENERGY_GRADIENT_CLIP = 0.5

    # Clustering
    N_CLUSTERS = 10

    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR = 'working_outputs'
    EARLY_STOPPING_PATIENCE = 15

    # Anomaly detection - Use reconstruction-based primarily
    USE_RECONSTRUCTION_ONLY = True  # Skip energy detector if unstable
    THRESHOLD_PERCENTILE = 95  # Start with simple threshold


def inject_realistic_anomalies(data, anomaly_ratio=0.05):
    """Inject realistic anomalies"""
    n_samples = len(data)
    n_anomalies = int(n_samples * anomaly_ratio)

    price_std = data['close'].std()

    anomaly_mask = np.zeros(n_samples, dtype=bool)
    data_modified = data.copy()

    safe_indices = np.arange(WorkingConfig.WINDOW_SIZE, n_samples - WorkingConfig.WINDOW_SIZE)
    if len(safe_indices) < n_anomalies:
        n_anomalies = len(safe_indices) // 2

    anomaly_indices = np.random.choice(safe_indices, n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['price_spike', 'volatility_spike'])

        if anomaly_type == 'price_spike':
            multiplier = np.random.uniform(1.5, 2.5)
            direction = np.random.choice([-1, 1])
            spike = price_std * multiplier * direction

            data_modified.iloc[idx, data_modified.columns.get_loc('close')] += spike
            data_modified.iloc[idx, data_modified.columns.get_loc('high')] = max(
                data_modified.iloc[idx]['high'],
                data_modified.iloc[idx]['close']
            )
            data_modified.iloc[idx, data_modified.columns.get_loc('low')] = min(
                data_modified.iloc[idx]['low'],
                data_modified.iloc[idx]['close']
            )
        else:
            multiplier = np.random.uniform(1.5, 2.5)
            base_range = data_modified.iloc[idx]['high'] - data_modified.iloc[idx]['low']
            new_range = base_range * multiplier

            mid = (data_modified.iloc[idx]['high'] + data_modified.iloc[idx]['low']) / 2
            data_modified.iloc[idx, data_modified.columns.get_loc('high')] = mid + new_range / 2
            data_modified.iloc[idx, data_modified.columns.get_loc('low')] = mid - new_range / 2

        anomaly_mask[idx] = True

    print(f"✓ Injected {n_anomalies} realistic anomalies ({anomaly_ratio*100:.1f}%)")
    return data_modified, anomaly_mask


def main():
    """Main training pipeline with working defaults"""
    print("="*80)
    print("WORKING VERSION - Anomaly Detection Training")
    print("Using tested defaults that work with current codebase")
    print("="*80)

    os.makedirs(WorkingConfig.OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{WorkingConfig.OUTPUT_DIR}/checkpoints", exist_ok=True)

    # Load data
    print("\n[1/6] Loading data...")
    df = load_forex_data(WorkingConfig.DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # Inject anomalies
    df_with_anomalies, ground_truth = inject_realistic_anomalies(df, anomaly_ratio=0.05)

    # Preprocess
    preprocessor = FinancialDataPreprocessor(
        window_size=WorkingConfig.WINDOW_SIZE,
        stride=1
    )

    sequences, feature_names = preprocessor.prepare_data(df_with_anomalies, fit_scaler=True)
    print(f"✓ Created {len(sequences)} sequences with {len(feature_names)} features")

    # Align ground truth
    ground_truth_aligned = ground_truth[WorkingConfig.WINDOW_SIZE-1:]
    ground_truth_aligned = ground_truth_aligned[:len(sequences)]

    # Split
    n_samples = len(sequences)
    n_train = int(n_samples * WorkingConfig.TRAIN_RATIO)
    n_val = int(n_samples * WorkingConfig.VAL_RATIO)

    train_data = sequences[:n_train]
    val_data = sequences[n_train:n_train+n_val]
    test_data = sequences[n_train+n_val:]

    train_gt = ground_truth_aligned[:n_train]
    val_gt = ground_truth_aligned[n_train:n_train+n_val]
    test_gt = ground_truth_aligned[n_train+n_val:]

    print(f"✓ Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print(f"  Test anomalies: {test_gt.sum()} ({test_gt.sum()/len(test_gt)*100:.1f}%)")

    # Create loaders
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)
    test_tensor = torch.FloatTensor(test_data)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=WorkingConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=WorkingConfig.BATCH_SIZE, shuffle=False)

    # Initialize model
    print("\n[2/6] Initializing model...")
    n_features = train_data.shape[2]

    model = SelfSupervisedTemporalModel(
        n_features=n_features,
        d_model=WorkingConfig.D_MODEL,
        n_heads=WorkingConfig.N_HEADS,
        n_layers=WorkingConfig.N_LAYERS,
        dropout=WorkingConfig.DROPOUT,
        contrastive_weight=WorkingConfig.CONTRASTIVE_WEIGHT,
        reconstruction_weight=WorkingConfig.RECONSTRUCTION_WEIGHT
    ).to(WorkingConfig.DEVICE)

    print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    print(f"\n[3/6] Training ({WorkingConfig.N_EPOCHS} epochs)...")

    optimizer = optim.AdamW(model.parameters(), lr=WorkingConfig.LEARNING_RATE, weight_decay=WorkingConfig.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=WorkingConfig.N_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(WorkingConfig.N_EPOCHS):
        model.train()
        train_losses = []

        for x, in tqdm(train_loader, desc=f'Epoch {epoch+1}/{WorkingConfig.N_EPOCHS}', leave=False):
            x = x.to(WorkingConfig.DEVICE)

            optimizer.zero_grad()
            loss, losses = model(x, use_contrastive=True, use_reconstruction=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), WorkingConfig.GRADIENT_CLIP)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, in val_loader:
                x = x.to(WorkingConfig.DEVICE)
                loss, _ = model(x, use_contrastive=True, use_reconstruction=True)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{WorkingConfig.OUTPUT_DIR}/checkpoints/best_model.pt")
            print("  ✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= WorkingConfig.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best
    model.load_state_dict(torch.load(f"{WorkingConfig.OUTPUT_DIR}/checkpoints/best_model.pt"))
    print("✓ Training complete")

    # Fit reconstruction detector
    print("\n[4/6] Fitting reconstruction detector...")
    reconstruction_detector = ReconstructionBasedDetector(
        reconstructor=model.reconstructor,
        threshold_percentile=WorkingConfig.THRESHOLD_PERCENTILE
    )
    reconstruction_detector.fit(train_tensor.to(WorkingConfig.DEVICE))

    # Test
    print("\n[5/6] Testing...")
    model.eval()

    with torch.no_grad():
        scores, is_anomaly = reconstruction_detector.predict(test_tensor.to(WorkingConfig.DEVICE))

    predictions = is_anomaly.cpu().numpy() if torch.is_tensor(is_anomaly) else is_anomaly

    # Metrics
    tp = np.sum((predictions == True) & (test_gt == True))
    fp = np.sum((predictions == True) & (test_gt == False))
    fn = np.sum((predictions == False) & (test_gt == True))
    tn = np.sum((predictions == False) & (test_gt == False))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*60)
    print("RESULTS (Reconstruction-Based Detection)")
    print("="*60)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TN: {tn:4d}")
    print("="*60)

    # Save
    print("\n[6/6] Saving results...")
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'n_epochs': epoch + 1,
        'best_val_loss': best_val_loss
    }

    import json
    with open(f"{WorkingConfig.OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results
    }, f"{WorkingConfig.OUTPUT_DIR}/final_model.pt")

    print(f"✓ Results saved to {WorkingConfig.OUTPUT_DIR}/results.json")

    if f1 >= 0.60:
        print("\n🎉 Good results! F1 > 60%")
    elif f1 >= 0.40:
        print("\n✅ Acceptable results. May need more training or tuning.")
    else:
        print("\n⚠️  Low F1. Try:")
        print("   - More epochs (150-200)")
        print("   - Lower learning rate")
        print("   - Different threshold percentile")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()

