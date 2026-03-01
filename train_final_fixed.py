"""
FINAL FIXED VERSION - Proper Loss Normalization
This version uses MSE normalization and proper threshold tuning
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.models.temporal_transformer import SelfSupervisedTemporalModel
from anomaly_detection.data.preprocessing import FinancialDataPreprocessor, load_forex_data


class FinalConfig:
    """Final working configuration"""

    DATA_PATH = '../forexPredictor/H4_EURUSD_2015.csv'
    WINDOW_SIZE = 60
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Model
    D_MODEL = 128
    N_HEADS = 8
    N_LAYERS = 3  # Reduced for faster convergence
    DROPOUT = 0.2  # Increased for better generalization

    # Training
    N_EPOCHS = 100
    BATCH_SIZE = 64  # Larger batch for more stable gradients
    LEARNING_RATE = 5e-4  # Higher LR
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0

    # Loss weights - CRITICAL
    CONTRASTIVE_WEIGHT = 0.0  # Turn off contrastive for now
    RECONSTRUCTION_WEIGHT = 1.0

    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR = 'final_outputs'
    EARLY_STOPPING_PATIENCE = 20

    # Detection
    THRESHOLD_PERCENTILE = 95


def inject_anomalies(data, anomaly_ratio=0.05):
    """Simple anomaly injection"""
    n_samples = len(data)
    n_anomalies = int(n_samples * anomaly_ratio)

    price_std = data['close'].std()
    anomaly_mask = np.zeros(n_samples, dtype=bool)
    data_modified = data.copy()

    safe_indices = np.arange(FinalConfig.WINDOW_SIZE, n_samples - FinalConfig.WINDOW_SIZE)
    if len(safe_indices) < n_anomalies:
        n_anomalies = len(safe_indices) // 2

    anomaly_indices = np.random.choice(safe_indices, n_anomalies, replace=False)

    for idx in anomaly_indices:
        # Price spike
        multiplier = np.random.uniform(2.0, 3.0)
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

        anomaly_mask[idx] = True

    print(f"✓ Injected {n_anomalies} anomalies ({anomaly_ratio*100:.1f}%)")
    return data_modified, anomaly_mask


def compute_reconstruction_scores(model, data_loader, device):
    """Compute reconstruction error scores"""
    model.eval()
    all_scores = []

    with torch.no_grad():
        for x, in data_loader:
            x = x.to(device)

            # Get reconstruction
            encoder_output = model.encoder(x)
            reconstructed = model.reconstructor.reconstruction_head(encoder_output)

            # Compute MSE per sample
            mse = ((x - reconstructed) ** 2).mean(dim=(1, 2))
            all_scores.append(mse.cpu())

    return torch.cat(all_scores)


def main():
    print("="*80)
    print("FINAL FIXED VERSION - Anomaly Detection")
    print("With proper loss normalization and threshold tuning")
    print("="*80)

    os.makedirs(FinalConfig.OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{FinalConfig.OUTPUT_DIR}/checkpoints", exist_ok=True)

    # Load data
    print("\n[1/7] Loading data...")
    df = load_forex_data(FinalConfig.DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # Inject anomalies
    df_with_anomalies, ground_truth = inject_anomalies(df, anomaly_ratio=0.05)

    # Preprocess
    preprocessor = FinancialDataPreprocessor(window_size=FinalConfig.WINDOW_SIZE, stride=1)
    sequences, feature_names = preprocessor.prepare_data(df_with_anomalies, fit_scaler=True)
    print(f"✓ Created {len(sequences)} sequences with {len(feature_names)} features")

    # Align ground truth
    ground_truth_aligned = ground_truth[FinalConfig.WINDOW_SIZE-1:]
    ground_truth_aligned = ground_truth_aligned[:len(sequences)]

    # Split
    n_samples = len(sequences)
    n_train = int(n_samples * FinalConfig.TRAIN_RATIO)
    n_val = int(n_samples * FinalConfig.VAL_RATIO)

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

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=FinalConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=FinalConfig.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=FinalConfig.BATCH_SIZE, shuffle=False)

    # Initialize model
    print("\n[2/7] Initializing model...")
    n_features = train_data.shape[2]

    model = SelfSupervisedTemporalModel(
        n_features=n_features,
        d_model=FinalConfig.D_MODEL,
        n_heads=FinalConfig.N_HEADS,
        n_layers=FinalConfig.N_LAYERS,
        dropout=FinalConfig.DROPOUT,
        contrastive_weight=FinalConfig.CONTRASTIVE_WEIGHT,
        reconstruction_weight=FinalConfig.RECONSTRUCTION_WEIGHT
    ).to(FinalConfig.DEVICE)

    print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    print(f"\n[3/7] Training ({FinalConfig.N_EPOCHS} epochs)...")

    optimizer = optim.AdamW(model.parameters(), lr=FinalConfig.LEARNING_RATE, weight_decay=FinalConfig.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(FinalConfig.N_EPOCHS):
        model.train()
        train_losses = []

        for x, in train_loader:
            x = x.to(FinalConfig.DEVICE)

            optimizer.zero_grad()

            # Only reconstruction loss (contrastive turned off)
            loss, losses = model(x, use_contrastive=False, use_reconstruction=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FinalConfig.GRADIENT_CLIP)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, in val_loader:
                x = x.to(FinalConfig.DEVICE)
                loss, _ = model(x, use_contrastive=False, use_reconstruction=True)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{FinalConfig.OUTPUT_DIR}/checkpoints/best_model.pt")
            print("  ✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= FinalConfig.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best
    model.load_state_dict(torch.load(f"{FinalConfig.OUTPUT_DIR}/checkpoints/best_model.pt"))
    print("\n✓ Training complete")

    # Compute reconstruction scores
    print("\n[4/7] Computing reconstruction scores...")

    train_scores = compute_reconstruction_scores(model, train_loader, FinalConfig.DEVICE)
    val_scores = compute_reconstruction_scores(model, val_loader, FinalConfig.DEVICE)
    test_scores = compute_reconstruction_scores(model, test_loader, FinalConfig.DEVICE)

    print(f"Train scores: mean={train_scores.mean():.4f}, std={train_scores.std():.4f}")
    print(f"Val scores:   mean={val_scores.mean():.4f}, std={val_scores.std():.4f}")
    print(f"Test scores:  mean={test_scores.mean():.4f}, std={test_scores.std():.4f}")

    # Tune threshold on validation set
    print("\n[5/7] Tuning threshold on validation set...")

    best_f1 = 0
    best_threshold = None
    best_percentile = None

    for percentile in range(80, 99, 2):
        threshold = np.percentile(train_scores.numpy(), percentile)
        predictions = val_scores.numpy() > threshold

        tp = np.sum((predictions == True) & (val_gt == True))
        fp = np.sum((predictions == True) & (val_gt == False))
        fn = np.sum((predictions == False) & (val_gt == True))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_percentile = percentile

    print(f"✓ Best threshold: {best_threshold:.4f} (percentile {best_percentile}, Val F1: {best_f1:.3f})")

    # Test
    print("\n[6/7] Testing...")

    predictions = test_scores.numpy() > best_threshold

    tp = np.sum((predictions == True) & (test_gt == True))
    fp = np.sum((predictions == True) & (test_gt == False))
    fn = np.sum((predictions == False) & (test_gt == True))
    tn = np.sum((predictions == False) & (test_gt == False))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(test_gt)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TN: {tn:4d}")
    print(f"\nDetected {predictions.sum()} anomalies out of {len(test_gt)} samples")
    print(f"Ground truth: {test_gt.sum()} anomalies")
    print("="*60)

    # Save
    print("\n[7/7] Saving results...")
    results = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'threshold': float(best_threshold),
        'threshold_percentile': int(best_percentile),
        'n_epochs': epoch + 1,
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(avg_train_loss),
        'final_val_loss': float(avg_val_loss)
    }

    import json
    with open(f"{FinalConfig.OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
        'config': {
            'd_model': FinalConfig.D_MODEL,
            'n_heads': FinalConfig.N_HEADS,
            'n_layers': FinalConfig.N_LAYERS,
            'threshold': best_threshold
        }
    }, f"{FinalConfig.OUTPUT_DIR}/final_model.pt")

    print(f"✓ Results saved to {FinalConfig.OUTPUT_DIR}/")

    # Final assessment
    print("\n" + "="*80)
    if f1 >= 0.60:
        print("🎉 EXCELLENT! F1 > 60% - Thesis ready!")
    elif f1 >= 0.40:
        print("✅ GOOD! F1 > 40% - Acceptable for thesis")
        print("   Consider: longer training or parameter tuning for better results")
    else:
        print("⚠️  F1 < 40% - May need adjustments")
        print("   Suggestions:")
        print("   - Check if anomalies are realistic enough")
        print("   - Try different threshold percentiles")
        print("   - Increase model capacity")
    print("="*80)


if __name__ == '__main__':
    main()

