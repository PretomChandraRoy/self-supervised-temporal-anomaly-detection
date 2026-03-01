"""
QUICK TEST VERSION - Fast validation of improvements
Tests all fixes with only 20 epochs (~30 minutes)
Use this to verify everything works before running 100 epochs
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
from pathlib import Path
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
# QUICK TEST CONFIGURATION
# ============================================================================

class QuickConfig:
    """Fast configuration for testing (20 epochs, ~30 min)"""

    # Data
    DATA_PATH = '../forexPredictor/H4_EURUSD_2015.csv'
    WINDOW_SIZE = 60
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Model (smaller for speed)
    D_MODEL = 64  # Reduced from 128
    N_HEADS = 4   # Reduced from 8
    N_LAYERS = 2  # Reduced from 4
    DROPOUT = 0.1

    # Training (QUICK TEST)
    N_EPOCHS = 20  # Quick test
    BATCH_SIZE = 64  # Larger for speed
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0

    # Energy detector (FIXED)
    ENERGY_EPOCHS = 10  # Quick test
    ENERGY_LR = 5e-5
    ENERGY_GRADIENT_CLIP = 0.5

    # Clustering
    N_CLUSTERS = 8  # Reduced from 10

    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR = 'quick_test_outputs'

    # Sample size (use subset of data for speed)
    MAX_SAMPLES = 5000  # Use first 5000 samples only


def inject_quick_anomalies(data, anomaly_ratio=0.05):
    """Quick realistic anomaly injection"""
    n_samples = len(data)
    n_anomalies = int(n_samples * anomaly_ratio)

    price_std = data['close'].std()

    anomaly_mask = np.zeros(n_samples, dtype=bool)
    data_modified = data.copy()

    safe_indices = np.arange(QuickConfig.WINDOW_SIZE, n_samples - QuickConfig.WINDOW_SIZE)
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

    print(f"✓ Injected {n_anomalies} anomalies ({anomaly_ratio*100:.1f}%)")
    return data_modified, anomaly_mask


def train_energy_detector_quick(energy_detector, model, train_loader, cluster_labels,
                                  n_epochs, device):
    """Quick energy detector training"""
    print("\n" + "="*60)
    print("Training Energy Detector (Quick)")
    print("="*60)

    optimizer = optim.AdamW(
        energy_detector.parameters(),
        lr=QuickConfig.ENERGY_LR,
        weight_decay=1e-6,
        eps=1e-8
    )

    for epoch in range(n_epochs):
        energy_detector.train()
        model.eval()

        epoch_losses = []
        pbar = tqdm(train_loader, desc=f'Energy Epoch {epoch+1}/{n_epochs}')

        for batch_idx, (x,) in enumerate(pbar):
            x = x.to(device)
            batch_size = x.size(0)

            with torch.no_grad():
                embeddings = model.get_embeddings(x)
                embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)

            batch_start = batch_idx * train_loader.batch_size
            batch_labels = []
            for i in range(batch_start, batch_start + batch_size):
                batch_labels.append(cluster_labels.get(i, 0))
            batch_labels = torch.tensor(batch_labels, device=device)

            optimizer.zero_grad()

            energy_scores = energy_detector.compute_energy(embeddings)

            if torch.isnan(energy_scores).any() or torch.isinf(energy_scores).any():
                continue

            energy_mean = energy_scores.mean()
            energy_std = energy_scores.std() + 1e-8
            normalized_energy = (energy_scores - energy_mean) / energy_std

            loss = normalized_energy.abs().mean()

            for param in energy_detector.parameters():
                loss += 1e-6 * param.pow(2).sum()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                energy_detector.parameters(),
                max_norm=QuickConfig.ENERGY_GRADIENT_CLIP
            )
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        if len(epoch_losses) > 0:
            print(f"Energy Epoch {epoch+1}: Loss = {np.mean(epoch_losses):.4f}")
        else:
            print(f"Energy Epoch {epoch+1}: No valid batches")
            break

    return energy_detector


def main():
    """Quick test pipeline"""
    print("="*80)
    print("QUICK TEST - Improved Anomaly Detection")
    print("20 epochs, reduced model, ~30 minutes")
    print("="*80)

    os.makedirs(QuickConfig.OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\n[1/7] Loading data...")
    df = load_forex_data(QuickConfig.DATA_PATH)

    # Use subset for speed
    df = df.iloc[:QuickConfig.MAX_SAMPLES]
    print(f"Using {len(df)} samples for quick test")

    # Inject anomalies
    df_with_anomalies, ground_truth = inject_quick_anomalies(df, anomaly_ratio=0.05)

    # Preprocess
    preprocessor = FinancialDataPreprocessor(
        window_size=QuickConfig.WINDOW_SIZE,
        stride=1
    )

    # Prepare data (includes feature engineering, scaling, and sequence creation)
    sequences, feature_names = preprocessor.prepare_data(df_with_anomalies, fit_scaler=True)

    print(f"✓ Created {len(sequences)} sequences with {len(feature_names)} features")

    # Align ground truth
    ground_truth_aligned = ground_truth[QuickConfig.WINDOW_SIZE-1:]
    ground_truth_aligned = ground_truth_aligned[:len(sequences)]

    # Split
    n_samples = len(sequences)
    n_train = int(n_samples * QuickConfig.TRAIN_RATIO)
    n_val = int(n_samples * QuickConfig.VAL_RATIO)

    train_data = sequences[:n_train]
    val_data = sequences[n_train:n_train+n_val]
    test_data = sequences[n_train+n_val:]

    train_gt = ground_truth_aligned[:n_train]
    val_gt = ground_truth_aligned[n_train:n_train+n_val]
    test_gt = ground_truth_aligned[n_train+n_val:]

    print(f"✓ Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # Create loaders
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)
    test_tensor = torch.FloatTensor(test_data)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=QuickConfig.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=QuickConfig.BATCH_SIZE,
        shuffle=False
    )

    # Initialize models
    print("\n[2/7] Initializing models...")
    n_features = train_data.shape[2]

    model = SelfSupervisedTemporalModel(
        n_features=n_features,
        d_model=QuickConfig.D_MODEL,
        n_heads=QuickConfig.N_HEADS,
        n_layers=QuickConfig.N_LAYERS,
        dropout=QuickConfig.DROPOUT
    ).to(QuickConfig.DEVICE)

    clustering = DensityAwareClustering(
        n_clusters=QuickConfig.N_CLUSTERS,
        method='kmeans'
    )

    energy_detector = EnergyBasedAnomalyDetector(
        embedding_dim=QuickConfig.D_MODEL,
        n_clusters=QuickConfig.N_CLUSTERS
    ).to(QuickConfig.DEVICE)

    reconstruction_detector = ReconstructionBasedDetector(
        reconstructor=model.reconstructor,
        threshold_percentile=90
    )

    print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train main model
    print("\n[3/7] Training main model (20 epochs)...")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=QuickConfig.LEARNING_RATE,
        weight_decay=QuickConfig.WEIGHT_DECAY
    )

    best_val_loss = float('inf')

    for epoch in range(QuickConfig.N_EPOCHS):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{QuickConfig.N_EPOCHS}')

        for x, in pbar:
            x = x.to(QuickConfig.DEVICE)

            optimizer.zero_grad()
            loss, losses = model(x, use_contrastive=True, use_reconstruction=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), QuickConfig.GRADIENT_CLIP)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, in val_loader:
                x = x.to(QuickConfig.DEVICE)
                loss, _ = model(x, use_contrastive=True, use_reconstruction=True)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{QuickConfig.OUTPUT_DIR}/best_model.pt")

    # Load best
    model.load_state_dict(torch.load(f"{QuickConfig.OUTPUT_DIR}/best_model.pt"))
    print("✓ Loaded best model")

    # Clustering
    print("\n[4/7] Clustering...")
    model.eval()

    train_embeddings = []
    with torch.no_grad():
        for x, in tqdm(train_loader, desc="Extracting embeddings"):
            x = x.to(QuickConfig.DEVICE)
            emb = model.get_embeddings(x)
            train_embeddings.append(emb.cpu().numpy())

    train_embeddings = np.vstack(train_embeddings)
    cluster_labels = clustering.fit(train_embeddings)  # fit() returns labels
    cluster_dict = {i: label for i, label in enumerate(cluster_labels)}

    print(f"✓ Clustered into {len(np.unique(cluster_labels))} clusters")

    # Train energy detector
    print("\n[5/7] Training energy detector...")
    energy_detector = train_energy_detector_quick(
        energy_detector, model, train_loader, cluster_dict,
        QuickConfig.ENERGY_EPOCHS, QuickConfig.DEVICE
    )

    # Fit detectors
    print("\n[6/7] Fitting detectors...")
    reconstruction_detector.fit(train_tensor.to(QuickConfig.DEVICE))

    hybrid_detector = HybridAnomalyDetector(
        energy_detector=energy_detector,
        reconstruction_detector=reconstruction_detector,
        energy_weight=0.5,
        reconstruction_weight=0.5
    )

    # Tune threshold on validation
    val_embeddings = []
    with torch.no_grad():
        for x, in val_loader:
            x = x.to(QuickConfig.DEVICE)
            emb = model.get_embeddings(x)
            val_embeddings.append(emb)
    val_embeddings = torch.cat(val_embeddings, dim=0)

    val_clusters = clustering.predict(val_embeddings.cpu().numpy())
    val_clusters = torch.tensor(val_clusters, device=QuickConfig.DEVICE)

    # Find best threshold
    with torch.no_grad():
        scores, _, _ = hybrid_detector.predict(
            val_tensor.to(QuickConfig.DEVICE),
            val_embeddings,
            val_clusters
        )
        scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else scores

    best_f1 = 0
    best_threshold = None

    for percentile in range(70, 96, 5):
        threshold = np.percentile(scores_np, percentile)
        predictions = scores_np > threshold

        tp = np.sum((predictions == True) & (val_gt == True))
        fp = np.sum((predictions == True) & (val_gt == False))
        fn = np.sum((predictions == False) & (val_gt == True))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    hybrid_detector.threshold = best_threshold
    print(f"✓ Best threshold: {best_threshold:.4f} (Val F1: {best_f1:.3f})")

    # Test
    print("\n[7/7] Testing...")

    test_embeddings = []
    with torch.no_grad():
        for x, in DataLoader(TensorDataset(test_tensor), batch_size=QuickConfig.BATCH_SIZE):
            x = x.to(QuickConfig.DEVICE)
            emb = model.get_embeddings(x)
            test_embeddings.append(emb)
    test_embeddings = torch.cat(test_embeddings, dim=0)

    test_clusters = clustering.predict(test_embeddings.cpu().numpy())
    test_clusters = torch.tensor(test_clusters, device=QuickConfig.DEVICE)

    with torch.no_grad():
        scores, is_anomaly, _ = hybrid_detector.predict(
            test_tensor.to(QuickConfig.DEVICE),
            test_embeddings,
            test_clusters
        )

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
    print("QUICK TEST RESULTS")
    print("="*60)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TN: {tn:4d}")
    print("="*60)

    if f1 >= 0.50:
        print("✅ Good! Run train_improved.py for 100 epochs to get F1 > 70%")
    else:
        print("⚠️  Results may need more tuning")

    print("\nQuick test complete!")


if __name__ == '__main__':
    main()

