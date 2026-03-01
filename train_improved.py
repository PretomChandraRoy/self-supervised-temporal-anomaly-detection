"""
IMPROVED TRAINING SCRIPT - Comprehensive Anomaly Detection
Implements all improvements for achieving F1 > 70%:
1. Extended training (100 epochs)
2. Fixed energy detector with gradient clipping
3. Validation-based threshold tuning
4. Better synthetic anomaly generation
5. Improved hybrid fusion
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
# CONFIGURATION
# ============================================================================

class Config:
    """Improved configuration for better results"""

    # Data
    DATA_PATH = '../forexPredictor/H4_EURUSD_2015.csv'
    WINDOW_SIZE = 60
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Model
    D_MODEL = 128
    N_HEADS = 8
    N_LAYERS = 4
    DROPOUT = 0.1

    # Training
    N_EPOCHS = 100  # INCREASED from 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0

    # Energy detector (FIXED)
    ENERGY_EPOCHS = 50  # INCREASED
    ENERGY_LR = 5e-5  # LOWER for stability
    ENERGY_GRADIENT_CLIP = 0.5  # AGGRESSIVE clipping

    # Clustering
    N_CLUSTERS = 10

    # Anomaly detection
    THRESHOLD_PERCENTILES = [80, 85, 90, 95]  # Test multiple
    FUSION_WEIGHTS = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]  # Test multiple

    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR = 'improved_outputs'
    SAVE_CHECKPOINTS = True
    EARLY_STOPPING_PATIENCE = 15


# ============================================================================
# IMPROVED SYNTHETIC ANOMALY INJECTION
# ============================================================================

def inject_realistic_anomalies(data, anomaly_ratio=0.05):
    """
    Inject realistic anomalies matching the training distribution

    Args:
        data: DataFrame with OHLC data
        anomaly_ratio: fraction of samples to inject anomalies

    Returns:
        data_with_anomalies: DataFrame with injected anomalies
        anomaly_mask: Boolean mask of anomalies
    """
    n_samples = len(data)
    n_anomalies = int(n_samples * anomaly_ratio)

    # Calculate realistic scaling factors from training data statistics
    price_std = data['close'].std()
    volume_std = data.get('tick_volume', data.get('volume', pd.Series([1000]))).std()

    # Use smaller multipliers for realistic anomalies
    price_spike_multipliers = [1.5, 2.0, 2.5]  # Reduced from 5-20
    volatility_multipliers = [1.5, 2.0, 2.5]    # Reduced from 3-5
    trend_break_multipliers = [0.03, 0.05, 0.08]  # Reduced from 0.15

    anomaly_mask = np.zeros(n_samples, dtype=bool)
    data_modified = data.copy()

    # Get indices, ensuring we don't modify edges
    safe_indices = np.arange(Config.WINDOW_SIZE, n_samples - Config.WINDOW_SIZE)
    anomaly_indices = np.random.choice(safe_indices, n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['price_spike', 'volatility_spike', 'trend_break'])

        if anomaly_type == 'price_spike':
            # Sudden price movement
            multiplier = np.random.choice(price_spike_multipliers)
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

        elif anomaly_type == 'volatility_spike':
            # Increased volatility
            multiplier = np.random.choice(volatility_multipliers)
            base_range = data_modified.iloc[idx]['high'] - data_modified.iloc[idx]['low']
            new_range = base_range * multiplier

            mid = (data_modified.iloc[idx]['high'] + data_modified.iloc[idx]['low']) / 2
            data_modified.iloc[idx, data_modified.columns.get_loc('high')] = mid + new_range / 2
            data_modified.iloc[idx, data_modified.columns.get_loc('low')] = mid - new_range / 2

        elif anomaly_type == 'trend_break':
            # Sudden reversal
            multiplier = np.random.choice(trend_break_multipliers)
            window = data_modified.iloc[max(0, idx-10):idx]

            if len(window) > 0:
                trend = window['close'].iloc[-1] - window['close'].iloc[0]
                reversal = -trend * multiplier

                data_modified.iloc[idx, data_modified.columns.get_loc('close')] += reversal

        anomaly_mask[idx] = True

    print(f"✓ Injected {n_anomalies} realistic anomalies ({anomaly_ratio*100:.1f}%)")
    print(f"  - Anomaly types distribution:")

    return data_modified, anomaly_mask


# ============================================================================
# IMPROVED ENERGY DETECTOR TRAINING
# ============================================================================

def train_energy_detector_fixed(energy_detector, model, train_loader, cluster_labels,
                                 n_epochs, device, lr=5e-5):
    """
    Fixed energy detector training with aggressive NaN prevention
    """
    print("\n" + "="*60)
    print("Training Energy-Based Detector (FIXED)")
    print("="*60)

    optimizer = optim.AdamW(
        energy_detector.parameters(),
        lr=lr,
        weight_decay=1e-6,
        eps=1e-8  # Numerical stability
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epochs):
        energy_detector.train()
        model.eval()

        epoch_losses = []
        pbar = tqdm(train_loader, desc=f'Energy Epoch {epoch+1}/{n_epochs}')

        for batch_idx, (x,) in enumerate(pbar):
            x = x.to(device)
            batch_size = x.size(0)

            # Get embeddings (no gradients)
            with torch.no_grad():
                embeddings = model.get_embeddings(x)

                # Normalize embeddings for stability
                embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)

            # Get cluster labels
            batch_start = batch_idx * train_loader.batch_size
            batch_labels = []
            for i in range(batch_start, batch_start + batch_size):
                batch_labels.append(cluster_labels.get(i, 0))
            batch_labels = torch.tensor(batch_labels, device=device)

            optimizer.zero_grad()

            # Compute energy with numerical stability
            energy_scores = energy_detector.compute_energy(embeddings)

            # Check for NaN/Inf BEFORE computing loss
            if torch.isnan(energy_scores).any() or torch.isinf(energy_scores).any():
                print(f"\n⚠️ NaN/Inf in energy scores at epoch {epoch+1}, batch {batch_idx}")
                print(f"   Embedding stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
                print(f"   Skipping batch...")
                continue

            # Improved loss: push energy down for normal samples
            # Use Huber loss for robustness to outliers
            energy_mean = energy_scores.mean()
            energy_std = energy_scores.std() + 1e-8

            # Normalize energy scores
            normalized_energy = (energy_scores - energy_mean) / energy_std

            # Loss: minimize normalized energy magnitude
            loss = normalized_energy.abs().mean()

            # Add small regularization
            for param in energy_detector.parameters():
                loss += 1e-6 * param.pow(2).sum()

            # Check loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ NaN/Inf in loss at epoch {epoch+1}, batch {batch_idx}")
                continue

            loss.backward()

            # AGGRESSIVE gradient clipping
            torch.nn.utils.clip_grad_norm_(
                energy_detector.parameters(),
                max_norm=Config.ENERGY_GRADIENT_CLIP
            )

            # Check gradients
            total_norm = 0
            for p in energy_detector.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            if np.isnan(total_norm) or np.isinf(total_norm):
                print(f"\n⚠️ Invalid gradient norm at epoch {epoch+1}, batch {batch_idx}")
                continue

            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'grad_norm': f"{total_norm:.4f}"})

        if len(epoch_losses) == 0:
            print(f"\n❌ No valid batches in epoch {epoch+1}, stopping energy training")
            break

        avg_loss = np.mean(epoch_losses)
        print(f"\nEnergy Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        scheduler.step(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("\n✓ Energy detector training complete")
    return energy_detector


# ============================================================================
# VALIDATION-BASED THRESHOLD TUNING
# ============================================================================

def tune_threshold_on_validation(hybrid_detector, val_data, val_embeddings,
                                  val_clusters, val_ground_truth):
    """
    Find optimal threshold using validation set

    Args:
        hybrid_detector: HybridAnomalyDetector
        val_data: validation sequences
        val_embeddings: validation embeddings
        val_clusters: validation cluster labels
        val_ground_truth: ground truth anomaly labels

    Returns:
        best_threshold: optimal threshold
        best_f1: best F1 score achieved
    """
    print("\n" + "="*60)
    print("Tuning Threshold on Validation Set")
    print("="*60)

    # Get scores without threshold
    with torch.no_grad():
        scores, _, details = hybrid_detector.predict(val_data, val_embeddings, val_clusters)

        if torch.is_tensor(scores):
            scores_np = scores.cpu().numpy()
        else:
            scores_np = scores

    # Test multiple thresholds
    best_f1 = 0
    best_threshold = None
    best_metrics = None

    # Test percentile-based thresholds
    test_percentiles = range(70, 99, 2)

    results = []

    for percentile in test_percentiles:
        threshold = np.percentile(scores_np, percentile)
        predictions = scores_np > threshold

        # Compute metrics
        tp = np.sum((predictions == True) & (val_ground_truth == True))
        fp = np.sum((predictions == True) & (val_ground_truth == False))
        fn = np.sum((predictions == False) & (val_ground_truth == True))
        tn = np.sum((predictions == False) & (val_ground_truth == False))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'percentile': percentile,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = results[-1]

    # Print results
    print(f"\nThreshold Tuning Results:")
    print(f"{'Percentile':<12} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)

    for r in results[::2]:  # Print every other result
        print(f"{r['percentile']:<12} {r['threshold']:<12.4f} {r['precision']:<12.3f} "
              f"{r['recall']:<12.3f} {r['f1']:<12.3f}")

    print("\n" + "="*60)
    print(f"✓ Best Threshold: {best_threshold:.4f} (Percentile: {best_metrics['percentile']})")
    print(f"  Precision: {best_metrics['precision']:.3f}")
    print(f"  Recall: {best_metrics['recall']:.3f}")
    print(f"  F1 Score: {best_metrics['f1']:.3f}")
    print("="*60)

    return best_threshold, best_f1, best_metrics


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """
    Complete improved training pipeline
    """
    print("="*80)
    print("IMPROVED ANOMALY DETECTION TRAINING")
    print("Target: F1 > 70%")
    print("="*80)

    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{Config.OUTPUT_DIR}/checkpoints", exist_ok=True)

    # ========================================================================
    # 1. LOAD AND PREPARE DATA
    # ========================================================================

    print("\n[1/7] Loading and preparing data...")

    # Load data
    df = load_forex_data(Config.DATA_PATH)
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Inject realistic anomalies
    df_with_anomalies, ground_truth = inject_realistic_anomalies(df, anomaly_ratio=0.05)

    # Preprocess
    preprocessor = FinancialDataPreprocessor(
        window_size=Config.WINDOW_SIZE,
        stride=1
    )

    # Prepare data (includes feature engineering, scaling, and sequence creation)
    sequences, feature_names = preprocessor.prepare_data(df_with_anomalies, fit_scaler=True)

    print(f"✓ Created {len(sequences)} sequences with {len(feature_names)} features")

    # Align ground truth with sequences
    ground_truth_aligned = ground_truth[Config.WINDOW_SIZE-1:]
    ground_truth_aligned = ground_truth_aligned[:len(sequences)]

    # Split into train/val/test
    n_samples = len(sequences)
    n_train = int(n_samples * Config.TRAIN_RATIO)
    n_val = int(n_samples * Config.VAL_RATIO)

    train_data = sequences[:n_train]
    val_data = sequences[n_train:n_train+n_val]
    test_data = sequences[n_train+n_val:]

    train_gt = ground_truth_aligned[:n_train]
    val_gt = ground_truth_aligned[n_train:n_train+n_val]
    test_gt = ground_truth_aligned[n_train+n_val:]

    print(f"✓ Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print(f"  Train anomalies: {train_gt.sum()} ({train_gt.sum()/len(train_gt)*100:.1f}%)")
    print(f"  Val anomalies: {val_gt.sum()} ({val_gt.sum()/len(val_gt)*100:.1f}%)")
    print(f"  Test anomalies: {test_gt.sum()} ({test_gt.sum()/len(test_gt)*100:.1f}%)")

    # Create data loaders
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)
    test_tensor = torch.FloatTensor(test_data)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(test_tensor),
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    # ========================================================================
    # 2. INITIALIZE MODELS
    # ========================================================================

    print("\n[2/7] Initializing models...")

    n_features = train_data.shape[2]

    # Main model
    model = SelfSupervisedTemporalModel(
        n_features=n_features,
        d_model=Config.D_MODEL,
        n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)

    # Clustering
    clustering = DensityAwareClustering(
        n_clusters=Config.N_CLUSTERS,
        method='kmeans'
    )

    # Energy detector
    energy_detector = EnergyBasedAnomalyDetector(
        embedding_dim=Config.D_MODEL,
        n_clusters=Config.N_CLUSTERS
    ).to(Config.DEVICE)

    # Reconstruction detector
    reconstruction_detector = ReconstructionBasedDetector(
        reconstructor=model.reconstructor,
        threshold_percentile=90
    )

    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # ========================================================================
    # 3. TRAIN MAIN MODEL (100 EPOCHS)
    # ========================================================================

    print("\n[3/7] Training main model (100 epochs)...")
    print("="*60)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.N_EPOCHS,
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'contrastive_loss': [],
        'reconstruction_loss': []
    }

    for epoch in range(Config.N_EPOCHS):
        # Training
        model.train()
        train_losses = []
        contrastive_losses = []
        reconstruction_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.N_EPOCHS}')

        for x, in pbar:
            x = x.to(Config.DEVICE)

            optimizer.zero_grad()

            loss, losses = model(x, use_contrastive=True, use_reconstruction=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            optimizer.step()

            train_losses.append(loss.item())
            contrastive_losses.append(losses.get('contrastive', 0))
            reconstruction_losses.append(losses.get('reconstruction', 0))

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'contrast': f"{losses.get('contrastive', 0):.4f}",
                'recon': f"{losses.get('reconstruction', 0):.4f}"
            })

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, in val_loader:
                x = x.to(Config.DEVICE)
                loss, _ = model(x, use_contrastive=True, use_reconstruction=True)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_contrastive = np.mean(contrastive_losses)
        avg_reconstruction = np.mean(reconstruction_losses)

        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['contrastive_loss'].append(avg_contrastive)
        training_history['reconstruction_loss'].append(avg_reconstruction)

        print(f"\nEpoch {epoch+1}/{Config.N_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  Contrastive: {avg_contrastive:.4f}, Reconstruction: {avg_reconstruction:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            if Config.SAVE_CHECKPOINTS:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'training_history': training_history
                }, f"{Config.OUTPUT_DIR}/checkpoints/best_model.pt")
                print("  ✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                break

    print("\n✓ Main model training complete")

    # Load best model
    checkpoint = torch.load(f"{Config.OUTPUT_DIR}/checkpoints/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")

    # ========================================================================
    # 4. CLUSTERING
    # ========================================================================

    print("\n[4/7] Performing clustering...")

    model.eval()

    # Extract embeddings
    train_embeddings = []
    with torch.no_grad():
        for x, in tqdm(train_loader, desc="Extracting embeddings"):
            x = x.to(Config.DEVICE)
            emb = model.get_embeddings(x)
            train_embeddings.append(emb.cpu().numpy())

    train_embeddings = np.vstack(train_embeddings)
    print(f"✓ Extracted {len(train_embeddings)} embeddings")

    # Fit clustering
    cluster_labels = clustering.fit(train_embeddings)  # fit() returns labels

    # Create label dictionary
    cluster_dict = {i: label for i, label in enumerate(cluster_labels)}

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"✓ Clustering complete: {len(unique_labels)} clusters")
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")

    # ========================================================================
    # 5. TRAIN ENERGY DETECTOR (FIXED)
    # ========================================================================

    print("\n[5/7] Training energy detector...")

    energy_detector = train_energy_detector_fixed(
        energy_detector=energy_detector,
        model=model,
        train_loader=train_loader,
        cluster_labels=cluster_dict,
        n_epochs=Config.ENERGY_EPOCHS,
        device=Config.DEVICE,
        lr=Config.ENERGY_LR
    )

    # ========================================================================
    # 6. FIT ANOMALY DETECTORS
    # ========================================================================

    print("\n[6/7] Fitting anomaly detectors...")

    # Fit reconstruction detector
    reconstruction_detector.fit(train_tensor.to(Config.DEVICE))

    # Create hybrid detector
    hybrid_detector = HybridAnomalyDetector(
        energy_detector=energy_detector,
        reconstruction_detector=reconstruction_detector,
        energy_weight=0.5,
        reconstruction_weight=0.5,
        fusion_method='weighted_sum'
    )

    # Get validation embeddings
    val_embeddings = []
    with torch.no_grad():
        for x, in val_loader:
            x = x.to(Config.DEVICE)
            emb = model.get_embeddings(x)
            val_embeddings.append(emb)
    val_embeddings = torch.cat(val_embeddings, dim=0)

    # Get validation clusters
    val_clusters = clustering.predict(val_embeddings.cpu().numpy())
    val_clusters = torch.tensor(val_clusters, device=Config.DEVICE)

    # Tune threshold on validation set
    best_threshold, best_f1, metrics = tune_threshold_on_validation(
        hybrid_detector=hybrid_detector,
        val_data=val_tensor.to(Config.DEVICE),
        val_embeddings=val_embeddings,
        val_clusters=val_clusters,
        val_ground_truth=val_gt
    )

    # Set the optimal threshold
    hybrid_detector.threshold = best_threshold

    # ========================================================================
    # 7. EVALUATE ON TEST SET
    # ========================================================================

    print("\n[7/7] Evaluating on test set...")
    print("="*60)

    # Get test embeddings
    test_embeddings = []
    with torch.no_grad():
        for x, in test_loader:
            x = x.to(Config.DEVICE)
            emb = model.get_embeddings(x)
            test_embeddings.append(emb)
    test_embeddings = torch.cat(test_embeddings, dim=0)

    # Get test clusters
    test_clusters = clustering.predict(test_embeddings.cpu().numpy())
    test_clusters = torch.tensor(test_clusters, device=Config.DEVICE)

    # Predict
    with torch.no_grad():
        scores, is_anomaly, details = hybrid_detector.predict(
            test_tensor.to(Config.DEVICE),
            test_embeddings,
            test_clusters
        )

    # Convert to numpy
    if torch.is_tensor(is_anomaly):
        predictions = is_anomaly.cpu().numpy()
        scores_np = scores.cpu().numpy()
    else:
        predictions = is_anomaly
        scores_np = scores

    # Compute metrics
    tp = np.sum((predictions == True) & (test_gt == True))
    fp = np.sum((predictions == True) & (test_gt == False))
    fn = np.sum((predictions == False) & (test_gt == True))
    tn = np.sum((predictions == False) & (test_gt == False))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(test_gt)

    print("\n" + "="*60)
    print("FINAL TEST SET RESULTS")
    print("="*60)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:5d}  FP: {fp:5d}")
    print(f"  FN: {fn:5d}  TN: {tn:5d}")
    print(f"\nDetected {predictions.sum()} anomalies out of {len(test_gt)} samples")
    print(f"Ground truth: {test_gt.sum()} anomalies")
    print("="*60)

    # Save results
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'threshold': best_threshold,
        'n_epochs': Config.N_EPOCHS,
        'best_val_loss': best_val_loss
    }

    import json
    with open(f"{Config.OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {Config.OUTPUT_DIR}/results.json")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'energy_detector_state_dict': energy_detector.state_dict(),
        'clustering_model': clustering,
        'threshold': best_threshold,
        'results': results,
        'training_history': training_history
    }, f"{Config.OUTPUT_DIR}/final_model.pt")

    print(f"✓ Model saved to {Config.OUTPUT_DIR}/final_model.pt")

    print("\n" + "="*80)
    if f1 >= 0.70:
        print("🎉 SUCCESS! Achieved F1 > 70%")
    else:
        print(f"⚠️  F1 = {f1:.1%} (Target: 70%)")
        print("   Consider: more epochs, different hyperparameters, or more data")
    print("="*80)


if __name__ == '__main__':
    main()

