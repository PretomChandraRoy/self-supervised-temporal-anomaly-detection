"""
IMPROVED FULL TRAINING - Production-Ready Anomaly Detection
100 epochs, stable energy detector, hybrid fusion, F1 > 70% target
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
import json
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
#git test
# Add current directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.temporal_transformer import SelfSupervisedTemporalModel
from models.clustering import DensityAwareClustering
from models.anomaly_detector import (
    EnergyBasedAnomalyDetector,
    ReconstructionBasedDetector,
    HybridAnomalyDetector
)
from data.preprocessing import FinancialDataPreprocessor, load_forex_data

# Import Excel generation
from report_generator import generate_detailed_results_excel


# ============================================================================
# IMPROVED CONFIGURATION
# ============================================================================

class ImprovedConfig:
    """Optimized configuration for production-level results"""

    # Data
    DATA_PATH = 'data/H4_EURUSD_2015.csv'
    WINDOW_SIZE = 60
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Model Architecture - Increased capacity
    D_MODEL = 192  # Increased for better representation
    N_HEADS = 8
    N_LAYERS = 5  # Deeper model
    DROPOUT = 0.2  # Higher dropout to prevent overfitting

    # Training - Extended with better convergence
    N_EPOCHS = 150  # More epochs for better convergence
    BATCH_SIZE = 64  # Larger batch for stability
    LEARNING_RATE = 1e-4  # Slightly higher LR with larger batch
    WEIGHT_DECAY = 1e-4  # Stronger regularization
    GRADIENT_CLIP = 1.0  # Standard clipping

    # Loss weights - Emphasize reconstruction for anomaly detection
    CONTRASTIVE_WEIGHT = 0.1  # Moderate contrastive learning
    RECONSTRUCTION_WEIGHT = 1.0

    # Energy detector - More training with better hyperparameters
    USE_ENERGY_DETECTOR = True
    ENERGY_EPOCHS = 80  # Extended training for better convergence
    ENERGY_LR = 1e-4  # Higher LR for faster convergence
    ENERGY_GRADIENT_CLIP = 0.5
    ENERGY_WEIGHT_DECAY = 1e-5  # Less weight decay for energy detector

    # Clustering - Fewer, more meaningful clusters
    N_CLUSTERS = 8  # Reduced for clearer separation
    MIN_CLUSTER_SIZE = 100  # Require larger clusters

    # Hybrid Detection - Reconstruction is now strongest (bottleneck makes it discriminative)
    USE_HYBRID = True
    ENERGY_WEIGHT = 0.20  # Energy detector weight (important for thesis)
    RECON_WEIGHT = 0.60   # Reconstruction weight (strongest with bottleneck)
    CLUSTER_WEIGHT = 0.20 # Cluster-based anomaly scoring weight

    # Precision constraint for threshold tuning
    MIN_PRECISION = 0.30  # Require higher precision to reduce false positives

    # Anomaly Injection - Higher intensity since we no longer clip outliers
    ANOMALY_RATIO = 0.07  # 7% anomalies for sufficient training signal
    ANOMALY_INTENSITY = 4.0  # INCREASED - no more clipping, need strong post-scaling signal

    # Threshold Tuning - More granular search at higher percentiles
    USE_VALIDATION_TUNING = True
    THRESHOLD_SEARCH_STEPS = 200  # More steps for finer search
    THRESHOLD_PERCENTILE_MIN = 90  # Start higher for better precision
    THRESHOLD_PERCENTILE_MAX = 99.9

    # Ensemble Detection
    USE_ISOLATION_FOREST = False  # DISABLED - adds noise, hurts precision
    ISOLATION_CONTAMINATION = 0.05  # Expected contamination rate

    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR = 'improved_outputs'
    EARLY_STOPPING_PATIENCE = 30  # More patience

    # Reporting
    SAVE_PLOTS = True
    SAVE_PREDICTIONS = True


def inject_diverse_anomalies(data, anomaly_ratio=0.05, intensity=2.0):
    """
    Inject diverse, realistic financial anomalies with STRONG signals
    """
    n_samples = len(data)
    n_anomalies = int(n_samples * anomaly_ratio)

    price_std = data['close'].std()
    volume_std = data['tick_volume'].std()

    # Use rolling statistics for more context-aware anomalies
    rolling_std = data['close'].rolling(window=20).std().fillna(price_std)

    anomaly_mask = np.zeros(n_samples, dtype=bool)
    data_modified = data.copy()

    # Convert tick_volume to float to allow decimal values from multiplication
    if 'tick_volume' in data_modified.columns:
        data_modified['tick_volume'] = data_modified['tick_volume'].astype(float)

    safe_indices = np.arange(ImprovedConfig.WINDOW_SIZE, n_samples - ImprovedConfig.WINDOW_SIZE)
    if len(safe_indices) < n_anomalies:
        n_anomalies = len(safe_indices) // 2

    anomaly_indices = np.random.choice(safe_indices, n_anomalies, replace=False)

    anomaly_types = []
    for idx in anomaly_indices:
        # Diverse anomaly types with weighted distribution
        anomaly_type = np.random.choice([
            'price_spike', 'volatility_spike', 'volume_spike',
            'trend_break', 'flash_crash', 'gap_anomaly'
        ], p=[0.2, 0.2, 0.15, 0.15, 0.2, 0.1])
        anomaly_types.append(anomaly_type)

        # Use local volatility for context-aware anomalies
        local_std = rolling_std.iloc[idx] if not np.isnan(rolling_std.iloc[idx]) else price_std

        if anomaly_type == 'price_spike':
            # Strong sudden price jump (use higher multiplier)
            multiplier = np.random.uniform(intensity, intensity + 2.0)
            direction = np.random.choice([-1, 1])
            spike = local_std * multiplier * direction

            data_modified.iloc[idx, data_modified.columns.get_loc('close')] += spike
            data_modified.iloc[idx, data_modified.columns.get_loc('high')] = max(
                data_modified.iloc[idx]['high'],
                data_modified.iloc[idx]['close'] + abs(spike) * 0.3
            )
            data_modified.iloc[idx, data_modified.columns.get_loc('low')] = min(
                data_modified.iloc[idx]['low'],
                data_modified.iloc[idx]['close'] - abs(spike) * 0.3
            )

        elif anomaly_type == 'volatility_spike':
            # Extreme volatility (much larger range)
            multiplier = np.random.uniform(intensity + 1.0, intensity + 3.0)
            base_range = data_modified.iloc[idx]['high'] - data_modified.iloc[idx]['low']
            new_range = base_range * multiplier

            mid = (data_modified.iloc[idx]['high'] + data_modified.iloc[idx]['low']) / 2
            data_modified.iloc[idx, data_modified.columns.get_loc('high')] = mid + new_range / 2
            data_modified.iloc[idx, data_modified.columns.get_loc('low')] = mid - new_range / 2

        elif anomaly_type == 'volume_spike':
            # Very unusual volume
            multiplier = np.random.uniform(intensity + 3.0, intensity + 8.0)
            data_modified.iloc[idx, data_modified.columns.get_loc('tick_volume')] *= multiplier

        elif anomaly_type == 'trend_break':
            # Strong sudden reversal affecting multiple bars
            window_start = max(0, idx-5)
            mean_price = data_modified.iloc[window_start:idx]['close'].mean()
            deviation = local_std * intensity * 1.5
            new_price = mean_price + deviation if np.random.rand() > 0.5 else mean_price - deviation
            data_modified.iloc[idx, data_modified.columns.get_loc('close')] = new_price
            data_modified.iloc[idx, data_modified.columns.get_loc('open')] = mean_price

        elif anomaly_type == 'flash_crash':
            # Quick severe drop and partial recovery
            crash_depth = local_std * intensity * 2.0
            data_modified.iloc[idx, data_modified.columns.get_loc('low')] -= crash_depth
            data_modified.iloc[idx, data_modified.columns.get_loc('close')] -= crash_depth * 0.7
            data_modified.iloc[idx, data_modified.columns.get_loc('open')] -= crash_depth * 0.2

        elif anomaly_type == 'gap_anomaly':
            # Gap up/down from previous close
            if idx > 0:
                prev_close = data_modified.iloc[idx-1]['close']
                gap = local_std * intensity * np.random.choice([-1, 1])
                data_modified.iloc[idx, data_modified.columns.get_loc('open')] = prev_close + gap
                data_modified.iloc[idx, data_modified.columns.get_loc('high')] = max(
                    data_modified.iloc[idx]['high'], prev_close + gap
                )
                data_modified.iloc[idx, data_modified.columns.get_loc('low')] = min(
                    data_modified.iloc[idx]['low'], prev_close + gap
                )

        anomaly_mask[idx] = True

    print(f"✓ Injected {n_anomalies} diverse anomalies ({anomaly_ratio*100:.1f}%)")
    print(f"  Types: {dict(pd.Series(anomaly_types).value_counts())}")
    return data_modified, anomaly_mask


def train_energy_detector_stable(energy_detector, train_tensor, train_gt, embedder, config):
    """
    Train energy detector using ACTUAL ground truth labels (not cluster labels).
    Uses a dedicated non-shuffled DataLoader to keep label alignment correct.
    """
    print("\n" + "="*60)
    print("Training Stable Energy Detector (Supervised with Ground Truth)")
    print("="*60)

    # Create a DataLoader that includes ground truth labels to avoid shuffle misalignment
    gt_tensor = torch.FloatTensor(train_gt.astype(np.float32))
    energy_dataset = TensorDataset(train_tensor, gt_tensor)
    energy_loader = DataLoader(energy_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    n_anomalies = train_gt.sum()
    n_normal = len(train_gt) - n_anomalies
    print(f"  Training samples: {n_normal} normal, {n_anomalies} anomalous ({n_anomalies/len(train_gt)*100:.1f}%)")

    optimizer = optim.AdamW(
        energy_detector.parameters(),
        lr=config.ENERGY_LR,
        weight_decay=config.ENERGY_WEIGHT_DECAY
    )

    embedder.eval()
    energy_detector.train()

    best_loss = float('inf')
    nan_count = 0

    for epoch in range(config.ENERGY_EPOCHS):
        epoch_losses = []

        for x, labels in energy_loader:
            x = x.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Get embeddings from frozen encoder
            with torch.no_grad():
                embeddings = embedder.get_embeddings(x)

            # Compute raw energy (no cluster normalization during training)
            energies = energy_detector.compute_energy(embeddings)

            # Use actual ground truth: normal (0) vs anomaly (1)
            is_anomaly = labels.bool()
            is_normal = ~is_anomaly

            if is_normal.sum() > 0 and is_anomaly.sum() > 0:
                normal_energy = energies[is_normal]
                anomaly_energy = energies[is_anomaly]

                # Clamp for numerical stability
                normal_energy = torch.clamp(normal_energy, -10, 10)
                anomaly_energy = torch.clamp(anomaly_energy, -10, 10)

                # Margin loss: anomaly energy should be at least `margin` higher than normal
                margin = 3.0
                margin_loss = torch.relu(margin + normal_energy.mean() - anomaly_energy.mean())

                # Push normal energies low, anomaly energies high
                normal_push = torch.relu(normal_energy - 0.0).mean()   # Push below 0
                anomaly_push = torch.relu(2.0 - anomaly_energy).mean()  # Push above 2

                # Binary cross-entropy style: sigmoid of energy as anomaly probability
                anomaly_prob = torch.sigmoid(energies)
                bce_loss = nn.functional.binary_cross_entropy(anomaly_prob, labels, reduction='mean')

                # L2 regularization
                reg_loss = 0.001 * (energies ** 2).mean()

                loss = margin_loss + 0.3 * normal_push + 0.3 * anomaly_push + 0.5 * bce_loss + reg_loss
            elif is_normal.sum() > 0:
                # Only normal samples in this batch
                loss = torch.relu(energies[is_normal]).mean() + 0.001 * (energies ** 2).mean()
            else:
                # Only anomaly samples (rare)
                loss = torch.relu(2.0 - energies[is_anomaly]).mean() + 0.001 * (energies ** 2).mean()

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count > 10:
                    print(f"⚠️  Too many NaN/Inf, stopping energy training at epoch {epoch+1}")
                    return False
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(energy_detector.parameters(), config.ENERGY_GRADIENT_CLIP)
            optimizer.step()

            epoch_losses.append(loss.item())

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            if epoch % 5 == 0:
                print(f"Energy Epoch {epoch+1}/{config.ENERGY_EPOCHS}: Loss = {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss

    print(f"✓ Energy detector trained successfully (best loss: {best_loss:.4f})")
    return True


def _find_best_threshold_for_component(scores, gt, name, n_steps=300):
    """Find the threshold that maximises F1 for a single score vector."""
    best_f1, best_t, best_m = 0, np.median(scores), {'p': 0, 'r': 0, 'f1': 0}
    lo, hi = np.percentile(scores, 50), np.percentile(scores, 99.9)
    for t in np.linspace(lo, hi, n_steps):
        pred = scores > t
        tp = np.sum(pred & (gt == 1))
        fp = np.sum(pred & (gt == 0))
        fn = np.sum(~pred & (gt == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_m = {'p': p, 'r': r, 'f1': f1}
    print(f"    {name:12s}: best F1={best_m['f1']:.3f}  P={best_m['p']:.3f}  R={best_m['r']:.3f}  thr={best_t:.4f}")
    return best_t, best_m


def tune_threshold_on_validation(model, recon_detector, energy_detector, clustering, val_data, val_gt, config):
    """
    Per-component OR-ensemble threshold tuning.

    Instead of mixing scores into one number (which destroys separation when
    individual distributions overlap), we find the optimal threshold for EACH
    component independently and flag a sample as anomalous if ANY component
    exceeds its own threshold.  We then do a joint grid-search over the three
    thresholds to maximise combined F1.
    """
    print("\n" + "="*60)
    print("Tuning Threshold on Validation Set")
    print("="*60)

    model.eval()
    val_tensor = torch.FloatTensor(val_data).to(config.DEVICE)

    # ---- Collect raw scores ----
    with torch.no_grad():
        recon_scores, _ = recon_detector.predict(val_tensor)
        recon_scores = recon_scores.cpu().numpy() if torch.is_tensor(recon_scores) else recon_scores

        embeddings = model.get_embeddings(val_tensor)
        embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings

        cluster_labels = clustering.predict(embeddings_np)
        cluster_scores = clustering.compute_cluster_anomaly_scores(embeddings_np, cluster_labels)

        energy_scores = None
        if config.USE_HYBRID and energy_detector is not None:
            energy_scores_t = energy_detector(embeddings, cluster_labels=None)
            energy_scores = energy_scores_t.cpu().numpy() if torch.is_tensor(energy_scores_t) else energy_scores_t

    # ---- Per-component diagnostics ----
    normal_mask = val_gt == 0
    anomaly_mask = val_gt == 1
    n_normal = int(normal_mask.sum())
    n_anomaly = int(anomaly_mask.sum())
    print(f"\n  Validation: {n_normal} normal, {n_anomaly} anomalous")

    components = {'recon': recon_scores, 'cluster': cluster_scores}
    if energy_scores is not None:
        components['energy'] = energy_scores

    print("\n  Per-component score distributions:")
    for name, sc in components.items():
        if n_normal > 0 and n_anomaly > 0:
            nm, am = sc[normal_mask].mean(), sc[anomaly_mask].mean()
            ns, asd = sc[normal_mask].std(), sc[anomaly_mask].std()
            sep = (am - nm) / (0.5 * (ns + asd) + 1e-8)
            print(f"    {name:12s}: normal={nm:.4f}±{ns:.4f}  anomaly={am:.4f}±{asd:.4f}  "
                  f"d'={sep:.3f}")

    # ---- Find best per-component threshold ----
    print("\n  Per-component best F1:")
    comp_thresholds = {}
    comp_metrics = {}
    for name, sc in components.items():
        t, m = _find_best_threshold_for_component(sc, val_gt, name)
        comp_thresholds[name] = t
        comp_metrics[name] = m

    # ---- OR-ensemble: joint grid search over multipliers ----
    # For each component, search thresholds around the best one found above.
    # A sample is anomalous if ANY component score exceeds its threshold.
    print("\n  OR-ensemble joint search...")

    best_f1 = 0
    best_thresholds = dict(comp_thresholds)
    best_metrics = {'precision': 0, 'recall': 0, 'f1': 0}

    # Build per-component search grids (narrow range around each best threshold)
    grids = {}
    for name, sc in components.items():
        center = comp_thresholds[name]
        sc_range = np.percentile(sc, 99) - np.percentile(sc, 50)
        lo = center - 0.3 * sc_range
        hi = center + 0.3 * sc_range
        grids[name] = np.linspace(lo, hi, 20)

    comp_names = list(components.keys())
    comp_arrays = [components[n] for n in comp_names]
    comp_grids = [grids[n] for n in comp_names]

    # Iterate over all combinations (20^2 or 20^3 = 400 or 8000 – very fast)
    from itertools import product as iter_product
    for thrs in iter_product(*comp_grids):
        # OR-ensemble: anomalous if ANY component exceeds its threshold
        pred = np.zeros(len(val_gt), dtype=bool)
        for sc_arr, t in zip(comp_arrays, thrs):
            pred |= (sc_arr > t)

        tp = np.sum(pred & (val_gt == 1))
        fp = np.sum(pred & (val_gt == 0))
        fn = np.sum(~pred & (val_gt == 1))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = dict(zip(comp_names, thrs))
            best_metrics = {'precision': float(p), 'recall': float(r), 'f1': float(f1)}

    # Also try the weighted-sum approach as a fallback (keep best of both)
    # Percentile normalisation + weighted sum
    recon_p5, recon_p95 = np.percentile(recon_scores, [5, 95])
    cluster_p5, cluster_p95 = np.percentile(cluster_scores, [5, 95])
    recon_norm = np.clip((recon_scores - recon_p5) / (recon_p95 - recon_p5 + 1e-8), 0, 1)
    cluster_norm = np.clip((cluster_scores - cluster_p5) / (cluster_p95 - cluster_p5 + 1e-8), 0, 1)
    if energy_scores is not None:
        energy_p5, energy_p95 = np.percentile(energy_scores, [5, 95])
        energy_norm = np.clip((energy_scores - energy_p5) / (energy_p95 - energy_p5 + 1e-8), 0, 1)
        ws = config.RECON_WEIGHT * recon_norm + config.CLUSTER_WEIGHT * cluster_norm + config.ENERGY_WEIGHT * energy_norm
    else:
        energy_p5, energy_p95 = 0, 1
        energy_norm = np.zeros_like(recon_norm)
        tw = config.RECON_WEIGHT + config.CLUSTER_WEIGHT
        ws = (config.RECON_WEIGHT / tw) * recon_norm + (config.CLUSTER_WEIGHT / tw) * cluster_norm
    combined_scores = ws  # For return / visualizations

    # Single-threshold search on weighted sum
    ws_t, ws_m = _find_best_threshold_for_component(ws, val_gt, 'weighted_sum')
    print(f"\n  OR-ensemble best:   F1={best_metrics['f1']:.3f}  P={best_metrics['precision']:.3f}  R={best_metrics['recall']:.3f}")
    print(f"  Weighted-sum best:  F1={ws_m['f1']:.3f}  P={ws_m['p']:.3f}  R={ws_m['r']:.3f}")

    # Pick whichever strategy won
    use_or_ensemble = best_metrics['f1'] >= ws_m['f1']
    if use_or_ensemble:
        print(f"  → Using OR-ensemble (better F1)")
        for n in comp_names:
            print(f"    {n} threshold = {best_thresholds[n]:.4f}")
    else:
        print(f"  → Using weighted-sum (better F1)")
        best_metrics = {'precision': ws_m['p'], 'recall': ws_m['r'], 'f1': ws_m['f1']}
        # Store weighted-sum threshold as a single 'combined' threshold
        best_thresholds = {'combined': ws_t}
        combined_scores = ws

    print(f"\n✓ Best Val F1: {best_metrics['f1']:.3f}, Precision: {best_metrics['precision']:.3f}, Recall: {best_metrics['recall']:.3f}")

    # Pack norm stats for test-set reuse
    norm_stats = {
        'recon_p5': recon_p5, 'recon_p95': recon_p95,
        'cluster_p5': cluster_p5, 'cluster_p95': cluster_p95,
        'energy_p5': energy_p5, 'energy_p95': energy_p95,
        'use_or_ensemble': use_or_ensemble,
        'comp_thresholds': best_thresholds,
    }

    # Return: threshold (single number for backward compat), metrics, scores, stats
    single_threshold = best_thresholds.get('combined', 0.5)
    return single_threshold, best_metrics, combined_scores, norm_stats


def save_training_plots(train_losses, val_losses, output_dir):
    """Save training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves")


def generate_thesis_visualizations(train_losses, val_losses, train_contrastive,
                                  train_reconstruction, anomaly_scores, predictions,
                                  ground_truth, tp, fp, fn, tn, precision, recall,
                                  f1, accuracy, output_dir):
    """
    Generate comprehensive visualizations for thesis
    """
    import seaborn as sns
    sns.set_style("whitegrid")

    # Create figures directory
    fig_dir = f"{output_dir}/thesis_figures"
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Training Loss Curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

    # Total loss
    axes[0, 0].plot(train_losses, label='Train', linewidth=2, color='#2E86AB')
    axes[0, 0].plot(val_losses, label='Validation', linewidth=2, color='#A23B72')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss', fontsize=12)
    axes[0, 0].set_title('Total Loss Over Epochs', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Contrastive loss
    axes[0, 1].plot(train_contrastive, linewidth=2, color='#F18F01')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Contrastive Loss', fontsize=12)
    axes[0, 1].set_title('Contrastive Learning Loss', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1, 0].plot(train_reconstruction, linewidth=2, color='#06A77D')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[1, 0].set_title('Reconstruction Loss', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Loss reduction percentage
    axes[1, 1].plot([((train_losses[0] - l) / train_losses[0]) * 100 for l in train_losses],
                    linewidth=2, color='#D62828')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss Reduction (%)', fontsize=12)
    axes[1, 1].set_title('Training Improvement', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% target')
    axes[1, 1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/1_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved training curves")

    # 2. Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                cbar_kws={'label': 'Count'}, ax=ax, annot_kws={"size": 16})

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(f'Confusion Matrix (F1={f1:.3f})', fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/2_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved confusion matrix")

    # 3. Performance Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [precision, recall, f1, accuracy]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Metrics', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Target (0.70)', alpha=0.7)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/3_performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved performance metrics")

    # 4. Anomaly Score Distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    axes[0].hist(anomaly_scores[ground_truth == 0], bins=50, alpha=0.7,
                 label='Normal', color='#06A77D', edgecolor='black')
    axes[0].hist(anomaly_scores[ground_truth == 1], bins=50, alpha=0.7,
                 label='Anomaly', color='#D62828', edgecolor='black')
    axes[0].set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Box plot
    box_data = [anomaly_scores[ground_truth == 0], anomaly_scores[ground_truth == 1]]
    bp = axes[1].boxplot(box_data, labels=['Normal', 'Anomaly'], patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Score Distribution by Class', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/4_anomaly_score_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved anomaly score distribution")

    # 5. ROC-style curve (Precision-Recall)
    from sklearn.metrics import precision_recall_curve, auc

    precision_curve, recall_curve, thresholds = precision_recall_curve(ground_truth, anomaly_scores)
    pr_auc = auc(recall_curve, precision_curve)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall_curve, precision_curve, linewidth=3, color='#2E86AB',
            label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax.fill_between(recall_curve, precision_curve, alpha=0.3, color='#2E86AB')

    # Mark current operating point
    ax.scatter([recall], [precision], s=200, c='red', marker='*',
               edgecolors='black', linewidths=2, zorder=5,
               label=f'Operating Point (P={precision:.3f}, R={recall:.3f})')

    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/5_precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved precision-recall curve")

    # 6. Detection Examples Timeline
    fig, ax = plt.subplots(figsize=(16, 5))

    # Show first 500 samples
    n_show = min(500, len(anomaly_scores))
    x = np.arange(n_show)

    ax.plot(x, anomaly_scores[:n_show], linewidth=1, color='gray', alpha=0.5, label='Anomaly Score')

    # Highlight true anomalies
    true_anomalies = np.where(ground_truth[:n_show] == 1)[0]
    if len(true_anomalies) > 0:
        ax.scatter(true_anomalies, anomaly_scores[true_anomalies],
                  c='red', s=100, marker='o', label='True Anomaly',
                  edgecolors='black', linewidths=1.5, zorder=5)

    # Highlight detected anomalies
    detected = np.where(predictions[:n_show] == 1)[0]
    if len(detected) > 0:
        ax.scatter(detected, anomaly_scores[detected],
                  c='blue', s=50, marker='^', label='Detected',
                  edgecolors='black', linewidths=1, zorder=4, alpha=0.7)

    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax.set_title('Anomaly Detection Timeline (First 500 Samples)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/6_detection_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved detection timeline")

    # 7. Summary Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], ax=ax1)
    ax1.set_title('Confusion Matrix', fontweight='bold')

    # Metrics
    ax2 = fig.add_subplot(gs[0, 1:])
    metrics_list = ['Precision', 'Recall', 'F1', 'Accuracy']
    values_list = [precision, recall, f1, accuracy]
    colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    bars = ax2.barh(metrics_list, values_list, color=colors_list, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, values_list):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontweight='bold')
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Performance Metrics', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Training curves
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(train_losses, label='Train Loss', linewidth=2, color='#2E86AB')
    ax3.plot(val_losses, label='Val Loss', linewidth=2, color='#A23B72')
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Loss', fontweight='bold')
    ax3.set_title('Training Progress', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Score distribution
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.hist(anomaly_scores[ground_truth == 0], bins=40, alpha=0.7,
            label='Normal', color='#06A77D', edgecolor='black')
    ax4.hist(anomaly_scores[ground_truth == 1], bins=40, alpha=0.7,
            label='Anomaly', color='#D62828', edgecolor='black')
    ax4.set_xlabel('Anomaly Score', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Score Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Stats text
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    stats_text = f"""
    STATISTICS
    ═══════════
    Total Samples: {len(ground_truth)}
    True Anomalies: {ground_truth.sum()}
    Detected: {predictions.sum()}
    
    PERFORMANCE
    ═══════════
    True Positives: {tp}
    False Positives: {fp}
    False Negatives: {fn}
    True Negatives: {tn}
    
    Precision: {precision:.3f}
    Recall: {recall:.3f}
    F1 Score: {f1:.3f}
    Accuracy: {accuracy:.3f}
    """
    ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')

    fig.suptitle('Anomaly Detection Results Dashboard', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(f"{fig_dir}/7_results_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved results dashboard")

    print(f"\n✓ All {7} thesis-ready visualizations saved to {fig_dir}/")
    return fig_dir


def main():
    """Main training pipeline"""
    print("="*80)
    print("IMPROVED FULL TRAINING - Production-Ready Anomaly Detection")
    print("Target: F1 > 70%, 100 epochs, stable energy detector, hybrid fusion")
    print("="*80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{ImprovedConfig.OUTPUT_DIR}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    # Save config
    config_dict = {k: v for k, v in vars(ImprovedConfig).items() if not k.startswith('_')}
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    # ========================================================================
    # [1] LOAD AND PREPARE DATA
    # ========================================================================
    print("\n[1/8] Loading and preparing data...")
    df = load_forex_data(ImprovedConfig.DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # Inject diverse anomalies
    df_with_anomalies, ground_truth = inject_diverse_anomalies(
        df,
        anomaly_ratio=ImprovedConfig.ANOMALY_RATIO,
        intensity=ImprovedConfig.ANOMALY_INTENSITY
    )

    # Preprocess — clip_outliers=False so injected anomalies survive into model input
    preprocessor = FinancialDataPreprocessor(
        window_size=ImprovedConfig.WINDOW_SIZE,
        stride=1,
        clip_outliers=False
    )

    sequences, feature_names = preprocessor.prepare_data(df_with_anomalies, fit_scaler=True)
    print(f"✓ Created {len(sequences)} sequences with {len(feature_names)} features")

    # ---- CRITICAL: Align ground truth with surviving indices ----
    # Preprocessing drops rows (NaN from technical indicators), so we must
    # re-index ground_truth to match the rows that actually survive.
    surviving_indices = preprocessor.surviving_indices_
    ground_truth_surviving = ground_truth[surviving_indices]

    # Use LAST-POINT labeling: a sequence is anomalous if the last point
    # in the window is anomalous.  The previous "ANY-in-window" approach
    # labelled almost every sequence as anomalous (98%+) because with 7%
    # point-level anomaly rate and window=60, the probability that a window
    # contains zero anomalies is only ~(0.93)^60 ≈ 1%.
    n_sequences = len(sequences)
    ground_truth_aligned = np.zeros(n_sequences, dtype=bool)
    for i in range(n_sequences):
        # Label by the last point of the window (index i + WINDOW_SIZE - 1)
        last_idx = i + ImprovedConfig.WINDOW_SIZE - 1
        if last_idx < len(ground_truth_surviving):
            ground_truth_aligned[i] = ground_truth_surviving[last_idx]

    print(f"  Ground truth alignment: {surviving_indices.shape[0]} surviving rows, "
          f"{ground_truth_aligned.sum()} anomalous sequences ({ground_truth_aligned.sum()/len(ground_truth_aligned)*100:.1f}%)")

    # Split
    n_samples = len(sequences)
    n_train = int(n_samples * ImprovedConfig.TRAIN_RATIO)
    n_val = int(n_samples * ImprovedConfig.VAL_RATIO)

    train_data = sequences[:n_train]
    val_data = sequences[n_train:n_train+n_val]
    test_data = sequences[n_train+n_val:]

    train_gt = ground_truth_aligned[:n_train]
    val_gt = ground_truth_aligned[n_train:n_train+n_val]
    test_gt = ground_truth_aligned[n_train+n_val:]

    print(f"✓ Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print(f"  Val anomalies: {val_gt.sum()} ({val_gt.sum()/len(val_gt)*100:.1f}%)")
    print(f"  Test anomalies: {test_gt.sum()} ({test_gt.sum()/len(test_gt)*100:.1f}%)")

    # Create loaders
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)
    test_tensor = torch.FloatTensor(test_data)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=ImprovedConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=ImprovedConfig.BATCH_SIZE, shuffle=False)

    # ========================================================================
    # [2] INITIALIZE MODEL
    # ========================================================================
    print("\n[2/8] Initializing model...")
    n_features = train_data.shape[2]

    model = SelfSupervisedTemporalModel(
        n_features=n_features,
        d_model=ImprovedConfig.D_MODEL,
        n_heads=ImprovedConfig.N_HEADS,
        n_layers=ImprovedConfig.N_LAYERS,
        dropout=ImprovedConfig.DROPOUT,
        contrastive_weight=ImprovedConfig.CONTRASTIVE_WEIGHT,
        reconstruction_weight=ImprovedConfig.RECONSTRUCTION_WEIGHT
    ).to(ImprovedConfig.DEVICE)

    print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ========================================================================
    # [3] TRAIN MAIN MODEL
    # ========================================================================
    print(f"\n[3/8] Training main model ({ImprovedConfig.N_EPOCHS} epochs)...")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=ImprovedConfig.LEARNING_RATE,
        weight_decay=ImprovedConfig.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=ImprovedConfig.N_EPOCHS,
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_contrastive = []
    train_reconstruction = []

    for epoch in range(ImprovedConfig.N_EPOCHS):
        # Train
        model.train()
        epoch_train_losses = []
        epoch_contrastive_losses = []
        epoch_reconstruction_losses = []

        for x, in train_loader:
            x = x.to(ImprovedConfig.DEVICE)

            optimizer.zero_grad()
            loss, losses = model(x, use_contrastive=True, use_reconstruction=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ImprovedConfig.GRADIENT_CLIP)
            optimizer.step()

            epoch_train_losses.append(loss.item())
            epoch_contrastive_losses.append(losses.get('contrastive_loss', 0))
            epoch_reconstruction_losses.append(losses.get('reconstruction_loss', 0))

        # Validate
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for x, in val_loader:
                x = x.to(ImprovedConfig.DEVICE)
                loss, _ = model(x, use_contrastive=True, use_reconstruction=True)
                epoch_val_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_contrastive.append(np.mean(epoch_contrastive_losses))
        train_reconstruction.append(np.mean(epoch_reconstruction_losses))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        scheduler.step()

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{output_dir}/checkpoints/best_model.pt")
            if (epoch + 1) % 10 == 0:
                print(f"  ✓ Saved best model (epoch {epoch+1})")
        else:
            patience_counter += 1
            if patience_counter >= ImprovedConfig.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best
    model.load_state_dict(torch.load(f"{output_dir}/checkpoints/best_model.pt"))
    print("✓ Training complete, loaded best model")

    if ImprovedConfig.SAVE_PLOTS:
        save_training_plots(train_losses, val_losses, output_dir)

    # ========================================================================
    # [4] CLUSTERING
    # ========================================================================
    print("\n[4/8] Performing clustering...")

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        train_embeddings = model.get_embeddings(train_tensor.to(ImprovedConfig.DEVICE)).cpu().numpy()

    # Cluster
    clustering = DensityAwareClustering(
        n_clusters=ImprovedConfig.N_CLUSTERS,
        min_cluster_size=ImprovedConfig.MIN_CLUSTER_SIZE
    )
    clustering.fit(train_embeddings)
    cluster_labels = clustering.labels_

    print(f"✓ Created {len(np.unique(cluster_labels))} clusters")
    print(f"  Cluster sizes: {np.bincount(cluster_labels)}")

    # Diagnostic: Check cluster-anomaly separation quality
    normal_cluster_mask = cluster_labels < ImprovedConfig.N_CLUSTERS - 1
    outlier_cluster_mask = ~normal_cluster_mask

    train_gt_anomaly_in_normal = train_gt[normal_cluster_mask].sum()
    train_gt_anomaly_in_outlier = train_gt[outlier_cluster_mask].sum()
    total_train_anomalies = train_gt.sum()

    if total_train_anomalies > 0:
        outlier_cluster_capture = train_gt_anomaly_in_outlier / total_train_anomalies * 100
        print(f"  Cluster separation quality:")
        print(f"    Anomalies in outlier cluster: {train_gt_anomaly_in_outlier}/{int(total_train_anomalies)} ({outlier_cluster_capture:.1f}%)")
        if outlier_cluster_capture < 20:
            print(f"    ⚠️ Low capture rate - energy detector may struggle to learn")

    cluster_labels_tensor = torch.LongTensor(cluster_labels).to(ImprovedConfig.DEVICE)

    # ========================================================================
    # [5] RECONSTRUCTION DETECTOR
    # ========================================================================
    print("\n[5/8] Fitting reconstruction detector...")

    # Build feature weights: emphasize price-sensitive features that anomalies affect
    price_sensitive = {'close', 'open', 'high', 'low', 'returns', 'log_returns',
                       'high_low_range', 'close_open_range', 'atr', 'atr_pct'}
    feature_weights = []
    for fname in feature_names:
        if fname.lower() in price_sensitive:
            feature_weights.append(3.0)
        elif any(slow in fname.lower() for slow in ['sma', 'ema', 'adx', 'bb_']):
            feature_weights.append(0.5)
        else:
            feature_weights.append(1.0)
    feature_weights_tensor = torch.FloatTensor(feature_weights)
    print(f"  Feature weights: {dict(zip(feature_names, feature_weights))}")

    recon_detector = ReconstructionBasedDetector(
        reconstructor=model.reconstructor,
        threshold_percentile=97,  # Higher threshold to reduce false positives
        feature_weights=feature_weights_tensor
    )
    recon_detector.fit(train_tensor.to(ImprovedConfig.DEVICE))
    print("✓ Reconstruction detector fitted")

    # ========================================================================
    # [6] ENERGY DETECTOR (STABLE)
    # ========================================================================
    energy_detector = None
    if ImprovedConfig.USE_ENERGY_DETECTOR:
        print("\n[6/8] Training stable energy detector...")
        energy_detector = EnergyBasedAnomalyDetector(
            embedding_dim=ImprovedConfig.D_MODEL,
            n_clusters=ImprovedConfig.N_CLUSTERS
        ).to(ImprovedConfig.DEVICE)

        success = train_energy_detector_stable(
            energy_detector,
            train_tensor,
            train_gt,
            model,
            ImprovedConfig
        )

        if not success:
            print("⚠️  Energy detector training failed, using reconstruction only")
            energy_detector = None
    else:
        print("\n[6/8] Skipping energy detector (reconstruction only)")

    # ========================================================================
    # [7] THRESHOLD TUNING
    # ========================================================================
    print("\n[7/8] Tuning threshold on validation set...")
    best_threshold, val_metrics, val_combined_scores, val_norm_stats = tune_threshold_on_validation(
        model,
        recon_detector,
        energy_detector,
        clustering,
        val_data,
        val_gt,
        ImprovedConfig
    )

    # ========================================================================
    # [8] FINAL TESTING
    # ========================================================================
    print("\n[8/8] Testing on test set...")
    model.eval()
    test_tensor_gpu = test_tensor.to(ImprovedConfig.DEVICE)

    with torch.no_grad():
        # Reconstruction scores
        recon_scores, _ = recon_detector.predict(test_tensor_gpu)
        recon_scores = recon_scores.cpu().numpy() if torch.is_tensor(recon_scores) else recon_scores

        # Get embeddings for clustering and energy
        embeddings = model.get_embeddings(test_tensor_gpu)
        embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings

        # Cluster-based anomaly scores
        test_cluster_labels = clustering.predict(embeddings_np)
        cluster_scores = clustering.compute_cluster_anomaly_scores(embeddings_np, test_cluster_labels)

        energy_scores = None
        if ImprovedConfig.USE_HYBRID and energy_detector is not None:
            energy_scores_t = energy_detector(embeddings, cluster_labels=None)
            energy_scores = energy_scores_t.detach().cpu().numpy() if torch.is_tensor(energy_scores_t) else energy_scores_t

    # ---- Determine predictions using same strategy as validation ----
    use_or = val_norm_stats.get('use_or_ensemble', False)
    comp_thresholds = val_norm_stats.get('comp_thresholds', {})

    if use_or:
        # OR-ensemble: anomalous if ANY component exceeds its threshold
        predictions = np.zeros(len(test_gt), dtype=bool)
        components = {'recon': recon_scores, 'cluster': cluster_scores}
        if energy_scores is not None:
            components['energy'] = energy_scores

        for name, sc in components.items():
            if name in comp_thresholds:
                predictions |= (sc > comp_thresholds[name])

        detection_method = "OR-Ensemble (Reconstruction | Cluster | Energy)"
        # For visualisations, build a combined score (max of per-component z-like scores)
        final_scores = np.maximum.reduce([
            (recon_scores - recon_scores.mean()) / (recon_scores.std() + 1e-8),
            (cluster_scores - cluster_scores.mean()) / (cluster_scores.std() + 1e-8),
        ] + ([
            (energy_scores - energy_scores.mean()) / (energy_scores.std() + 1e-8)
        ] if energy_scores is not None else []))
    else:
        # Weighted-sum with single threshold (fallback)
        recon_p5 = val_norm_stats['recon_p5']
        recon_p95 = val_norm_stats['recon_p95']
        cluster_p5 = val_norm_stats['cluster_p5']
        cluster_p95 = val_norm_stats['cluster_p95']

        recon_norm = np.clip((recon_scores - recon_p5) / (recon_p95 - recon_p5 + 1e-8), 0, 1)
        cluster_norm = np.clip((cluster_scores - cluster_p5) / (cluster_p95 - cluster_p5 + 1e-8), 0, 1)

        if energy_scores is not None:
            energy_p5 = val_norm_stats['energy_p5']
            energy_p95 = val_norm_stats['energy_p95']
            energy_norm = np.clip((energy_scores - energy_p5) / (energy_p95 - energy_p5 + 1e-8), 0, 1)
            final_scores = (ImprovedConfig.RECON_WEIGHT * recon_norm +
                           ImprovedConfig.CLUSTER_WEIGHT * cluster_norm +
                           ImprovedConfig.ENERGY_WEIGHT * energy_norm)
            detection_method = "Hybrid (Reconstruction + Cluster + Energy)"
        else:
            tw = ImprovedConfig.RECON_WEIGHT + ImprovedConfig.CLUSTER_WEIGHT
            final_scores = ((ImprovedConfig.RECON_WEIGHT / tw) * recon_norm +
                           (ImprovedConfig.CLUSTER_WEIGHT / tw) * cluster_norm)
            detection_method = "Hybrid (Reconstruction + Cluster)"

        predictions = final_scores > best_threshold

    # Compute metrics
    tp = np.sum((predictions == True) & (test_gt == True))
    fp = np.sum((predictions == True) & (test_gt == False))
    fn = np.sum((predictions == False) & (test_gt == True))
    tn = np.sum((predictions == False) & (test_gt == False))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(predictions)

    # ========================================================================
    # RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Detection Method: {detection_method}")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"\nTest Performance:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TN: {tn:4d}")
    print("="*80)

    # Save results
    results = {
        'test': {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        },
        'validation': val_metrics,
        'training': {
            'n_epochs': epoch + 1,
            'best_val_loss': float(best_val_loss),
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1])
        },
        'threshold': float(best_threshold),
        'detection_method': detection_method,
        'timestamp': timestamp
    }

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save model and detectors
    torch.save({
        'model_state_dict': model.state_dict(),
        'energy_detector_state_dict': energy_detector.state_dict() if energy_detector else None,
        'results': results,
        'config': config_dict,
        'threshold': best_threshold
    }, f"{output_dir}/final_model.pt")

    # Save predictions
    if ImprovedConfig.SAVE_PREDICTIONS:
        pred_df = pd.DataFrame({
            'anomaly_score': final_scores,
            'prediction': predictions,
            'ground_truth': test_gt,
            'correct': predictions == test_gt
        })
        pred_df.to_csv(f"{output_dir}/predictions.csv", index=False)
        print(f"\n✓ Saved predictions to {output_dir}/predictions.csv")

    # ========================================================================
    # VISUALIZATIONS FOR THESIS
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS FOR THESIS")
    print("="*80)

    generate_thesis_visualizations(
        train_losses, val_losses, train_contrastive, train_reconstruction,
        final_scores, predictions, test_gt, tp, fp, fn, tn,
        precision, recall, f1, accuracy, output_dir
    )

    # ========================================================================
    # DETAILED EXCEL REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING DETAILED EXCEL REPORT")
    print("="*80)

    generate_detailed_results_excel(f"{output_dir}/results.json", output_dir)

    print(f"\n✓ All results, visualizations, and Excel report saved to {output_dir}/")

    # Performance assessment
    print("\n" + "="*80)
    if f1 >= 0.70:
        print("🎉 EXCELLENT! F1 ≥ 70% - Target achieved!")
    elif f1 >= 0.60:
        print("✅ GOOD! F1 ≥ 60% - Strong performance")
    elif f1 >= 0.50:
        print("✓ ACCEPTABLE - F1 ≥ 50%")
    else:
        print("⚠️  NEEDS IMPROVEMENT - F1 < 50%")
        print("\nSuggestions:")
        print("  1. Train for more epochs (150-200)")
        print("  2. Adjust anomaly intensity (try 2.5-3.0)")
        print("  3. Tune RECON_WEIGHT vs ENERGY_WEIGHT")
        print("  4. Increase model capacity (D_MODEL=256)")
    print("="*80)


if __name__ == '__main__':
    main()


