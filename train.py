"""
IMPROVED FULL TRAINING - Production-Ready Anomaly Detection
100 epochs, stable energy detector, hybrid fusion, F1 > 70% target
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#git test
# Add current directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.temporal_transformer import SelfSupervisedTemporalModel
from models.clustering import DensityAwareClustering, LatentSpaceRegularizer
from models.anomaly_detector import (
    EnergyBasedAnomalyDetector,
    ReconstructionBasedDetector
)
from data.preprocessing import FinancialDataPreprocessor, load_forex_data

# Import Excel generation
from report_generator import generate_detailed_results_excel

# Publication figure style
import figs_style as FS


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

    # Model Architecture - Small to prevent overfitting on ~14k samples
    D_MODEL = 64   # Smaller → less memorization, better generalization
    N_HEADS = 4
    N_LAYERS = 2   # 2 layers sufficient for 60-step sequences
    DROPOUT = 0.25  # Higher dropout for regularization
    MASK_RATIO = 0.40  # Higher mask ratio forces learning structure, not memorizing

    # Training - Extended with better convergence
    N_EPOCHS = 150  # More epochs for better convergence
    BATCH_SIZE = 64  # Larger batch for stability
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-3  # Stronger regularization to close train/val gap
    GRADIENT_CLIP = 1.0  # Standard clipping

    # Loss weights - Reconstruction-focused for anomaly detection
    CONTRASTIVE_WEIGHT = 0.05  # Low - contrastive is secondary
    RECONSTRUCTION_WEIGHT = 1.0

    # Energy detector - More training with better hyperparameters
    USE_ENERGY_DETECTOR = True
    ENERGY_EPOCHS = 150  # Extended training for deeper energy network
    ENERGY_LR = 3e-4  # Moderate LR to prevent NaN with per-sample hinge loss
    ENERGY_GRADIENT_CLIP = 1.0  # Wider clip to prevent gradient starvation
    ENERGY_WEIGHT_DECAY = 1e-5  # Less weight decay for energy detector

    # Clustering - Fewer, more meaningful clusters
    N_CLUSTERS = 8  # Reduced for clearer separation
    MIN_CLUSTER_SIZE = 100  # Require larger clusters

    # Latent Space Regularization (center loss + separation loss after clustering)
    REGULARIZATION_EPOCHS = 10  # Fine-tune with regularization after clustering
    REGULARIZATION_LR = 1e-4  # Lower LR for fine-tuning
    REGULARIZATION_WEIGHT = 0.1  # Weight of center/separation loss

    # Hybrid Detection - Reconstruction + Energy
    # Energy consistently has highest d' (~0.86); recon varies (0.5-0.9)
    USE_HYBRID = True
    ENERGY_WEIGHT = 0.50  # Energy detector (d'≈0.86, most consistent signal)
    RECON_WEIGHT = 0.50   # Reconstruction (d' varies 0.5-0.9)
    CLUSTER_WEIGHT = 0.00 # DISABLED - cluster d' is low/negative

    # Precision constraint for threshold tuning - low floor to let F1 optimizer work freely
    MIN_PRECISION = 0.20  # Low floor — F1 itself penalizes bad precision

    # Anomaly Injection - STRONG intensity so anomalies survive scaling
    ANOMALY_RATIO = 0.05  # 5% anomalies (×3 timesteps → ~15% sequence rate)
    ANOMALY_INTENSITY = 12.0  # Strong - must survive RobustScaler + clipping
    ANOMALY_WINDOW = 3  # Affect 3 consecutive timesteps per anomaly

    # Threshold Tuning - Wide search for optimal F1 balance
    USE_VALIDATION_TUNING = True
    THRESHOLD_SEARCH_STEPS = 300  # More steps for finer search
    THRESHOLD_PERCENTILE_MIN = 60  # Start lower to find recall-boosting thresholds
    THRESHOLD_PERCENTILE_MAX = 99.9

    # Ensemble Detection
    USE_ISOLATION_FOREST = False  # DISABLED - adds noise, hurts precision
    ISOLATION_CONTAMINATION = 0.05  # Expected contamination rate

    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR = 'improved_outputs'
    EARLY_STOPPING_PATIENCE = 40  # Allow more patience for slow convergence

    # Reporting
    SAVE_PLOTS = True
    SAVE_PREDICTIONS = True


def inject_diverse_anomalies(data, anomaly_ratio=0.05, intensity=2.0):
    """
    Inject diverse, realistic financial anomalies with STRONG signals.
    Each anomaly affects ANOMALY_WINDOW consecutive timesteps so the signal
    survives windowing (60-step windows with last-point labeling).
    Uses a dedicated RNG for reproducibility across runs.
    """
    # Dedicated RNG ensures identical anomaly placement regardless of
    # upstream random state drift from data loading / feature engineering
    rng = np.random.RandomState(SEED)

    n_samples = len(data)
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_window = getattr(ImprovedConfig, 'ANOMALY_WINDOW', 3)

    price_std = data['close'].std()
    volume_std = data['tick_volume'].std()

    # Use rolling statistics for more context-aware anomalies
    rolling_std = data['close'].rolling(window=20).std().fillna(price_std)

    anomaly_mask = np.zeros(n_samples, dtype=bool)
    data_modified = data.copy()

    # Convert tick_volume to float to allow decimal values from multiplication
    if 'tick_volume' in data_modified.columns:
        data_modified['tick_volume'] = data_modified['tick_volume'].astype(float)

    # Ensure anomalies don't overlap: space them at least anomaly_window apart
    safe_indices = np.arange(ImprovedConfig.WINDOW_SIZE, n_samples - ImprovedConfig.WINDOW_SIZE - anomaly_window)
    if len(safe_indices) < n_anomalies:
        n_anomalies = len(safe_indices) // 2

    # Select non-overlapping indices
    all_candidates = rng.permutation(safe_indices)
    anomaly_indices = []
    used = set()
    for idx in all_candidates:
        if len(anomaly_indices) >= n_anomalies:
            break
        # Check no overlap with existing anomalies
        if any(abs(idx - u) < anomaly_window + 1 for u in used):
            continue
        anomaly_indices.append(idx)
        used.add(idx)
    anomaly_indices = np.array(anomaly_indices)

    anomaly_types = []
    for idx in anomaly_indices:
        # Diverse anomaly types with weighted distribution
        anomaly_type = rng.choice([
            'price_spike', 'volatility_spike', 'volume_spike',
            'trend_break', 'flash_crash', 'gap_anomaly'
        ], p=[0.2, 0.2, 0.15, 0.15, 0.2, 0.1])
        anomaly_types.append(anomaly_type)

        # Use local volatility for context-aware anomalies
        local_std = rolling_std.iloc[idx] if not np.isnan(rolling_std.iloc[idx]) else price_std

        # Apply anomaly to a window of consecutive timesteps
        for offset in range(anomaly_window):
            t = idx + offset
            if t >= n_samples:
                break

            # Scale intensity slightly for each timestep in the window
            t_intensity = intensity * (1.0 - 0.075 * offset)  # Gentle decay: 1.0, 0.925, 0.85

            if anomaly_type == 'price_spike':
                # Strong sudden price jump
                multiplier = rng.uniform(t_intensity, t_intensity + 3.0)
                direction = rng.choice([-1, 1])
                spike = local_std * multiplier * direction

                data_modified.iloc[t, data_modified.columns.get_loc('close')] += spike
                data_modified.iloc[t, data_modified.columns.get_loc('high')] = max(
                    data_modified.iloc[t]['high'],
                    data_modified.iloc[t]['close'] + abs(spike) * 0.3
                )
                data_modified.iloc[t, data_modified.columns.get_loc('low')] = min(
                    data_modified.iloc[t]['low'],
                    data_modified.iloc[t]['close'] - abs(spike) * 0.3
                )

            elif anomaly_type == 'volatility_spike':
                # Extreme volatility (much larger range)
                multiplier = rng.uniform(t_intensity + 2.0, t_intensity + 5.0)
                base_range = data_modified.iloc[t]['high'] - data_modified.iloc[t]['low']
                new_range = base_range * multiplier

                mid = (data_modified.iloc[t]['high'] + data_modified.iloc[t]['low']) / 2
                data_modified.iloc[t, data_modified.columns.get_loc('high')] = mid + new_range / 2
                data_modified.iloc[t, data_modified.columns.get_loc('low')] = mid - new_range / 2

            elif anomaly_type == 'volume_spike':
                # Very unusual volume + correlated price movement
                # tick_volume gets dropped in preprocessing, so we MUST also
                # modify price features to make this anomaly type visible.
                multiplier = rng.uniform(t_intensity + 5.0, t_intensity + 15.0)
                data_modified.iloc[t, data_modified.columns.get_loc('tick_volume')] *= multiplier
                # Add correlated price impact (high volume → price movement + wider range)
                price_shift = local_std * t_intensity * 0.8 * rng.choice([-1, 1])
                data_modified.iloc[t, data_modified.columns.get_loc('close')] += price_shift
                range_expansion = abs(price_shift) * 0.5
                data_modified.iloc[t, data_modified.columns.get_loc('high')] += range_expansion
                data_modified.iloc[t, data_modified.columns.get_loc('low')] -= range_expansion

            elif anomaly_type == 'trend_break':
                # Strong sudden reversal
                window_start = max(0, t-5)
                mean_price = data_modified.iloc[window_start:t]['close'].mean()
                deviation = local_std * t_intensity * 2.0
                new_price = mean_price + deviation if rng.rand() > 0.5 else mean_price - deviation
                data_modified.iloc[t, data_modified.columns.get_loc('close')] = new_price
                data_modified.iloc[t, data_modified.columns.get_loc('open')] = mean_price

            elif anomaly_type == 'flash_crash':
                # Quick severe drop and partial recovery
                crash_depth = local_std * t_intensity * 3.0
                data_modified.iloc[t, data_modified.columns.get_loc('low')] -= crash_depth
                data_modified.iloc[t, data_modified.columns.get_loc('close')] -= crash_depth * 0.7
                data_modified.iloc[t, data_modified.columns.get_loc('open')] -= crash_depth * 0.2

            elif anomaly_type == 'gap_anomaly':
                # Gap up/down from previous close — also shift close so the
                # most heavily-weighted feature (close, weight=3.0) sees it
                if t > 0:
                    prev_close = data_modified.iloc[t-1]['close']
                    gap = local_std * t_intensity * 2.5 * rng.choice([-1, 1])
                    data_modified.iloc[t, data_modified.columns.get_loc('open')] = prev_close + gap
                    data_modified.iloc[t, data_modified.columns.get_loc('close')] = prev_close + gap * 0.6
                    data_modified.iloc[t, data_modified.columns.get_loc('high')] = max(
                        data_modified.iloc[t]['high'], prev_close + abs(gap)
                    )
                    data_modified.iloc[t, data_modified.columns.get_loc('low')] = min(
                        data_modified.iloc[t]['low'], prev_close - abs(gap) * 0.3
                    )

            anomaly_mask[t] = True

    # Build per-sample anomaly type map (index -> type string)
    anomaly_type_map = {}
    for idx, atype in zip(anomaly_indices, anomaly_types):
        for offset in range(anomaly_window):
            t = idx + offset
            if t < n_samples:
                anomaly_type_map[t] = atype

    print(f"✓ Injected {len(anomaly_indices)} diverse anomalies ({anomaly_ratio*100:.1f}%)")
    print(f"  Each affects {anomaly_window} consecutive timesteps → {anomaly_mask.sum()} anomalous points")
    print(f"  Types: {dict(pd.Series(anomaly_types).value_counts())}")
    return data_modified, anomaly_mask, anomaly_type_map


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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.ENERGY_EPOCHS, eta_min=1e-6
    )

    embedder.eval()
    energy_detector.train()

    best_loss = float('inf')

    for epoch in range(config.ENERGY_EPOCHS):
        epoch_losses = []
        epoch_nan_count = 0  # Reset per epoch to avoid cumulative false-abort

        # Warm up margin: start at 2.0, ramp to 5.0 over first 30 epochs
        # Prevents huge gradients early when means are still close
        margin = min(2.0 + 3.0 * epoch / 30.0, 5.0)

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
                normal_energy = torch.clamp(normal_energy, -15, 15)
                anomaly_energy = torch.clamp(anomaly_energy, -15, 15)

                # Detach means to prevent double-gradient through both
                # the per-sample term AND the mean — this was causing NaN
                normal_mean = normal_energy.mean().detach()
                anomaly_mean = anomaly_energy.mean().detach()

                # 1. Per-sample hinge loss: each anomaly must individually exceed
                #    normal mean by margin. Each normal must individually be below
                #    anomaly mean minus margin. This prevents overlapping tails.
                per_anomaly_hinge = torch.relu(margin + normal_mean - anomaly_energy).mean()
                per_normal_hinge = torch.relu(margin + normal_energy - anomaly_mean).mean()

                # 2. Variance penalty: compress both distributions' tails
                #    Scale down to prevent gradient explosion (was 0.15, now 0.05)
                variance_penalty = 0.05 * (normal_energy.var() + anomaly_energy.var())

                # 3. Push targets: normal → low energy, anomaly → high energy
                normal_push = torch.relu(normal_energy - (-2.0)).mean()
                anomaly_push = torch.relu(5.0 - anomaly_energy).mean()

                # 4. Focal loss for hard examples near decision boundary
                #    Clamp sigmoid input to [-10, 10] and output to [eps, 1-eps]
                #    to prevent log(0) in BCE
                clamped_logits = torch.clamp(energies - 1.5, -10, 10)
                anomaly_prob = torch.sigmoid(clamped_logits).clamp(1e-6, 1 - 1e-6)
                bce_raw = nn.functional.binary_cross_entropy(anomaly_prob, labels, reduction='none')
                pt = torch.where(labels == 1, anomaly_prob, 1 - anomaly_prob)
                alpha_t = torch.where(labels == 1, torch.tensor(0.75), torch.tensor(0.25)).to(energies.device)
                focal_weight = alpha_t * (1 - pt) ** 2.0
                focal_loss = (focal_weight * bce_raw).mean()

                # 5. L2 regularization
                reg_loss = 0.001 * (energies ** 2).mean()

                loss = (per_anomaly_hinge + per_normal_hinge +
                        variance_penalty +
                        0.3 * normal_push + 0.3 * anomaly_push +
                        1.5 * focal_loss + reg_loss)
            elif is_normal.sum() > 0:
                # Only normal samples in this batch
                loss = torch.relu(energies[is_normal]).mean() + 0.001 * (energies ** 2).mean()
            else:
                # Only anomaly samples (rare)
                loss = torch.relu(2.0 - energies[is_anomaly]).mean() + 0.001 * (energies ** 2).mean()

            # Check for NaN — per-epoch counter to avoid cumulative false-abort
            if torch.isnan(loss) or torch.isinf(loss):
                epoch_nan_count += 1
                if epoch_nan_count > 20:
                    print(f"⚠️  Too many NaN/Inf in epoch {epoch+1}, stopping energy training")
                    return epoch > 5  # Return True if we trained for at least a few epochs
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

        scheduler.step()

    print(f"✓ Energy detector trained successfully (best loss: {best_loss:.4f})")
    return True


def _find_best_threshold_for_component(scores, gt, name, n_steps=500, min_precision=0.0):
    """Find the threshold that maximises F1 for a single score vector."""
    best_f1, best_t, best_m = 0, np.median(scores), {'p': 0, 'r': 0, 'f1': 0}
    lo, hi = np.percentile(scores, 20), np.percentile(scores, 99.9)
    for t in np.linspace(lo, hi, n_steps):
        pred = scores > t
        tp = np.sum(pred & (gt == 1))
        fp = np.sum(pred & (gt == 0))
        fn = np.sum(~pred & (gt == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1 and p >= min_precision:
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

        # Regime transition scores: detect rapid cluster switches in consecutive sequences
        regime_transition_scores = clustering.compute_regime_transition_scores(cluster_labels)

        energy_scores = None
        if config.USE_HYBRID and energy_detector is not None:
            # Pass cluster labels for cluster-conditioned energy scoring
            cluster_labels_t = torch.LongTensor(cluster_labels).to(config.DEVICE)
            energy_scores_t = energy_detector(embeddings, cluster_labels=cluster_labels_t)
            energy_scores = energy_scores_t.cpu().numpy() if torch.is_tensor(energy_scores_t) else energy_scores_t

    # ---- Per-component diagnostics ----
    normal_mask = val_gt == 0
    anomaly_mask = val_gt == 1
    n_normal = int(normal_mask.sum())
    n_anomaly = int(anomaly_mask.sum())
    print(f"\n  Validation: {n_normal} normal, {n_anomaly} anomalous")

    components = {'recon': recon_scores, 'cluster': cluster_scores, 'regime': regime_transition_scores}
    if energy_scores is not None:
        components['energy'] = energy_scores

    print("\n  Per-component score distributions:")
    for name, sc in components.items():
        if n_normal > 0 and n_anomaly > 0:
            nm, am = sc[normal_mask].mean(), sc[anomaly_mask].mean()
            ns, asd = sc[normal_mask].std(), sc[anomaly_mask].std()
            sep = FS.d_prime(sc[anomaly_mask], sc[normal_mask])
            print(f"    {name:12s}: normal={nm:.4f}±{ns:.4f}  anomaly={am:.4f}±{asd:.4f}  "
                  f"d'={sep:.3f}")

    # ---- Find best per-component threshold ----
    print("\n  Per-component best F1:")
    comp_thresholds = {}
    comp_metrics = {}
    comp_dprime = {}
    for name, sc in components.items():
        t, m = _find_best_threshold_for_component(sc, val_gt, name)
        comp_thresholds[name] = t
        comp_metrics[name] = m
        # Gate d': this formula and the > 0.5 threshold at line ~515 were
        # calibrated together.  It intentionally uses the arithmetic-mean-of-stds
        # denominator (not the corrected pooled-variance d' used for reporting).
        # Do NOT swap in the corrected formula without recalibrating the cutoff.
        if n_normal > 0 and n_anomaly > 0:
            nm, am = sc[normal_mask].mean(), sc[anomaly_mask].mean()
            ns, asd = sc[normal_mask].std(), sc[anomaly_mask].std()
            comp_dprime[name] = (am - nm) / (0.5 * (ns + asd) + 1e-8)
        else:
            comp_dprime[name] = 0.0

    # ---- OR-ensemble: only use discriminative components (d' > 0.5) ----
    print("\n  OR-ensemble joint search...")
    discriminative = {n: sc for n, sc in components.items() if comp_dprime.get(n, 0) > 0.5}
    print(f"    Discriminative components (d'>0.5): {list(discriminative.keys())}")

    best_or_f1 = 0
    best_or_thresholds = {}
    best_or_metrics = {'precision': 0, 'recall': 0, 'f1': 0}

    if len(discriminative) >= 1:
        # Build per-component search grids — wide range to explore recall-boosting thresholds
        # OR-ensemble flags if ANY component exceeds its threshold, so lower thresholds
        # boost recall. We search well below the per-component best to find combos.
        or_grids = {}
        for name, sc in discriminative.items():
            center = comp_thresholds[name]
            sc_range = np.percentile(sc, 99) - np.percentile(sc, 10)
            lo = center - 1.5 * sc_range  # Go well below best threshold for recall
            hi = center + 0.5 * sc_range
            or_grids[name] = np.linspace(lo, hi, 60)  # 60 steps (3600 combos for 2 components)

        or_names = list(discriminative.keys())
        or_arrays = [discriminative[n] for n in or_names]
        or_grid_list = [or_grids[n] for n in or_names]

        from itertools import product as iter_product
        for thrs in iter_product(*or_grid_list):
            pred = np.zeros(len(val_gt), dtype=bool)
            for sc_arr, t in zip(or_arrays, thrs):
                pred |= (sc_arr > t)

            tp = np.sum(pred & (val_gt == 1))
            fp = np.sum(pred & (val_gt == 0))
            fn = np.sum(~pred & (val_gt == 1))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            # Require minimum precision of 0.3 for OR-ensemble to avoid pure-recall solutions
            if f1 > best_or_f1 and p >= max(config.MIN_PRECISION, 0.3):
                best_or_f1 = f1
                best_or_thresholds = dict(zip(or_names, thrs))
                best_or_metrics = {'precision': float(p), 'recall': float(r), 'f1': float(f1)}

    # ---- Weighted-sum approach ----
    # Percentile normalisation + weighted sum
    recon_p5, recon_p95 = np.percentile(recon_scores, [5, 95])
    cluster_p5, cluster_p95 = np.percentile(cluster_scores, [5, 95])
    recon_norm = np.clip((recon_scores - recon_p5) / (recon_p95 - recon_p5 + 1e-8), 0, 1)
    cluster_norm = np.clip((cluster_scores - cluster_p5) / (cluster_p95 - cluster_p5 + 1e-8), 0, 1)
    if energy_scores is not None:
        energy_p5, energy_p95 = np.percentile(energy_scores, [5, 95])
        energy_norm = np.clip((energy_scores - energy_p5) / (energy_p95 - energy_p5 + 1e-8), 0, 1)
        total_w = config.RECON_WEIGHT + config.CLUSTER_WEIGHT + config.ENERGY_WEIGHT
        ws = (config.RECON_WEIGHT/total_w) * recon_norm + (config.CLUSTER_WEIGHT/total_w) * cluster_norm + (config.ENERGY_WEIGHT/total_w) * energy_norm
    else:
        energy_p5, energy_p95 = 0, 1
        energy_norm = np.zeros_like(recon_norm)
        tw = config.RECON_WEIGHT + config.CLUSTER_WEIGHT
        ws = (config.RECON_WEIGHT / tw) * recon_norm + (config.CLUSTER_WEIGHT / tw) * cluster_norm
    combined_scores = ws

    # Search for F1-optimal threshold on weighted-sum with NO min_precision first
    # This finds the true F1-optimal point without bias toward high precision
    ws_t_free, ws_m_free = _find_best_threshold_for_component(
        ws, val_gt, 'ws_free', n_steps=1000, min_precision=0.0)
    # Also search with min_precision constraint
    ws_t_constr, ws_m_constr = _find_best_threshold_for_component(
        ws, val_gt, 'ws_constr', n_steps=1000, min_precision=config.MIN_PRECISION)

    # Pick the weighted-sum result with better F1
    if ws_m_free['f1'] > ws_m_constr['f1']:
        ws_t, ws_m = ws_t_free, ws_m_free
        print(f"    → Using unconstrained threshold (better F1)")
    else:
        ws_t, ws_m = ws_t_constr, ws_m_constr

    # Also try recon-only
    recon_t, recon_m = _find_best_threshold_for_component(
        recon_norm, val_gt, 'recon_only', n_steps=1000, min_precision=0.0)

    # ---- Cascade approach: recon first, energy rescues recon's misses ----
    # Key insight: recon has P≈0.9 but R≈0.6 — it misses ~40% of anomalies.
    # Energy can catch some of those misses. If we only apply energy to samples
    # that recon marks as normal, energy's FPs only come from that subset,
    # preserving recon's high precision while boosting recall.
    cascade_f1 = 0.0
    cascade_recon_t = 0.0
    cascade_energy_t = 0.0
    cascade_metrics = {'precision': 0, 'recall': 0, 'f1': 0}

    if energy_scores is not None and 'energy' in comp_dprime and comp_dprime['energy'] > 0.3:
        # Search over recon thresholds × energy thresholds
        recon_thresholds = np.linspace(
            np.percentile(recon_norm, 70), np.percentile(recon_norm, 99), 40)
        energy_thresholds = np.linspace(
            np.percentile(energy_scores, 80), np.percentile(energy_scores, 99.5), 40)

        for rt in recon_thresholds:
            recon_pred = recon_norm > rt
            for et in energy_thresholds:
                # Cascade: flag if recon says anomaly, OR if recon says normal but energy says anomaly
                energy_rescue = (~recon_pred) & (energy_scores > et)
                pred = recon_pred | energy_rescue

                tp = np.sum(pred & (val_gt == 1))
                fp = np.sum(pred & (val_gt == 0))
                fn = np.sum(~pred & (val_gt == 1))

                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

                if f1 > cascade_f1 and p >= 0.3:
                    cascade_f1 = f1
                    cascade_recon_t = rt
                    cascade_energy_t = et
                    cascade_metrics = {'precision': float(p), 'recall': float(r), 'f1': float(f1)}

        print(f"  Cascade best:       F1={cascade_metrics['f1']:.3f}  P={cascade_metrics['precision']:.3f}  R={cascade_metrics['recall']:.3f}")

    print(f"\n  OR-ensemble best:   F1={best_or_metrics['f1']:.3f}  P={best_or_metrics['precision']:.3f}  R={best_or_metrics['recall']:.3f}")
    print(f"  Weighted-sum best:  F1={ws_m['f1']:.3f}  P={ws_m['p']:.3f}  R={ws_m['r']:.3f}")
    print(f"  Recon-only best:    F1={recon_m['f1']:.3f}  P={recon_m['p']:.3f}  R={recon_m['r']:.3f}")

    # Pick whichever strategy has the best F1
    # Tie-breaking rules:
    # 1. If OR-ensemble and recon-only are within 0.02 F1, prefer higher recall
    #    (current bottleneck is FN, not FP)
    # 2. If OR-ensemble and weighted-sum are within 0.01, prefer OR-ensemble
    #    (it uses complementary signals from both components)
    or_percentiles = {}
    candidates = [
        ('or_ensemble', best_or_metrics['f1'], best_or_metrics, best_or_thresholds, True),
        ('weighted_sum', ws_m['f1'], {'precision': ws_m['p'], 'recall': ws_m['r'], 'f1': ws_m['f1']}, {'combined': ws_t}, False),
        ('recon_only', recon_m['f1'], {'precision': recon_m['p'], 'recall': recon_m['r'], 'f1': recon_m['f1']}, {'combined': recon_t, 'recon_only': True}, False),
    ]
    # Add cascade if it was searched
    if cascade_f1 > 0:
        candidates.append(
            ('cascade', cascade_metrics['f1'], cascade_metrics,
             {'cascade_recon_t': cascade_recon_t, 'cascade_energy_t': cascade_energy_t, 'cascade': True}, False)
        )
    candidates.sort(key=lambda x: x[1], reverse=True)
    winner_name, winner_f1, winner_metrics, winner_thresholds, winner_is_or = candidates[0]

    # If recon-only won but OR-ensemble is within 0.02 F1 and has higher recall,
    # prefer OR-ensemble (it catches anomalies that recon misses)
    if winner_name == 'recon_only' and best_or_metrics['f1'] >= winner_f1 - 0.02:
        if best_or_metrics['recall'] > winner_metrics.get('recall', winner_metrics.get('r', 0)) + 0.02:
            print(f"  → Recon-only ({winner_f1:.3f}) and OR-ensemble ({best_or_metrics['f1']:.3f}) within 0.02")
            print(f"    Preferring OR-ensemble (higher recall: {best_or_metrics['recall']:.3f} vs {winner_metrics.get('recall', winner_metrics.get('r', 0)):.3f})")
            winner_name = 'or_ensemble'
            winner_metrics = best_or_metrics
            winner_thresholds = best_or_thresholds
            winner_is_or = True

    # If OR-ensemble won but weighted-sum is within 0.01 F1, prefer weighted-sum
    if winner_is_or and ws_m['f1'] >= winner_f1 - 0.01:
        print(f"  → OR-ensemble ({winner_f1:.3f}) and weighted-sum ({ws_m['f1']:.3f}) within 0.01")
        print(f"    Preferring weighted-sum (better test generalization)")
        winner_name = 'weighted_sum'
        winner_metrics = {'precision': ws_m['p'], 'recall': ws_m['r'], 'f1': ws_m['f1']}
        winner_thresholds = {'combined': ws_t}
        winner_is_or = False

    if winner_is_or:
        print(f"  → Using OR-ensemble (best F1)")
        use_cascade = False
        # Convert raw thresholds to percentiles for test-set generalization
        # Asymmetric relaxation: recon (primary, lower pctl) gets more slack,
        # energy (secondary, high pctl) stays tighter to limit false positives
        or_percentiles = {}
        for n in best_or_thresholds:
            raw_t = best_or_thresholds[n]
            sc = components[n]
            pctl = (sc < raw_t).mean() * 100.0
            # Adaptive relaxation proportional to 1/d':
            # High d' → less relaxation (already well-separated, keep tight)
            # Low d' → more relaxation (needs help capturing anomalies)
            d_prime = max(comp_dprime.get(n, 0.5), 0.3)
            relax = min(3.0 / d_prime, 5.0)  # Cap at 5pt to avoid over-relaxation
            pctl_relaxed = max(pctl - relax, 50.0)
            or_percentiles[n] = pctl_relaxed
            print(f"    {n} threshold = {raw_t:.4f} (p{pctl:.1f} → relaxed p{pctl_relaxed:.1f}, d'={d_prime:.2f}, relax={relax:.1f})")
        use_or_ensemble = True
    else:
        print(f"  → Using {winner_name} (best F1)")
        use_or_ensemble = False
        use_cascade = winner_thresholds.get('cascade', False)
        if winner_name == 'recon_only':
            combined_scores = recon_norm

    best_metrics = winner_metrics
    best_thresholds = winner_thresholds

    print(f"\n✓ Best Val F1: {best_metrics['f1']:.3f}, Precision: {best_metrics['precision']:.3f}, Recall: {best_metrics['recall']:.3f}")

    # Pack norm stats for test-set reuse
    single_threshold = best_thresholds.get('combined', 0.5)

    # Convert the val threshold to a percentile of the val combined scores.
    # On the test set we apply this percentile to the test score distribution
    # for distribution-invariant thresholding.
    threshold_percentile = float((combined_scores < single_threshold).mean() * 100.0)

    # For cascade, store percentiles of recon and energy thresholds
    cascade_recon_pctl = 0
    cascade_energy_pctl = 0
    if use_cascade:
        cr_t = best_thresholds.get('cascade_recon_t', 0)
        ce_t = best_thresholds.get('cascade_energy_t', 0)
        cascade_recon_pctl = float((recon_norm < cr_t).mean() * 100.0)
        cascade_energy_pctl = float((energy_scores < ce_t).mean() * 100.0)
        print(f"    Cascade recon threshold: {cr_t:.4f} (p{cascade_recon_pctl:.1f})")
        print(f"    Cascade energy threshold: {ce_t:.4f} (p{cascade_energy_pctl:.1f})")

    norm_stats = {
        'recon_p5': recon_p5, 'recon_p95': recon_p95,
        'cluster_p5': cluster_p5, 'cluster_p95': cluster_p95,
        'energy_p5': energy_p5, 'energy_p95': energy_p95,
        'use_or_ensemble': use_or_ensemble,
        'use_cascade': use_cascade if not use_or_ensemble else False,
        'cascade_recon_pctl': cascade_recon_pctl,
        'cascade_energy_pctl': cascade_energy_pctl,
        'comp_thresholds': best_thresholds,
        'or_comp_names': list(best_or_thresholds.keys()) if use_or_ensemble else [],
        'or_percentiles': or_percentiles if use_or_ensemble else {},
        'threshold_percentile': threshold_percentile,
    }

    return single_threshold, best_metrics, combined_scores, norm_stats


def save_training_plots(train_losses, val_losses, output_dir):
    """Save training curves (quick standalone version called during training)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label='Train', linewidth=1.8, color=FS.PRIMARY)
    ax.plot(val_losses, label='Validation', linewidth=1.8, color=FS.ACCENT)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    FS.save(fig, f"{output_dir}/training_curves")
    print(f"✓ Saved training curves")


# =========================================================================
# THESIS FIGURE FUNCTIONS  (all use figs_style)
# =========================================================================

def plot_training_curves(train_losses, val_losses, train_contrastive,
                         train_reconstruction, fig_dir):
    """1. Training loss curves — 2×2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Total loss
    axes[0, 0].plot(train_losses, lw=1.8, color=FS.PRIMARY, label='Train')
    axes[0, 0].plot(val_losses, lw=1.8, color=FS.ACCENT, label='Validation')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss'); axes[0, 0].legend()

    # Contrastive
    axes[0, 1].plot(train_contrastive, lw=1.8, color=FS.NEUTRALS[0])
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Contrastive Loss')
    axes[0, 1].set_title('Contrastive Loss')

    # Reconstruction
    axes[1, 0].plot(train_reconstruction, lw=1.8, color=FS.NEUTRALS[1])
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Reconstruction Loss')
    axes[1, 0].set_title('Reconstruction Loss')

    # Improvement %
    if train_losses[0] > 0:
        improv = [((train_losses[0] - l) / train_losses[0]) * 100 for l in train_losses]
        axes[1, 1].plot(improv, lw=1.8, color=FS.GOOD)
        axes[1, 1].axhline(y=50, color=FS.MUTED, ls='--', lw=1, label='50 %')
        axes[1, 1].legend()
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Loss Reduction (%)')
    axes[1, 1].set_title('Training Improvement')

    fig.tight_layout()
    FS.save(fig, f"{fig_dir}/1_training_curves")
    print("  ✓ Saved training curves")


def plot_confusion_matrix(tp, fp, fn, tn, precision, recall, f1, accuracy,
                          fig_dir):
    """2. Confusion matrix — sequential blue, counts + row-normalised %, metric strip."""
    import matplotlib.gridspec as gridspec

    cm = np.array([[tn, fp], [fn, tp]])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / (row_sums + 1e-8) * 100

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    fig = plt.figure(figsize=(6, 5.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 0.7], hspace=0.25)
    ax = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1])

    # Heatmap
    im = ax.imshow(cm, cmap=FS.SEQ_BLUE_CMAP, aspect='auto')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    # Annotate each cell: count + row %
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() * 0.6 else FS.INK
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Count')

    # Metric strip
    ax_strip.axis('off')
    strip_text = (f"Accuracy {accuracy:.3f}  │  "
                  f"Precision {precision:.3f}  │  "
                  f"Recall {recall:.3f}  │  "
                  f"Specificity {specificity:.3f}  │  "
                  f"F1 {f1:.3f}")
    ax_strip.text(0.5, 0.5, strip_text, ha='center', va='center',
                  fontsize=9.5, family='monospace',
                  transform=ax_strip.transAxes,
                  bbox=dict(boxstyle='round,pad=0.4', facecolor=FS.GRID, edgecolor='none'))

    FS.save(fig, f"{fig_dir}/2_confusion_matrix")
    print("  ✓ Saved confusion matrix")


def plot_performance_metrics(precision, recall, f1, accuracy, fig_dir):
    """3. Performance bar chart — F1 accented, target line at 0.70."""
    labels = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [precision, recall, f1, accuracy]
    colors = [FS.NEUTRALS[0], FS.NEUTRALS[1], FS.ACCENT, FS.NEUTRALS[2]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor=FS.INK, linewidth=0.7,
                  width=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=0.70, color=FS.MUTED, ls='--', lw=1.2, label='Target (0.70)')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.12)
    ax.set_title('Model Performance')
    ax.legend()
    FS.save(fig, f"{fig_dir}/3_performance_metrics")
    print("  ✓ Saved performance metrics")


def plot_score_distribution(anomaly_scores, ground_truth, fig_dir):
    """4. Anomaly-score distribution (histogram + box)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    norm_s = anomaly_scores[ground_truth == 0]
    anom_s = anomaly_scores[ground_truth == 1]

    axes[0].hist(norm_s, bins=50, alpha=0.7, label='Normal', color=FS.PRIMARY,
                 edgecolor='white', linewidth=0.4)
    axes[0].hist(anom_s, bins=50, alpha=0.7, label='Anomaly', color=FS.ACCENT,
                 edgecolor='white', linewidth=0.4)
    axes[0].set_xlabel('Anomaly Score'); axes[0].set_ylabel('Frequency')
    axes[0].set_title('Score Distribution'); axes[0].legend()

    bp = axes[1].boxplot([norm_s, anom_s], labels=['Normal', 'Anomaly'],
                         patch_artist=True,
                         medianprops=dict(color=FS.BAD, lw=2))
    bp['boxes'][0].set_facecolor(FS.PRIMARY_L)
    bp['boxes'][1].set_facecolor(FS.ACCENT_L)
    axes[1].set_ylabel('Anomaly Score')
    axes[1].set_title('Score Distribution by Class')

    fig.tight_layout()
    FS.save(fig, f"{fig_dir}/4_anomaly_score_distribution")
    print("  ✓ Saved anomaly score distribution")


def plot_precision_recall_curve(ground_truth, anomaly_scores, precision_val,
                                recall_val, fig_dir):
    """5. Precision-Recall curve — AUC in legend, operating point in ACCENT."""
    from sklearn.metrics import precision_recall_curve, auc as sk_auc

    prec_arr, rec_arr, _ = precision_recall_curve(ground_truth, anomaly_scores)
    pr_auc = sk_auc(rec_arr, prec_arr)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(rec_arr, prec_arr, lw=2, color=FS.PRIMARY,
            label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax.fill_between(rec_arr, prec_arr, alpha=0.15, color=FS.PRIMARY_L)
    ax.scatter([recall_val], [precision_val], s=140, c=FS.ACCENT, marker='*',
               edgecolors=FS.INK, linewidths=0.8, zorder=5,
               label=f'Operating Point (P={precision_val:.3f}, R={recall_val:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision–Recall Curve')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc='best')
    FS.save(fig, f"{fig_dir}/5_precision_recall_curve")
    print("  ✓ Saved precision-recall curve")
    return pr_auc


def plot_detection_timeline(anomaly_scores, predictions, ground_truth, fig_dir):
    """6. Detection timeline — score line, shaded true intervals, detection markers."""
    n_show = min(500, len(anomaly_scores))
    x = np.arange(n_show)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Shade true anomaly intervals
    gt_sub = ground_truth[:n_show]
    in_seg = False
    for i in range(n_show):
        if gt_sub[i] == 1 and not in_seg:
            seg_start = i; in_seg = True
        if (gt_sub[i] == 0 or i == n_show - 1) and in_seg:
            ax.axvspan(seg_start, i, alpha=0.18, color=FS.ACCENT_L, zorder=0)
            in_seg = False

    ax.plot(x, anomaly_scores[:n_show], lw=0.9, color=FS.PRIMARY, alpha=0.8,
            label='Anomaly Score')

    # Detection markers
    det_idx = np.where(predictions[:n_show] == 1)[0]
    if len(det_idx) > 0:
        ax.scatter(det_idx, anomaly_scores[det_idx], s=30, c=FS.ACCENT,
                   marker='v', edgecolors=FS.INK, linewidths=0.4, zorder=4,
                   label=f'Detected ({len(det_idx)})', alpha=0.85)

    # True anomaly markers (if separate from shading)
    true_idx = np.where(gt_sub == 1)[0]
    if len(true_idx) > 0:
        ax.scatter(true_idx, anomaly_scores[true_idx], s=18, c=FS.BAD,
                   marker='o', edgecolors='none', zorder=3, alpha=0.5,
                   label=f'True Anomaly ({len(true_idx)})')

    ax.set_xlabel('Sample Index'); ax.set_ylabel('Anomaly Score')
    ax.set_title('Detection Timeline (First 500 Samples)')
    ax.legend(fontsize=8, loc='upper right')
    FS.save(fig, f"{fig_dir}/6_detection_timeline")
    print("  ✓ Saved detection timeline")


def plot_results_dashboard(train_losses, val_losses, anomaly_scores, predictions,
                           ground_truth, tp, fp, fn, tn, precision, recall,
                           f1, accuracy, fig_dir):
    """7. Multi-panel results dashboard."""
    import seaborn as sns

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Confusion matrix (mini)
    ax1 = fig.add_subplot(gs[0, 0])
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap=FS.SEQ_BLUE_CMAP, cbar=False,
                xticklabels=['N', 'A'], yticklabels=['N', 'A'], ax=ax1,
                linewidths=0.5, linecolor='white')
    ax1.set_title('Confusion Matrix')

    # Metrics bars
    ax2 = fig.add_subplot(gs[0, 1:])
    met_names = ['Precision', 'Recall', 'F1', 'Accuracy']
    met_vals = [precision, recall, f1, accuracy]
    met_colors = [FS.NEUTRALS[0], FS.NEUTRALS[1], FS.ACCENT, FS.NEUTRALS[2]]
    bars = ax2.barh(met_names, met_vals, color=met_colors, edgecolor=FS.INK, lw=0.5)
    for bar, val in zip(bars, met_vals):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontweight='bold', fontsize=9)
    ax2.set_xlim(0, 1.15); ax2.set_title('Performance Metrics')

    # Training curves
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(train_losses, lw=1.5, color=FS.PRIMARY, label='Train')
    ax3.plot(val_losses, lw=1.5, color=FS.ACCENT, label='Validation')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Loss')
    ax3.set_title('Training Progress'); ax3.legend()

    # Score distribution
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.hist(anomaly_scores[ground_truth == 0], bins=40, alpha=0.7,
             label='Normal', color=FS.PRIMARY, edgecolor='white', lw=0.3)
    ax4.hist(anomaly_scores[ground_truth == 1], bins=40, alpha=0.7,
             label='Anomaly', color=FS.ACCENT, edgecolor='white', lw=0.3)
    ax4.set_xlabel('Anomaly Score'); ax4.set_ylabel('Frequency')
    ax4.set_title('Score Distribution'); ax4.legend()

    # Stats text
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    stats = (f"  Samples: {len(ground_truth)}\n"
             f"  Anomalies: {int(ground_truth.sum())}\n"
             f"  Detected: {int(predictions.sum())}\n\n"
             f"  TP: {tp}  FP: {fp}\n"
             f"  FN: {fn}  TN: {tn}\n\n"
             f"  F1: {f1:.3f}")
    ax5.text(0.1, 0.5, stats, fontsize=9.5, family='monospace',
             va='center', transform=ax5.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=FS.GRID, edgecolor='none'))

    fig.suptitle('Results Dashboard', fontsize=13, fontweight='bold', y=0.98)
    FS.save(fig, f"{fig_dir}/7_results_dashboard")
    print("  ✓ Saved results dashboard")


def plot_tsne_embeddings(embeddings_np, ground_truth, fig_dir):
    """8. t-SNE of learned embeddings."""
    from sklearn.manifold import TSNE

    max_pts = 3000
    if len(embeddings_np) > max_pts:
        idx = np.random.choice(len(embeddings_np), max_pts, replace=False)
        emb, gt = embeddings_np[idx], ground_truth[idx]
    else:
        emb, gt = embeddings_np, ground_truth

    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
    red = tsne.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(7, 6))
    norm_m = gt == 0; anom_m = gt == 1
    ax.scatter(red[norm_m, 0], red[norm_m, 1], c=FS.PRIMARY, alpha=0.25, s=8,
               label=f'Normal ({norm_m.sum()})', edgecolors='none')
    ax.scatter(red[anom_m, 0], red[anom_m, 1], c=FS.ACCENT, alpha=0.8, s=30,
               edgecolors=FS.INK, linewidths=0.4,
               label=f'Anomaly ({anom_m.sum()})')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE of Learned Embeddings')
    ax.legend(loc='best')
    FS.save(fig, f"{fig_dir}/8_tsne_embeddings")
    print("  ✓ Saved t-SNE embeddings")


def plot_roc_curve(ground_truth, anomaly_scores, recall_val, specificity_val,
                   fig_dir):
    """9. ROC curve — AUC in legend, operating point in ACCENT."""
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(ground_truth, anomaly_scores)
    roc_auc = roc_auc_score(ground_truth, anomaly_scores)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, lw=2, color=FS.PRIMARY,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.12, color=FS.PRIMARY_L)
    ax.plot([0, 1], [0, 1], ls='--', lw=1, color=FS.MUTED, label='Random')

    # Operating point
    op_fpr = 1 - specificity_val
    op_tpr = recall_val
    ax.scatter([op_fpr], [op_tpr], s=140, c=FS.ACCENT, marker='*',
               edgecolors=FS.INK, linewidths=0.8, zorder=5,
               label=f'Operating Point')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right')
    FS.save(fig, f"{fig_dir}/9_roc_curve")
    print("  ✓ Saved ROC curve")
    return roc_auc, fpr, tpr


def plot_component_score_comparison(recon_scores, energy_scores, cluster_scores,
                                     ground_truth, fig_dir):
    """10. Per-component score distributions with corrected d′."""
    components = [('Reconstruction', recon_scores),
                  ('Cluster', cluster_scores)]
    if energy_scores is not None:
        components.append(('Energy', energy_scores))

    n_cols = len(components)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 4.2))
    if n_cols == 1:
        axes = [axes]

    normal_mask = ground_truth == 0
    anomaly_mask = ground_truth == 1
    dprime_values = {}

    for ax, (name, scores) in zip(axes, components):
        ax.hist(scores[normal_mask], bins=50, alpha=0.65, label='Normal',
                color=FS.PRIMARY, edgecolor='white', linewidth=0.3, density=True)
        ax.hist(scores[anomaly_mask], bins=50, alpha=0.65, label='Anomaly',
                color=FS.ACCENT, edgecolor='white', linewidth=0.3, density=True)
        ax.set_xlabel('Score'); ax.set_ylabel('Density')
        ax.set_title(f'{name}')
        ax.legend(fontsize=8)

        if normal_mask.sum() > 0 and anomaly_mask.sum() > 0:
            dp = FS.d_prime(scores[anomaly_mask], scores[normal_mask])
            dprime_values[name.lower()] = dp
            ax.text(0.95, 0.95, FS.d_prime_label(dp),
                    transform=ax.transAxes, fontsize=9.5, fontweight='bold',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=FS.GRID,
                              edgecolor='none', alpha=0.9))

    fig.tight_layout()
    FS.save(fig, f"{fig_dir}/10_component_scores")
    print("  ✓ Saved component score comparison")
    return dprime_values


def plot_ablation_study(recon_scores, energy_scores, cluster_scores,
                        ground_truth, config, fig_dir):
    """11. Ablation study — honest ordering, ACCENT on 'Full Hybrid (Ours)'."""

    def compute_best_f1(scores, gt):
        best_f1 = 0
        lo, hi = np.percentile(scores, 50), np.percentile(scores, 99.5)
        for t in np.linspace(lo, hi, 300):
            pred = scores > t
            tp = np.sum(pred & (gt == 1))
            fp = np.sum(pred & (gt == 0))
            fn = np.sum(~pred & (gt == 1))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f > best_f1:
                best_f1 = f
        return best_f1

    def norm01(s):
        p5, p95 = np.percentile(s, [5, 95])
        return np.clip((s - p5) / (p95 - p5 + 1e-8), 0, 1)

    results = {}
    results['Recon Only'] = compute_best_f1(recon_scores, ground_truth)
    results['Cluster Only'] = compute_best_f1(cluster_scores, ground_truth)

    rc = 0.7 * norm01(recon_scores) + 0.3 * norm01(cluster_scores)
    results['Recon+Cluster'] = compute_best_f1(rc, ground_truth)

    if energy_scores is not None:
        results['Energy Only'] = compute_best_f1(energy_scores, ground_truth)
        re = 0.7 * norm01(recon_scores) + 0.3 * norm01(energy_scores)
        results['Recon+Energy'] = compute_best_f1(re, ground_truth)
        hybrid = (config.RECON_WEIGHT * norm01(recon_scores) +
                  config.CLUSTER_WEIGHT * norm01(cluster_scores) +
                  config.ENERGY_WEIGHT * norm01(energy_scores))
        results['Full Hybrid\n(Ours)'] = compute_best_f1(hybrid, ground_truth)

    names = list(results.keys())
    values = list(results.values())
    # Color: ACCENT for "Ours", NEUTRALS for everything else
    colors = [FS.ACCENT if 'Ours' in n else FS.NEUTRALS[i % len(FS.NEUTRALS)]
              for i, n in enumerate(names)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values, color=colors, edgecolor=FS.INK, linewidth=0.7,
                  width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=0.70, color=FS.MUTED, ls='--', lw=1.2, label='Target (0.70)')
    ax.set_ylabel('Best F1 Score'); ax.set_ylim(0, max(values) * 1.18 + 0.05)
    ax.set_title('Ablation Study')
    ax.legend()
    fig.tight_layout()
    FS.save(fig, f"{fig_dir}/11_ablation_study")
    print("  ✓ Saved ablation study")
    return results


def plot_per_anomaly_type_detection(anomaly_type_seq, predictions, ground_truth,
                                     fig_dir):
    """12. Per-anomaly-type detection rate — horizontal bars."""
    if anomaly_type_seq is None or len(anomaly_type_seq) == 0:
        print("  ⚠️ No anomaly type info, skipping per-type plot")
        return {}

    type_stats = {}
    for i in range(len(predictions)):
        if ground_truth[i] == 1 and i in anomaly_type_seq:
            atype = anomaly_type_seq[i]
            if atype not in type_stats:
                type_stats[atype] = {'total': 0, 'detected': 0}
            type_stats[atype]['total'] += 1
            if predictions[i] == 1:
                type_stats[atype]['detected'] += 1

    if not type_stats:
        print("  ⚠️ No matched anomaly types, skipping per-type plot")
        return {}

    types = sorted(type_stats.keys())
    rates = [type_stats[t]['detected'] / type_stats[t]['total']
             if type_stats[t]['total'] > 0 else 0 for t in types]
    counts = [type_stats[t]['total'] for t in types]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(types) * 0.7)))
    colors = [FS.NEUTRALS[i % len(FS.NEUTRALS)] for i in range(len(types))]
    y_pos = np.arange(len(types))
    bars = ax.barh(y_pos, rates, color=colors, edgecolor=FS.INK, linewidth=0.5,
                   height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types)

    for bar, rate, cnt in zip(bars, rates, counts):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{rate:.1%} (n={cnt})', va='center', fontsize=9)

    ax.axvline(x=0.5, color=FS.MUTED, ls='--', lw=1, label='50 % baseline')
    ax.set_xlabel('Detection Rate'); ax.set_xlim(0, 1.25)
    ax.set_title('Detection Rate by Anomaly Type')
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    FS.save(fig, f"{fig_dir}/12_per_type_detection")
    print("  ✓ Saved per-anomaly-type detection rates")

    return {t: {'rate': r, 'total': c, 'detected': type_stats[t]['detected']}
            for t, r, c in zip(types, rates, counts)}


def plot_cluster_visualization(embeddings_np, cluster_labels, ground_truth,
                               fig_dir):
    """13. Cluster visualization with anomaly overlay (PCA)."""
    from sklearn.decomposition import PCA

    max_pts = 3000
    if len(embeddings_np) > max_pts:
        idx = np.random.choice(len(embeddings_np), max_pts, replace=False)
        emb, cl, gt = embeddings_np[idx], cluster_labels[idx], ground_truth[idx]
    else:
        emb, cl, gt = embeddings_np, cluster_labels, ground_truth

    pca = PCA(n_components=2, random_state=SEED)
    red = pca.fit_transform(emb)
    ev = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: by cluster
    unique_cl = np.unique(cl)
    for i, c in enumerate(unique_cl):
        m = cl == c
        axes[0].scatter(red[m, 0], red[m, 1],
                        c=FS.NEUTRALS[i % len(FS.NEUTRALS)],
                        alpha=0.4, s=10, label=f'Cluster {c} ({m.sum()})')
    axes[0].set_xlabel(f'PC1 ({ev[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({ev[1]:.1%})')
    axes[0].set_title('By Cluster')
    axes[0].legend(fontsize=7, ncol=2)

    # Right: by ground truth
    norm_m = gt == 0; anom_m = gt == 1
    axes[1].scatter(red[norm_m, 0], red[norm_m, 1], c=FS.PRIMARY, alpha=0.25,
                    s=8, label=f'Normal ({norm_m.sum()})', edgecolors='none')
    axes[1].scatter(red[anom_m, 0], red[anom_m, 1], c=FS.ACCENT, alpha=0.8,
                    s=28, edgecolors=FS.INK, linewidths=0.4,
                    label=f'Anomaly ({anom_m.sum()})')
    axes[1].set_xlabel(f'PC1 ({ev[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({ev[1]:.1%})')
    axes[1].set_title('By Ground Truth')
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    FS.save(fig, f"{fig_dir}/13_cluster_visualization")
    print("  ✓ Saved cluster visualization")


def plot_threshold_sensitivity(anomaly_scores, ground_truth, fig_dir):
    """14. Threshold sensitivity — P/R/F1 vs threshold, crossover marked."""
    thresholds = np.linspace(np.percentile(anomaly_scores, 10),
                             np.percentile(anomaly_scores, 99.5), 200)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        pred = anomaly_scores > t
        tp = np.sum(pred & (ground_truth == 1))
        fp = np.sum(pred & (ground_truth == 0))
        fn = np.sum(~pred & (ground_truth == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precs.append(p); recs.append(r); f1s.append(f)

    best_idx = int(np.argmax(f1s))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precs, lw=1.5, color=FS.NEUTRALS[0], label='Precision')
    ax.plot(thresholds, recs, lw=1.5, color=FS.NEUTRALS[1], label='Recall')
    ax.plot(thresholds, f1s, lw=2.2, color=FS.ACCENT, label='F1 Score')

    ax.axvline(x=thresholds[best_idx], color=FS.MUTED, ls='--', lw=1,
               label=f'Optimal θ={thresholds[best_idx]:.3f} (F1={f1s[best_idx]:.3f})')
    ax.scatter([thresholds[best_idx]], [f1s[best_idx]], s=100, c=FS.ACCENT,
               marker='*', zorder=5, edgecolors=FS.INK, linewidths=0.5)

    ax.set_xlabel('Threshold'); ax.set_ylabel('Metric Value')
    ax.set_title('Threshold Sensitivity')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FS.save(fig, f"{fig_dir}/14_threshold_sensitivity")
    print("  ✓ Saved threshold sensitivity")


def plot_reconstruction_error_heatmap(model, test_tensor, ground_truth,
                                       feature_names, config, fig_dir):
    """15. Per-feature reconstruction error heatmap."""
    model.eval()
    device = config.DEVICE

    with torch.no_grad():
        n_sub = min(500, len(test_tensor))
        x = test_tensor[:n_sub].to(device)
        gt = ground_truth[:n_sub]
        encoded = model.encoder(x)
        reconstructed = model.reconstructor.reconstruction_head(encoded)
        errors = (x - reconstructed).pow(2).cpu().numpy()

    errors_per_feature = errors.mean(axis=1)
    normal_errors = errors_per_feature[gt == 0].mean(axis=0)
    anomaly_errors = errors_per_feature[gt == 1].mean(axis=0) if gt.sum() > 0 else np.zeros(len(feature_names))
    error_ratio = anomaly_errors / (normal_errors + 1e-8)

    sort_idx = np.argsort(error_ratio)[::-1]
    sorted_features = [feature_names[i] for i in sort_idx]
    sorted_normal = normal_errors[sort_idx]
    sorted_anomaly = anomaly_errors[sort_idx]
    sorted_ratio = error_ratio[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    n_feat = len(sorted_features)
    y_pos = np.arange(n_feat)

    # Left: normal vs anomaly
    axes[0].barh(y_pos - 0.18, sorted_normal, height=0.33, color=FS.PRIMARY,
                 edgecolor='white', lw=0.3, label='Normal', alpha=0.85)
    axes[0].barh(y_pos + 0.18, sorted_anomaly, height=0.33, color=FS.ACCENT,
                 edgecolor='white', lw=0.3, label='Anomaly', alpha=0.85)
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels(sorted_features, fontsize=8)
    axes[0].set_xlabel('Mean Reconstruction Error')
    axes[0].set_title('Error by Feature')
    axes[0].legend(fontsize=8); axes[0].invert_yaxis()

    # Right: error ratio
    bar_colors = [FS.BAD if r > 1.5 else FS.ACCENT if r > 1.0 else FS.GOOD
                  for r in sorted_ratio]
    axes[1].barh(y_pos, sorted_ratio, color=bar_colors, edgecolor='white', lw=0.3,
                 alpha=0.85)
    axes[1].axvline(x=1.0, color=FS.INK, ls='--', lw=1, alpha=0.5)
    axes[1].set_yticks(y_pos); axes[1].set_yticklabels(sorted_features, fontsize=8)
    axes[1].set_xlabel('Error Ratio (Anomaly / Normal)')
    axes[1].set_title('Feature Discriminability')
    axes[1].invert_yaxis()

    fig.tight_layout()
    FS.save(fig, f"{fig_dir}/15_reconstruction_error_heatmap")
    print("  ✓ Saved reconstruction error heatmap")


def plot_attention_heatmap(model, test_tensor, ground_truth, config, fig_dir):
    """16. Transformer self-attention heatmap (normal vs anomaly)."""
    model.eval()
    device = config.DEVICE

    normal_idx = np.where(ground_truth == 0)[0]
    anomaly_idx = np.where(ground_truth == 1)[0]
    if len(normal_idx) == 0 or len(anomaly_idx) == 0:
        print("  ⚠️ Need both classes for attention heatmap")
        return None, None

    samples = {
        'Normal': test_tensor[normal_idx[0]:normal_idx[0]+1].to(device),
        'Anomaly': test_tensor[anomaly_idx[0]:anomaly_idx[0]+1].to(device),
    }
    attn_data = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    with torch.no_grad():
        for ax, (label, x) in zip(axes, samples.items()):
            projected = model.encoder.input_projection(x)
            projected = model.encoder.pos_encoder(projected)
            layer = model.encoder.transformer_encoder.layers[0]
            src = layer.norm1(projected)
            _, attn_weights = layer.self_attn(
                src, src, src, need_weights=True, average_attn_weights=True)
            attn = attn_weights[0].cpu().numpy()
            attn_data[label.lower()] = attn

            im = ax.imshow(attn, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.8, label='Weight')
            ax.set_title(f'Attention ({label})')
            ax.set_xlabel('Key Position'); ax.set_ylabel('Query Position')

            # Sparse ticks
            step = max(1, attn.shape[0] // 6)
            ticks = list(range(0, attn.shape[0], step))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels(ticks, fontsize=7)
            ax.set_yticklabels(ticks, fontsize=7)

            # Disable grid for heatmaps
            ax.grid(False)

    fig.tight_layout()
    FS.save(fig, f"{fig_dir}/16_attention_heatmap")
    print("  ✓ Saved attention heatmap")
    return attn_data.get('normal'), attn_data.get('anomaly')


def plot_energy_score_landscape(embeddings_np, energy_scores, ground_truth,
                                fig_dir):
    """17. Energy score landscape — PCA with energy gradient + GT overlay."""
    if energy_scores is None:
        print("  ⚠️ No energy scores, skipping energy landscape")
        return

    from sklearn.decomposition import PCA

    max_pts = 3000
    if len(embeddings_np) > max_pts:
        idx = np.random.choice(len(embeddings_np), max_pts, replace=False)
        emb, es, gt = embeddings_np[idx], energy_scores[idx], ground_truth[idx]
    else:
        emb, es, gt = embeddings_np, energy_scores, ground_truth

    pca = PCA(n_components=2, random_state=SEED)
    red = pca.fit_transform(emb)
    ev = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: energy gradient
    sc = axes[0].scatter(red[:, 0], red[:, 1], c=es, cmap='magma',
                         s=10, alpha=0.55, edgecolors='none')
    plt.colorbar(sc, ax=axes[0], label='Energy Score', shrink=0.8)
    axes[0].set_xlabel(f'PC1 ({ev[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({ev[1]:.1%})')
    axes[0].set_title('Energy Score Landscape')
    axes[0].grid(False)

    # Right: ground truth
    norm_m = gt == 0; anom_m = gt == 1
    axes[1].scatter(red[norm_m, 0], red[norm_m, 1], c=FS.PRIMARY, alpha=0.25,
                    s=8, label='Normal', edgecolors='none')
    axes[1].scatter(red[anom_m, 0], red[anom_m, 1], c=FS.ACCENT, alpha=0.8,
                    s=28, edgecolors=FS.INK, linewidths=0.4, label='Anomaly')
    axes[1].set_xlabel(f'PC1 ({ev[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({ev[1]:.1%})')
    axes[1].set_title('Ground Truth Overlay')
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    FS.save(fig, f"{fig_dir}/17_energy_landscape")
    print("  ✓ Saved energy score landscape")


# =========================================================================
# ORCHESTRATOR
# =========================================================================

def generate_thesis_visualizations(train_losses, val_losses, train_contrastive,
                                  train_reconstruction, anomaly_scores, predictions,
                                  ground_truth, tp, fp, fn, tn, precision, recall,
                                  f1, accuracy, output_dir,
                                  embeddings_np=None, cluster_labels=None,
                                  recon_scores=None, energy_scores=None,
                                  cluster_scores=None, anomaly_type_seq=None,
                                  model=None, test_tensor=None, feature_names=None,
                                  config=None):
    """Generate all publication-quality thesis figures."""
    FS.set_style()

    fig_dir = f"{output_dir}/thesis_figures"
    os.makedirs(fig_dir, exist_ok=True)
    n_plots = 0

    # Collect data for paper_metrics
    collected = {}

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 1. Training curves
    try:
        plot_training_curves(train_losses, val_losses, train_contrastive,
                             train_reconstruction, fig_dir)
        n_plots += 1
    except Exception as e:
        print(f"  ⚠️ Training curves failed: {e}")

    # 2. Confusion matrix
    try:
        plot_confusion_matrix(tp, fp, fn, tn, precision, recall, f1, accuracy,
                              fig_dir)
        n_plots += 1
    except Exception as e:
        print(f"  ⚠️ Confusion matrix failed: {e}")

    # 3. Performance metrics
    try:
        plot_performance_metrics(precision, recall, f1, accuracy, fig_dir)
        n_plots += 1
    except Exception as e:
        print(f"  ⚠️ Performance metrics failed: {e}")

    # 4. Score distribution
    try:
        plot_score_distribution(anomaly_scores, ground_truth, fig_dir)
        n_plots += 1
    except Exception as e:
        print(f"  ⚠️ Score distribution failed: {e}")

    # 5. Precision-Recall curve
    pr_auc = None
    try:
        pr_auc = plot_precision_recall_curve(ground_truth, anomaly_scores,
                                              precision, recall, fig_dir)
        collected['pr_auc'] = pr_auc
        n_plots += 1
    except Exception as e:
        print(f"  ⚠️ PR curve failed: {e}")

    # 6. Detection timeline
    try:
        plot_detection_timeline(anomaly_scores, predictions, ground_truth, fig_dir)
        n_plots += 1
    except Exception as e:
        print(f"  ⚠️ Detection timeline failed: {e}")

    # 7. Dashboard
    try:
        plot_results_dashboard(train_losses, val_losses, anomaly_scores,
                               predictions, ground_truth, tp, fp, fn, tn,
                               precision, recall, f1, accuracy, fig_dir)
        n_plots += 1
    except Exception as e:
        print(f"  ⚠️ Dashboard failed: {e}")

    # 8. t-SNE
    if embeddings_np is not None:
        try:
            plot_tsne_embeddings(embeddings_np, ground_truth, fig_dir)
            n_plots += 1
        except Exception as e:
            print(f"  ⚠️ t-SNE failed: {e}")

    # 9. ROC curve
    roc_auc = None
    roc_fpr = roc_tpr = None
    try:
        roc_auc, roc_fpr, roc_tpr = plot_roc_curve(
            ground_truth, anomaly_scores, recall, specificity, fig_dir)
        collected['roc_auc'] = roc_auc
        n_plots += 1
    except Exception as e:
        print(f"  ⚠️ ROC curve failed: {e}")

    # 10. Component scores (corrected d′)
    dprime_values = {}
    if recon_scores is not None and cluster_scores is not None:
        try:
            dprime_values = plot_component_score_comparison(
                recon_scores, energy_scores, cluster_scores, ground_truth, fig_dir)
            collected['dprime'] = dprime_values
            n_plots += 1
        except Exception as e:
            print(f"  ⚠️ Component scores failed: {e}")

    # 11. Ablation study
    ablation_results = {}
    if recon_scores is not None and cluster_scores is not None and config is not None:
        try:
            ablation_results = plot_ablation_study(
                recon_scores, energy_scores, cluster_scores, ground_truth,
                config, fig_dir)
            collected['ablation'] = ablation_results
            n_plots += 1
        except Exception as e:
            print(f"  ⚠️ Ablation study failed: {e}")

    # 12. Per-type detection
    per_type_stats = {}
    if anomaly_type_seq is not None:
        try:
            per_type_stats = plot_per_anomaly_type_detection(
                anomaly_type_seq, predictions, ground_truth, fig_dir)
            collected['per_type'] = per_type_stats
            n_plots += 1
        except Exception as e:
            print(f"  ⚠️ Per-type detection failed: {e}")

    # 13. Cluster visualization
    if embeddings_np is not None and cluster_labels is not None:
        try:
            plot_cluster_visualization(embeddings_np, cluster_labels,
                                       ground_truth, fig_dir)
            n_plots += 1
        except Exception as e:
            print(f"  ⚠️ Cluster visualization failed: {e}")

    # 14. Threshold sensitivity
    try:
        plot_threshold_sensitivity(anomaly_scores, ground_truth, fig_dir)
        n_plots += 1
    except Exception as e:
        print(f"  ⚠️ Threshold sensitivity failed: {e}")

    # 15. Reconstruction error heatmap
    if model is not None and test_tensor is not None and feature_names is not None and config is not None:
        try:
            plot_reconstruction_error_heatmap(model, test_tensor, ground_truth,
                                              feature_names, config, fig_dir)
            n_plots += 1
        except Exception as e:
            print(f"  ⚠️ Reconstruction error heatmap failed: {e}")

    # 16. Attention heatmap
    attn_normal = attn_anomaly = None
    if model is not None and test_tensor is not None and config is not None:
        try:
            attn_normal, attn_anomaly = plot_attention_heatmap(
                model, test_tensor, ground_truth, config, fig_dir)
            n_plots += 1
        except Exception as e:
            print(f"  ⚠️ Attention heatmap failed: {e}")

    # 17. Energy landscape
    if embeddings_np is not None and energy_scores is not None:
        try:
            plot_energy_score_landscape(embeddings_np, energy_scores,
                                        ground_truth, fig_dir)
            n_plots += 1
        except Exception as e:
            print(f"  ⚠️ Energy landscape failed: {e}")

    print(f"\n✓ All {n_plots} figures saved to {fig_dir}/")

    # Return collected data for paper_metrics.json
    collected['attn_normal'] = attn_normal
    collected['attn_anomaly'] = attn_anomaly
    collected['roc_fpr'] = roc_fpr
    collected['roc_tpr'] = roc_tpr
    return fig_dir, collected




def main():
    """Main training pipeline"""
    FS.set_style()

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
    df_with_anomalies, ground_truth, anomaly_type_map = inject_diverse_anomalies(
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

    # Clip scaled data to [-10, 10] to prevent extreme anomaly values from
    # destabilizing training. Anomalies will be at ±10 (clearly outliers)
    # while normals stay around ±2. Without clipping, ranges like [-153, 187]
    # make reconstruction loss huge and unstable.
    clip_val = 10.0
    sequences = np.clip(sequences, -clip_val, clip_val)
    print(f"✓ Clipped scaled sequences to [{-clip_val}, {clip_val}]")
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

    # Align anomaly type map to sequence indices (last-point labeling)
    anomaly_type_seq = {}
    for i in range(n_sequences):
        last_idx = i + ImprovedConfig.WINDOW_SIZE - 1
        if last_idx < len(surviving_indices):
            original_idx = surviving_indices[last_idx]
            if original_idx in anomaly_type_map:
                anomaly_type_seq[i] = anomaly_type_map[original_idx]

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
    # CRITICAL: Train autoencoder on NORMAL data ONLY
    # This is the standard approach for anomaly detection autoencoders:
    # the model learns to reconstruct normal patterns well, and anomalies
    # produce high reconstruction error because the model has never seen them.
    train_normal_mask = ~train_gt.astype(bool)
    train_normal_data = train_data[train_normal_mask]
    print(f"  Autoencoder training: {train_normal_mask.sum()} normal sequences "
          f"(excluded {(~train_normal_mask).sum()} anomalies)")

    train_normal_tensor = torch.FloatTensor(train_normal_data)
    train_tensor = torch.FloatTensor(train_data)  # Full data for energy detector
    val_tensor = torch.FloatTensor(val_data)
    test_tensor = torch.FloatTensor(test_data)

    # Deterministic DataLoader seeding
    dl_generator = torch.Generator()
    dl_generator.manual_seed(SEED)

    # Autoencoder trains on normal-only data
    train_loader = DataLoader(TensorDataset(train_normal_tensor),
                              batch_size=ImprovedConfig.BATCH_SIZE, shuffle=True,
                              generator=dl_generator)
    # Validation uses ALL data (anomalies increase val loss = good stopping signal)
    val_loader = DataLoader(TensorDataset(val_tensor),
                            batch_size=ImprovedConfig.BATCH_SIZE, shuffle=False)

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
        mask_ratio=ImprovedConfig.MASK_RATIO,
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

    epoch = 0
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

    # Extract embeddings - fit clusters on normal data only
    model.eval()
    with torch.no_grad():
        train_normal_embeddings = model.get_embeddings(
            train_normal_tensor.to(ImprovedConfig.DEVICE)).cpu().numpy()
        train_embeddings = model.get_embeddings(
            train_tensor.to(ImprovedConfig.DEVICE)).cpu().numpy()

    # Cluster on normal data only (clusters represent normal patterns)
    clustering = DensityAwareClustering(
        n_clusters=ImprovedConfig.N_CLUSTERS,
        min_cluster_size=ImprovedConfig.MIN_CLUSTER_SIZE
    )
    clustering.fit(train_normal_embeddings)
    # Predict cluster labels for ALL training data (needed by energy detector)
    cluster_labels = clustering.predict(train_embeddings)

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
    # [4b] LATENT SPACE REGULARIZATION
    # Fine-tune embeddings with center loss + separation loss to improve
    # cluster structure (as described in the abstract)
    # ========================================================================
    print("\n[4b] Applying latent space regularization...")
    regularizer = LatentSpaceRegularizer(
        embedding_dim=ImprovedConfig.D_MODEL,
        n_clusters=ImprovedConfig.N_CLUSTERS,
        alpha=0.5
    ).to(ImprovedConfig.DEVICE)

    # Initialize regularizer centers from K-Means results
    with torch.no_grad():
        for k in range(ImprovedConfig.N_CLUSTERS):
            mask = cluster_labels == k
            if mask.sum() > 0:
                regularizer.centers.data[k] = torch.FloatTensor(
                    train_embeddings[mask].mean(axis=0)
                ).to(ImprovedConfig.DEVICE)

    reg_optimizer = optim.AdamW(
        list(model.parameters()) + list(regularizer.parameters()),
        lr=ImprovedConfig.REGULARIZATION_LR,
        weight_decay=ImprovedConfig.WEIGHT_DECAY
    )

    model.train()
    reg_loader = DataLoader(
        TensorDataset(train_normal_tensor, torch.LongTensor(
            clustering.predict(train_normal_embeddings)
        )),
        batch_size=ImprovedConfig.BATCH_SIZE, shuffle=True
    )

    for reg_epoch in range(ImprovedConfig.REGULARIZATION_EPOCHS):
        reg_losses = []
        for x_batch, cl_batch in reg_loader:
            x_batch = x_batch.to(ImprovedConfig.DEVICE)
            cl_batch = cl_batch.to(ImprovedConfig.DEVICE)

            reg_optimizer.zero_grad()
            recon_loss_val, _ = model(x_batch, use_contrastive=False, use_reconstruction=True)
            emb = model.get_embeddings(x_batch)
            reg_loss = regularizer(emb, cl_batch)
            total_loss = recon_loss_val + ImprovedConfig.REGULARIZATION_WEIGHT * reg_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ImprovedConfig.GRADIENT_CLIP)
            reg_optimizer.step()
            regularizer.update_centers(emb.detach(), cl_batch)
            reg_losses.append(total_loss.item())

        if (reg_epoch + 1) % 5 == 0 or reg_epoch == 0:
            print(f"  Reg epoch {reg_epoch+1}/{ImprovedConfig.REGULARIZATION_EPOCHS}: "
                  f"loss={np.mean(reg_losses):.4f}")

    print("✓ Latent space regularization complete")

    # Re-extract embeddings and re-cluster with regularized representations
    model.eval()
    with torch.no_grad():
        train_embeddings = model.get_embeddings(
            train_tensor.to(ImprovedConfig.DEVICE)).cpu().numpy()
        train_normal_embeddings = model.get_embeddings(
            train_normal_tensor.to(ImprovedConfig.DEVICE)).cpu().numpy()
    clustering.fit(train_normal_embeddings)
    cluster_labels = clustering.predict(train_embeddings)
    cluster_labels_tensor = torch.LongTensor(cluster_labels).to(ImprovedConfig.DEVICE)
    print(f"✓ Re-clustered with regularized embeddings: {len(np.unique(cluster_labels))} clusters")

    # ========================================================================
    # [5] RECONSTRUCTION DETECTOR
    # ========================================================================
    print("\n[5/8] Fitting reconstruction detector...")

    # Build feature weights: emphasize price-sensitive features, ZERO-weight slow indicators
    # Slow indicators (SMA, EMA, Bollinger, ADX) barely change from point anomalies
    # and add noise that dilutes the discriminative reconstruction error signal
    price_sensitive = {'close', 'open', 'high', 'low', 'returns', 'log_returns',
                       'high_low_range', 'close_open_range', 'atr', 'atr_pct'}
    slow_indicators = {'sma_20', 'sma_50', 'ema_12', 'ema_26', 'adx',
                       'bb_high', 'bb_low', 'bb_position'}
    feature_weights = []
    for fname in feature_names:
        if fname.lower() in price_sensitive:
            feature_weights.append(3.0)
        elif fname.lower() in slow_indicators:
            feature_weights.append(0.0)  # Zero weight - these dilute signal
        else:
            feature_weights.append(1.0)
    feature_weights_tensor = torch.FloatTensor(feature_weights)
    print(f"  Feature weights: {dict(zip(feature_names, feature_weights))}")

    recon_detector = ReconstructionBasedDetector(
        reconstructor=model.reconstructor,
        threshold_percentile=95,  # Lower threshold to catch more anomalies
        feature_weights=feature_weights_tensor
    )
    # Fit on NORMAL data only — threshold reflects normal reconstruction quality
    recon_detector.fit(train_normal_tensor.to(ImprovedConfig.DEVICE))
    print("✓ Reconstruction detector fitted (on normal data only)")

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
            # Update per-cluster energy statistics for cluster-conditioned scoring
            model.eval()
            energy_detector.eval()
            with torch.no_grad():
                train_emb = model.get_embeddings(train_tensor.to(ImprovedConfig.DEVICE))
                energy_detector.update_cluster_statistics(train_emb, cluster_labels_tensor)
            print("✓ Updated cluster-conditioned energy statistics")
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

        # Regime transition scores
        regime_scores = clustering.compute_regime_transition_scores(test_cluster_labels)

        energy_scores = None
        if ImprovedConfig.USE_HYBRID and energy_detector is not None:
            test_cluster_labels_t = torch.LongTensor(test_cluster_labels).to(ImprovedConfig.DEVICE)
            energy_scores_t = energy_detector(embeddings, cluster_labels=test_cluster_labels_t)
            energy_scores = energy_scores_t.detach().cpu().numpy() if torch.is_tensor(energy_scores_t) else energy_scores_t

    # ---- Determine predictions using same strategy as validation ----
    use_or = val_norm_stats.get('use_or_ensemble', False)
    use_cascade = val_norm_stats.get('use_cascade', False)
    comp_thresholds = val_norm_stats.get('comp_thresholds', {})
    is_recon_only = comp_thresholds.get('recon_only', False)

    if use_cascade and energy_scores is not None:
        # Cascade: recon first (high precision), energy rescues recon's misses
        # Use percentile-based thresholds for test-set generalization
        recon_p5 = val_norm_stats['recon_p5']
        recon_p95 = val_norm_stats['recon_p95']
        recon_norm = np.clip((recon_scores - recon_p5) / (recon_p95 - recon_p5 + 1e-8), 0, 1)

        cascade_recon_pctl = val_norm_stats.get('cascade_recon_pctl', 90)
        cascade_energy_pctl = val_norm_stats.get('cascade_energy_pctl', 90)
        test_recon_t = np.percentile(recon_norm, cascade_recon_pctl)
        test_energy_t = np.percentile(energy_scores, cascade_energy_pctl)

        recon_pred = recon_norm > test_recon_t
        energy_rescue = (~recon_pred) & (energy_scores > test_energy_t)
        predictions = recon_pred | energy_rescue

        detection_method = "Cascade (Reconstruction → Energy rescue)"
        # Combined score for visualizations
        final_scores = recon_norm  # Primary signal
    elif use_or:
        # OR-ensemble: use percentile-based thresholds (adapts to test distribution)
        predictions = np.zeros(len(test_gt), dtype=bool)
        or_comp_names = val_norm_stats.get('or_comp_names', [])
        or_percentiles = val_norm_stats.get('or_percentiles', {})
        components = {'recon': recon_scores, 'cluster': cluster_scores, 'regime': regime_scores}
        if energy_scores is not None:
            components['energy'] = energy_scores

        for name in or_comp_names:
            if name in components and name in or_percentiles:
                # Derive threshold from test distribution at the same percentile
                test_threshold = np.percentile(components[name], or_percentiles[name])
                predictions |= (components[name] > test_threshold)

        used_comps = [n for n in or_comp_names if n in or_percentiles]
        detection_method = f"OR-Ensemble ({' | '.join(used_comps)})"
        # For visualisations, build a combined score (max of per-component z-like scores)
        final_scores = np.maximum.reduce([
            (recon_scores - recon_scores.mean()) / (recon_scores.std() + 1e-8),
            (cluster_scores - cluster_scores.mean()) / (cluster_scores.std() + 1e-8),
        ] + ([
            (energy_scores - energy_scores.mean()) / (energy_scores.std() + 1e-8)
        ] if energy_scores is not None else []))
    elif is_recon_only:
        # Recon-only: use normalized reconstruction scores with recon threshold
        recon_p5 = val_norm_stats['recon_p5']
        recon_p95 = val_norm_stats['recon_p95']
        recon_norm = np.clip((recon_scores - recon_p5) / (recon_p95 - recon_p5 + 1e-8), 0, 1)
        final_scores = recon_norm
        predictions = final_scores > best_threshold
        detection_method = "Reconstruction-Only (strongest d' component)"
    else:
        # Weighted-sum with single threshold
        # Use VALIDATION normalization stats so the threshold is applied at
        # the exact same operating point that was optimized on the val set.
        # Re-computing p5/p95 from test data shifts the normalization and
        # causes the threshold to correspond to a different operating point.
        recon_p5_v = val_norm_stats['recon_p5']
        recon_p95_v = val_norm_stats['recon_p95']
        cluster_p5_v = val_norm_stats['cluster_p5']
        cluster_p95_v = val_norm_stats['cluster_p95']

        recon_norm = np.clip((recon_scores - recon_p5_v) / (recon_p95_v - recon_p5_v + 1e-8), 0, 1)
        cluster_norm = np.clip((cluster_scores - cluster_p5_v) / (cluster_p95_v - cluster_p5_v + 1e-8), 0, 1)

        if energy_scores is not None:
            energy_p5_v = val_norm_stats['energy_p5']
            energy_p95_v = val_norm_stats['energy_p95']
            energy_norm = np.clip((energy_scores - energy_p5_v) / (energy_p95_v - energy_p5_v + 1e-8), 0, 1)
            total_w = ImprovedConfig.RECON_WEIGHT + ImprovedConfig.CLUSTER_WEIGHT + ImprovedConfig.ENERGY_WEIGHT
            final_scores = ((ImprovedConfig.RECON_WEIGHT/total_w) * recon_norm +
                           (ImprovedConfig.CLUSTER_WEIGHT/total_w) * cluster_norm +
                           (ImprovedConfig.ENERGY_WEIGHT/total_w) * energy_norm)
            detection_method = "Hybrid (Reconstruction + Energy)"
        else:
            tw = ImprovedConfig.RECON_WEIGHT + ImprovedConfig.CLUSTER_WEIGHT
            final_scores = ((ImprovedConfig.RECON_WEIGHT / tw) * recon_norm +
                           (ImprovedConfig.CLUSTER_WEIGHT / tw) * cluster_norm)
            detection_method = "Hybrid (Reconstruction + Cluster)"

        # Apply the raw threshold directly (same operating point as validation)
        test_threshold = best_threshold
        predictions = final_scores > test_threshold

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

    # Build test-set anomaly type map (re-index from global to test-local indices)
    test_start_idx = n_train + n_val
    test_anomaly_type_seq = {}
    for global_idx, atype in anomaly_type_seq.items():
        local_idx = global_idx - test_start_idx
        if 0 <= local_idx < len(test_gt):
            test_anomaly_type_seq[local_idx] = atype

    fig_dir, viz_collected = generate_thesis_visualizations(
        train_losses, val_losses, train_contrastive, train_reconstruction,
        final_scores, predictions, test_gt, tp, fp, fn, tn,
        precision, recall, f1, accuracy, output_dir,
        embeddings_np=embeddings_np,
        cluster_labels=test_cluster_labels,
        recon_scores=recon_scores,
        energy_scores=energy_scores,
        cluster_scores=cluster_scores,
        anomaly_type_seq=test_anomaly_type_seq,
        model=model,
        test_tensor=test_tensor,
        feature_names=feature_names,
        config=ImprovedConfig
    )

    # ========================================================================
    # SAVE figure_data.npz (B2)
    # ========================================================================
    print("\nSaving figure_data.npz...")
    anomaly_type_array = np.array(
        [test_anomaly_type_seq.get(i, '') for i in range(len(test_gt))],
        dtype=object
    )
    npz_kwargs = dict(
        recon_scores=recon_scores,
        cluster_scores=cluster_scores,
        anomaly_scores=final_scores,
        ground_truth=test_gt,
        predictions=predictions.astype(np.int8),
        anomaly_types=anomaly_type_array,
        embeddings=embeddings_np,
        cluster_labels=test_cluster_labels,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        train_contrastive=np.array(train_contrastive),
        train_reconstruction=np.array(train_reconstruction),
    )
    if energy_scores is not None:
        npz_kwargs['energy_scores'] = energy_scores
    if viz_collected.get('roc_fpr') is not None:
        npz_kwargs['roc_fpr'] = viz_collected['roc_fpr']
        npz_kwargs['roc_tpr'] = viz_collected['roc_tpr']
    if viz_collected.get('attn_normal') is not None:
        npz_kwargs['attn_normal'] = viz_collected['attn_normal']
    if viz_collected.get('attn_anomaly') is not None:
        npz_kwargs['attn_anomaly'] = viz_collected['attn_anomaly']
    np.savez_compressed(f"{output_dir}/figure_data.npz", **npz_kwargs)
    print(f"✓ Saved figure_data.npz")

    # ========================================================================
    # SAVE paper_metrics.json (B3)
    # ========================================================================
    print("\nGenerating paper_metrics.json...")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    paper_metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'roc_auc': float(viz_collected.get('roc_auc', 0) or 0),
        'pr_auc': float(viz_collected.get('pr_auc', 0) or 0),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'detection_method': detection_method,
        'threshold': float(best_threshold),
    }

    # Per-type detection rates
    per_type = viz_collected.get('per_type', {})
    paper_metrics['per_type_detection'] = {
        t: {'rate': float(v['rate']), 'total': int(v['total']),
            'detected': int(v['detected'])}
        for t, v in per_type.items()
    }

    # Ablation F1
    ablation = viz_collected.get('ablation', {})
    paper_metrics['ablation_f1'] = {
        k.replace('\n', ' '): float(v) for k, v in ablation.items()
    }

    # Corrected d-prime per component
    dprime = viz_collected.get('dprime', {})
    paper_metrics['dprime'] = {k: float(v) for k, v in dprime.items()}

    with open(f"{output_dir}/paper_metrics.json", 'w') as f:
        json.dump(paper_metrics, f, indent=2)
    print(f"✓ Saved paper_metrics.json")

    # ========================================================================
    # DETAILED EXCEL REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING DETAILED EXCEL REPORT")
    print("="*80)

    generate_detailed_results_excel(f"{output_dir}/results.json", output_dir)

    # ========================================================================
    # FINAL SUMMARY TABLE
    # ========================================================================
    print("\n" + "="*80)
    print("PAPER METRICS SUMMARY")
    print("="*80)
    print(f"  {'Metric':<20s} {'Value':>10s}")
    print(f"  {'-'*20} {'-'*10}")
    for k in ['accuracy', 'precision', 'recall', 'specificity', 'f1',
              'roc_auc', 'pr_auc']:
        print(f"  {k:<20s} {paper_metrics[k]:>10.4f}")
    print(f"\n  Confusion Matrix:")
    cm = paper_metrics['confusion_matrix']
    print(f"    TP: {cm['tp']:>5d}   FP: {cm['fp']:>5d}")
    print(f"    FN: {cm['fn']:>5d}   TN: {cm['tn']:>5d}")

    if paper_metrics['per_type_detection']:
        print(f"\n  Per-Type Detection Rates:")
        for t, v in paper_metrics['per_type_detection'].items():
            print(f"    {t:<25s}  {v['rate']:.1%}  (detected {v['detected']}/{v['total']})")

    if paper_metrics['ablation_f1']:
        print(f"\n  Ablation F1 Scores:")
        for config_name, f1_val in paper_metrics['ablation_f1'].items():
            print(f"    {config_name:<25s}  {f1_val:.3f}")

    if paper_metrics['dprime']:
        print(f"\n  Corrected d' (pooled-variance):")
        for comp, dp in paper_metrics['dprime'].items():
            print(f"    {comp:<20s}  {FS.d_prime_label(dp)}")

    print(f"\n  Detection Method: {detection_method}")
    print("="*80)

    # Performance assessment
    if f1 >= 0.70:
        print("🎉 EXCELLENT! F1 ≥ 70% — Target achieved!")
    elif f1 >= 0.60:
        print("✅ GOOD! F1 ≥ 60%")
    elif f1 >= 0.50:
        print("✓ ACCEPTABLE — F1 ≥ 50%")
    else:
        print("⚠️  NEEDS IMPROVEMENT — F1 < 50%")
    print(f"\n✓ All outputs saved to {output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()



