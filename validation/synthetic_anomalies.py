"""
Synthetic Anomaly Validation
Inject known anomalies to validate detection performance
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def inject_price_spikes(data, n_anomalies=50, spike_range=(1.05, 1.20)):
    """
    Inject abnormal price spikes

    Args:
        data: DataFrame with OHLC columns
        n_anomalies: number of anomalies to inject
        spike_range: (min, max) factor for price spikes

    Returns:
        data: modified DataFrame
        anomaly_indices: indices where anomalies were injected
    """
    data = data.copy()
    anomaly_indices = np.random.choice(len(data), n_anomalies, replace=False)
    anomaly_mask = np.zeros(len(data), dtype=bool)
    anomaly_mask[anomaly_indices] = True

    for idx in anomaly_indices:
        if idx < len(data):
            spike_factor = np.random.uniform(*spike_range)

            # Spike in close and high prices
            data.loc[data.index[idx], 'close'] *= spike_factor
            data.loc[data.index[idx], 'high'] *= spike_factor

            # Ensure high >= close
            if data.loc[data.index[idx], 'high'] < data.loc[data.index[idx], 'close']:
                data.loc[data.index[idx], 'high'] = data.loc[data.index[idx], 'close']

    print(f"✓ Injected {n_anomalies} price spike anomalies")
    return data, anomaly_mask


def inject_volatility_spikes(data, n_anomalies=50, vol_factor_range=(3.0, 5.0)):
    """
    Inject abnormal volatility spikes

    Args:
        data: DataFrame with OHLC columns
        n_anomalies: number of anomalies to inject
        vol_factor_range: (min, max) factor for volatility increase

    Returns:
        data: modified DataFrame
        anomaly_indices: indices where anomalies were injected
    """
    data = data.copy()
    anomaly_indices = np.random.choice(len(data), n_anomalies, replace=False)
    anomaly_mask = np.zeros(len(data), dtype=bool)
    anomaly_mask[anomaly_indices] = True

    for idx in anomaly_indices:
        if idx < len(data) - 1:
            vol_factor = np.random.uniform(*vol_factor_range)
            close = data.loc[data.index[idx], 'close']

            # Create wide bar (high volatility)
            data.loc[data.index[idx], 'high'] = close * (1 + 0.03 * vol_factor)
            data.loc[data.index[idx], 'low'] = close * (1 - 0.03 * vol_factor)

    print(f"✓ Injected {n_anomalies} volatility spike anomalies")
    return data, anomaly_mask


def inject_trend_breaks(data, n_anomalies=30, break_magnitude=0.15):
    """
    Inject sudden trend breaks (reversals)

    Args:
        data: DataFrame with OHLC columns
        n_anomalies: number of anomalies to inject
        break_magnitude: magnitude of the break

    Returns:
        data: modified DataFrame
        anomaly_indices: indices where anomalies were injected
    """
    data = data.copy()
    anomaly_indices = np.random.choice(range(50, len(data)-50), n_anomalies, replace=False)
    anomaly_mask = np.zeros(len(data), dtype=bool)
    anomaly_mask[anomaly_indices] = True

    for idx in anomaly_indices:
        # Check recent trend
        recent_prices = data.loc[data.index[idx-10:idx], 'close'].values
        if len(recent_prices) > 5:
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]

            # Reverse the trend suddenly
            if trend > 0:  # Uptrend -> sudden drop
                data.loc[data.index[idx], 'close'] *= (1 - break_magnitude)
                data.loc[data.index[idx], 'low'] *= (1 - break_magnitude)
            else:  # Downtrend -> sudden spike
                data.loc[data.index[idx], 'close'] *= (1 + break_magnitude)
                data.loc[data.index[idx], 'high'] *= (1 + break_magnitude)

    print(f"✓ Injected {n_anomalies} trend break anomalies")
    return data, anomaly_mask


def inject_combined_anomalies(data, anomaly_ratio=0.05, seed=42):
    """
    Inject multiple types of anomalies

    Args:
        data: DataFrame with OHLC columns
        anomaly_ratio: fraction of data to make anomalous
        seed: random seed

    Returns:
        data: modified DataFrame
        anomaly_mask: boolean mask of anomalies
    """
    np.random.seed(seed)

    total_anomalies = int(len(data) * anomaly_ratio)
    n_per_type = total_anomalies // 3

    # Initialize mask
    anomaly_mask = np.zeros(len(data), dtype=bool)

    # Inject price spikes
    data, mask1 = inject_price_spikes(data, n_per_type)
    anomaly_mask |= mask1

    # Inject volatility spikes
    data, mask2 = inject_volatility_spikes(data, n_per_type)
    anomaly_mask |= mask2

    # Inject trend breaks
    data, mask3 = inject_trend_breaks(data, n_per_type)
    anomaly_mask |= mask3

    print(f"\n✓ Total anomalies injected: {anomaly_mask.sum()} ({anomaly_mask.sum()/len(data)*100:.2f}%)")

    return data, anomaly_mask


def evaluate_detection(predictions, ground_truth):
    """
    Calculate detection metrics

    Args:
        predictions: boolean array of predictions
        ground_truth: boolean array of true anomalies

    Returns:
        metrics: dictionary of evaluation metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()

    # Calculate metrics
    tp = np.sum(predictions & ground_truth)
    fp = np.sum(predictions & ~ground_truth)
    fn = np.sum(~predictions & ground_truth)
    tn = np.sum(~predictions & ~ground_truth)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    accuracy = (tp + tn) / len(predictions)

    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'fpr': fpr,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }

    return metrics


def print_evaluation_results(metrics):
    """Print formatted evaluation results"""
    print("\n" + "="*60)
    print("ANOMALY DETECTION EVALUATION")
    print("="*60)
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1 Score:   {metrics['f1']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"FPR:        {metrics['fpr']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']:5d}  |  FP: {metrics['fp']:5d}")
    print(f"  FN: {metrics['fn']:5d}  |  TN: {metrics['tn']:5d}")
    print("="*60)

    # Interpretation
    if metrics['f1'] >= 0.8:
        print("✅ EXCELLENT detection performance!")
    elif metrics['f1'] >= 0.6:
        print("✓ GOOD detection performance")
    elif metrics['f1'] >= 0.4:
        print("⚠️ MODERATE detection performance")
    else:
        print("❌ POOR detection performance - needs improvement")


def create_validation_report(metrics, save_path='validation_report.txt'):
    """Save validation report to file"""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SYNTHETIC ANOMALY VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")

        f.write("Metrics:\n")
        f.write(f"  Precision:  {metrics['precision']:.4f}\n")
        f.write(f"  Recall:     {metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:   {metrics['f1']:.4f}\n")
        f.write(f"  Accuracy:   {metrics['accuracy']:.4f}\n")
        f.write(f"  FPR:        {metrics['fpr']:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {metrics['tp']:5d}  |  FP: {metrics['fp']:5d}\n")
        f.write(f"  FN: {metrics['fn']:5d}  |  TN: {metrics['tn']:5d}\n\n")

        f.write("="*60 + "\n")

    print(f"✓ Validation report saved to: {save_path}")


if __name__ == '__main__':
    # Example usage
    print("Synthetic Anomaly Validation Module")
    print("Import this module to validate your anomaly detector")
    print("\nExample:")
    print("  from validation.synthetic_anomalies import inject_combined_anomalies, evaluate_detection")
    print("  data_with_anomalies, ground_truth = inject_combined_anomalies(clean_data)")
    print("  metrics = evaluate_detection(predictions, ground_truth)")

