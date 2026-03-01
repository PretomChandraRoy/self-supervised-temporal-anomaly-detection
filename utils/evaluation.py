"""
Evaluation metrics and utilities for anomaly detection
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import pandas as pd


def compute_anomaly_statistics(scores, is_anomaly):
    """
    Compute basic statistics for anomaly scores
    Args:
        scores: anomaly scores array
        is_anomaly: boolean mask
    Returns:
        stats: dictionary with statistics
    """
    stats = {
        'n_total': len(scores),
        'n_anomalies': int(is_anomaly.sum()),
        'anomaly_rate': float(is_anomaly.mean()),
        'mean_score': float(scores.mean()),
        'std_score': float(scores.std()),
        'min_score': float(scores.min()),
        'max_score': float(scores.max()),
        'median_score': float(np.median(scores)),
        'q25_score': float(np.percentile(scores, 25)),
        'q75_score': float(np.percentile(scores, 75)),
    }

    if is_anomaly.sum() > 0:
        stats['mean_anomaly_score'] = float(scores[is_anomaly].mean())
        stats['min_anomaly_score'] = float(scores[is_anomaly].min())
        stats['max_anomaly_score'] = float(scores[is_anomaly].max())

    if (~is_anomaly).sum() > 0:
        stats['mean_normal_score'] = float(scores[~is_anomaly].mean())
        stats['std_normal_score'] = float(scores[~is_anomaly].std())

    return stats


def evaluate_anomaly_detection(y_true, y_pred, scores=None):
    """
    Evaluate anomaly detection performance with labeled data
    Args:
        y_true: true labels (1 for anomaly, 0 for normal)
        y_pred: predicted labels
        scores: optional anomaly scores for AUC metrics
    Returns:
        metrics: dictionary with evaluation metrics
    """
    metrics = {}

    # Basic metrics
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # False positive rate
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    # AUC metrics if scores provided
    if scores is not None:
        try:
            metrics['auroc'] = roc_auc_score(y_true, scores)
            metrics['average_precision'] = average_precision_score(y_true, scores)
        except:
            metrics['auroc'] = None
            metrics['average_precision'] = None

    return metrics


def print_evaluation_report(metrics):
    """
    Print formatted evaluation report
    Args:
        metrics: dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("ANOMALY DETECTION EVALUATION REPORT")
    print("="*60)

    print("\n📊 Classification Metrics:")
    print(f"  Precision:  {metrics.get('precision', 0):.4f}")
    print(f"  Recall:     {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:   {metrics.get('f1', 0):.4f}")
    print(f"  Accuracy:   {metrics.get('accuracy', 0):.4f}")
    print(f"  Specificity: {metrics.get('specificity', 0):.4f}")

    print("\n📈 Confusion Matrix:")
    print(f"  True Positives:  {metrics.get('true_positives', 0)}")
    print(f"  False Positives: {metrics.get('false_positives', 0)}")
    print(f"  True Negatives:  {metrics.get('true_negatives', 0)}")
    print(f"  False Negatives: {metrics.get('false_negatives', 0)}")

    if metrics.get('auroc') is not None:
        print("\n🎯 AUC Metrics:")
        print(f"  AUROC:              {metrics.get('auroc', 0):.4f}")
        print(f"  Average Precision:  {metrics.get('average_precision', 0):.4f}")

    print("="*60 + "\n")


def analyze_anomaly_patterns(data, is_anomaly, window_size=10):
    """
    Analyze patterns in detected anomalies
    Args:
        data: time-series data (price, volume, etc.)
        is_anomaly: boolean mask
        window_size: window for pattern analysis
    Returns:
        patterns: dictionary with pattern statistics
    """
    patterns = {}

    # Anomaly clustering (consecutive anomalies)
    anomaly_indices = np.where(is_anomaly)[0]

    if len(anomaly_indices) == 0:
        return {
            'n_clusters': 0,
            'mean_cluster_size': 0,
            'max_cluster_size': 0,
            'isolated_anomalies': 0
        }

    # Find consecutive groups
    clusters = []
    current_cluster = [anomaly_indices[0]]

    for i in range(1, len(anomaly_indices)):
        if anomaly_indices[i] - anomaly_indices[i-1] == 1:
            current_cluster.append(anomaly_indices[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [anomaly_indices[i]]
    clusters.append(current_cluster)

    cluster_sizes = [len(c) for c in clusters]

    patterns['n_clusters'] = len(clusters)
    patterns['mean_cluster_size'] = np.mean(cluster_sizes)
    patterns['max_cluster_size'] = max(cluster_sizes)
    patterns['isolated_anomalies'] = sum(1 for s in cluster_sizes if s == 1)
    patterns['cluster_size_distribution'] = cluster_sizes

    # Temporal distribution
    if len(data) > 0:
        # Split into periods
        n_periods = min(10, len(data) // 100)
        if n_periods > 0:
            period_size = len(data) // n_periods
            period_anomaly_rates = []

            for i in range(n_periods):
                start = i * period_size
                end = start + period_size if i < n_periods - 1 else len(data)
                period_mask = is_anomaly[start:end]
                period_anomaly_rates.append(period_mask.mean())

            patterns['temporal_distribution'] = period_anomaly_rates
            patterns['temporal_variance'] = np.var(period_anomaly_rates)

    return patterns


def create_anomaly_report(
    scores,
    is_anomaly,
    prices=None,
    timestamps=None,
    top_k=20
):
    """
    Create detailed anomaly report
    Args:
        scores: anomaly scores
        is_anomaly: boolean mask
        prices: optional price data
        timestamps: optional timestamps
        top_k: number of top anomalies to report
    Returns:
        report_df: DataFrame with top anomalies
    """
    anomaly_indices = np.where(is_anomaly)[0]

    if len(anomaly_indices) == 0:
        return pd.DataFrame()

    # Get top-k highest scoring anomalies
    top_indices = anomaly_indices[np.argsort(scores[anomaly_indices])[-top_k:]][::-1]

    report_data = {
        'index': top_indices,
        'anomaly_score': scores[top_indices]
    }

    if prices is not None:
        report_data['price'] = prices[top_indices]

        # Price change around anomaly
        price_changes = []
        for idx in top_indices:
            if idx > 0 and idx < len(prices) - 1:
                before = prices[idx - 1]
                after = prices[idx + 1]
                change = ((after - before) / before) * 100
                price_changes.append(change)
            else:
                price_changes.append(0)

        report_data['price_change_pct'] = price_changes

    if timestamps is not None:
        report_data['timestamp'] = timestamps[top_indices]

    report_df = pd.DataFrame(report_data)

    return report_df


def calculate_financial_impact(
    prices,
    is_anomaly,
    transaction_cost=0.001
):
    """
    Calculate potential financial impact of anomalies
    Args:
        prices: price time series
        is_anomaly: boolean mask
        transaction_cost: cost per trade (fraction)
    Returns:
        impact: dictionary with financial metrics
    """
    impact = {}

    anomaly_indices = np.where(is_anomaly)[0]

    if len(anomaly_indices) == 0:
        return {
            'n_signals': 0,
            'total_return': 0,
            'avg_return_per_signal': 0
        }

    # Calculate returns around anomalies
    returns = []

    for idx in anomaly_indices:
        if idx > 0 and idx < len(prices) - 1:
            entry_price = prices[idx]
            exit_price = prices[idx + 1]

            # Long position
            ret = ((exit_price - entry_price) / entry_price) - (2 * transaction_cost)
            returns.append(ret)

    if len(returns) > 0:
        impact['n_signals'] = len(returns)
        impact['total_return'] = sum(returns)
        impact['avg_return_per_signal'] = np.mean(returns)
        impact['std_return'] = np.std(returns)
        impact['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        impact['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
    else:
        impact['n_signals'] = 0
        impact['total_return'] = 0
        impact['avg_return_per_signal'] = 0
        impact['std_return'] = 0
        impact['sharpe_ratio'] = 0
        impact['win_rate'] = 0

    return impact

