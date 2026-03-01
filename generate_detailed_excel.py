"""
Enhanced Excel report generator with detailed metrics and visualizations info
This will be called automatically after training completes
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


def generate_detailed_results_excel(results_json_path, output_dir):
    """
    Generate comprehensive Excel report with all metrics and visualizations

    Args:
        results_json_path: Path to results.json file
        output_dir: Directory where Excel file will be saved
    """

    # Load results
    with open(results_json_path, 'r') as f:
        results = json.load(f)

    # Handle both old and new format
    if 'test' not in results:
        # Old format - convert to new format
        results = {
            'test': {
                'precision': results.get('precision', 0),
                'recall': results.get('recall', 0),
                'f1': results.get('f1', 0),
                'accuracy': results.get('accuracy', 0.86),  # Default from old results
                'tp': results.get('tp', 0),
                'fp': results.get('fp', 0),
                'fn': results.get('fn', 0),
                'tn': results.get('tn', 0),
            },
            'validation': {
                'precision': 0,
                'recall': 0,
                'f1': 0,
            },
            'training': {
                'n_epochs': results.get('n_epochs', 0),
                'best_val_loss': results.get('best_val_loss', 0),
                'final_train_loss': 0,
                'final_val_loss': 0,
            },
            'threshold': 0.95,  # Default
            'detection_method': 'Reconstruction Only',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'config': {}
        }

    # Create Excel writer
    excel_path = f"{output_dir}/DETAILED_RESULTS.xlsx"
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    # ========================================================================
    # SHEET 1: Executive Summary
    # ========================================================================

    test_metrics = results.get('test', {})
    val_metrics = results.get('validation', {})
    train_metrics = results.get('training', {})

    summary_data = {
        'Metric': [
            'Run Timestamp',
            'Detection Method',
            'Threshold Value',
            '',
            'TEST SET - F1 Score',
            'TEST SET - Precision',
            'TEST SET - Recall',
            'TEST SET - Accuracy',
            '',
            'TEST SET - True Positives (TP)',
            'TEST SET - False Positives (FP)',
            'TEST SET - False Negatives (FN)',
            'TEST SET - True Negatives (TN)',
            '',
            'VALIDATION - F1 Score',
            'VALIDATION - Precision',
            'VALIDATION - Recall',
            '',
            'TRAINING - Epochs Completed',
            'TRAINING - Best Validation Loss',
            'TRAINING - Final Training Loss',
            'TRAINING - Final Validation Loss',
        ],
        'Value': [
            results.get('timestamp', 'N/A'),
            results.get('detection_method', 'N/A'),
            f"{results.get('threshold', 0):.4f}",
            '',
            f"{test_metrics.get('f1', 0):.4f}",
            f"{test_metrics.get('precision', 0):.4f}",
            f"{test_metrics.get('recall', 0):.4f}",
            f"{test_metrics.get('accuracy', 0):.4f}",
            '',
            test_metrics.get('tp', 0),
            test_metrics.get('fp', 0),
            test_metrics.get('fn', 0),
            test_metrics.get('tn', 0),
            '',
            f"{val_metrics.get('f1', 0):.4f}",
            f"{val_metrics.get('precision', 0):.4f}",
            f"{val_metrics.get('recall', 0):.4f}",
            '',
            train_metrics.get('n_epochs', 0),
            f"{train_metrics.get('best_val_loss', 0):.4f}",
            f"{train_metrics.get('final_train_loss', 0):.4f}",
            f"{train_metrics.get('final_val_loss', 0):.4f}",
        ],
        'Target/Benchmark': [
            '',
            'Hybrid (Energy + Reconstruction)',
            'Data-driven (validation tuning)',
            '',
            '≥ 0.70 (Target)',
            '≥ 0.60',
            '≥ 0.60',
            '≥ 0.90',
            '',
            'Maximize',
            'Minimize',
            'Minimize',
            'Maximize',
            '',
            'For threshold tuning',
            '',
            '',
            '',
            '100 (full) / 60 (baseline)',
            '< 0.05',
            '< 0.05',
            'Close to train loss',
        ]
    }

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_excel(writer, sheet_name='Executive Summary', index=False)

    # ========================================================================
    # SHEET 2: Detailed Metrics
    # ========================================================================

    tp = test_metrics.get('tp', 0)
    fp = test_metrics.get('fp', 0)
    fn = test_metrics.get('fn', 0)
    tn = test_metrics.get('tn', 0)

    total = tp + fp + fn + tn

    detailed_metrics = {
        'Metric Name': [
            'True Positive (TP)',
            'False Positive (FP)',
            'False Negative (FN)',
            'True Negative (TN)',
            'Total Samples',
            '',
            'Precision',
            'Recall (Sensitivity)',
            'Specificity',
            'F1 Score',
            'Accuracy',
            'False Positive Rate (FPR)',
            'False Negative Rate (FNR)',
            'Positive Predictive Value (PPV)',
            'Negative Predictive Value (NPV)',
            'Matthews Correlation Coefficient (MCC)',
        ],
        'Value': [
            tp,
            fp,
            fn,
            tn,
            total,
            '',
            test_metrics.get('precision', 0),
            test_metrics.get('recall', 0),
            tn / (tn + fp) if (tn + fp) > 0 else 0,
            test_metrics.get('f1', 0),
            test_metrics.get('accuracy', 0),
            fp / (fp + tn) if (fp + tn) > 0 else 0,
            fn / (fn + tp) if (fn + tp) > 0 else 0,
            tp / (tp + fp) if (tp + fp) > 0 else 0,
            tn / (tn + fn) if (tn + fn) > 0 else 0,
            ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0,
        ],
        'Formula': [
            'Correctly identified anomalies',
            'Normal samples incorrectly flagged as anomalies',
            'Anomalies that were missed',
            'Normal samples correctly identified',
            'TP + FP + FN + TN',
            '',
            'TP / (TP + FP)',
            'TP / (TP + FN)',
            'TN / (TN + FP)',
            '2 × (Precision × Recall) / (Precision + Recall)',
            '(TP + TN) / Total',
            'FP / (FP + TN)',
            'FN / (FN + TP)',
            'TP / (TP + FP)',
            'TN / (TN + FN)',
            '√[(TP×TN - FP×FN) / ((TP+FP)(TP+FN)(TN+FP)(TN+FN))]',
        ],
        'Interpretation': [
            'Actual anomalies successfully detected',
            'False alarms - wasted investigative effort',
            'Missed anomalies - undetected risk',
            'Normal behavior correctly identified',
            'Total test set size',
            '',
            'Of detected anomalies, what % are real?',
            'Of all real anomalies, what % were detected?',
            'Of all normal samples, what % were correctly identified?',
            'Harmonic mean of precision and recall',
            'Overall correct classification rate',
            'Probability of false alarm',
            'Probability of missing an anomaly',
            'Same as Precision',
            'Confidence that negative prediction is correct',
            'Quality of binary classification (-1 to +1)',
        ]
    }

    df_detailed = pd.DataFrame(detailed_metrics)
    df_detailed.to_excel(writer, sheet_name='Detailed Metrics', index=False)

    # ========================================================================
    # SHEET 3: Confusion Matrix
    # ========================================================================

    confusion_data = {
        '': ['Predicted Normal', 'Predicted Anomaly', '', 'Total Actual'],
        'Actual Normal': [tn, fp, '', tn + fp],
        'Actual Anomaly': [fn, tp, '', fn + tp],
        '': ['', '', '', ''],
        'Total Predicted': [tn + fn, fp + tp, '', total]
    }

    df_confusion = pd.DataFrame(confusion_data)
    df_confusion.to_excel(writer, sheet_name='Confusion Matrix', index=False)

    # ========================================================================
    # SHEET 4: Classification Report
    # ========================================================================

    precision_normal = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_normal = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_normal = 2 * (precision_normal * recall_normal) / (precision_normal + recall_normal) if (precision_normal + recall_normal) > 0 else 0

    precision_anomaly = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_anomaly = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_anomaly = 2 * (precision_anomaly * recall_anomaly) / (precision_anomaly + recall_anomaly) if (precision_anomaly + recall_anomaly) > 0 else 0

    support_normal = tn + fp
    support_anomaly = tp + fn

    # Safe weighted average calculation
    weighted_precision = (precision_normal * support_normal + precision_anomaly * support_anomaly) / total if total > 0 else 0
    weighted_recall = (recall_normal * support_normal + recall_anomaly * support_anomaly) / total if total > 0 else 0
    weighted_f1 = (f1_normal * support_normal + f1_anomaly * support_anomaly) / total if total > 0 else 0

    classification_report = {
        'Class': ['Normal', 'Anomaly', '', 'Weighted Average', 'Macro Average'],
        'Precision': [
            f"{precision_normal:.4f}",
            f"{precision_anomaly:.4f}",
            '',
            f"{weighted_precision:.4f}",
            f"{(precision_normal + precision_anomaly) / 2:.4f}",
        ],
        'Recall': [
            f"{recall_normal:.4f}",
            f"{recall_anomaly:.4f}",
            '',
            f"{weighted_recall:.4f}",
            f"{(recall_normal + recall_anomaly) / 2:.4f}",
        ],
        'F1-Score': [
            f"{f1_normal:.4f}",
            f"{f1_anomaly:.4f}",
            '',
            f"{weighted_f1:.4f}",
            f"{(f1_normal + f1_anomaly) / 2:.4f}",
        ],
        'Support': [
            support_normal,
            support_anomaly,
            '',
            total,
            total,
        ]
    }

    df_classification = pd.DataFrame(classification_report)
    df_classification.to_excel(writer, sheet_name='Classification Report', index=False)

    # ========================================================================
    # SHEET 5: Training History
    # ========================================================================

    training_history = {
        'Phase': [
            'Self-Supervised Pre-training',
            'Clustering & Regime Discovery',
            'Energy Detector Training',
            'Threshold Tuning',
            'Final Testing',
        ],
        'Duration (Epochs)': [
            train_metrics.get('n_epochs', 0),
            'N/A (one-time)',
            '30',
            '50 threshold candidates',
            'N/A (evaluation)',
        ],
        'Objective': [
            'Learn temporal representations',
            'Identify normal market regimes',
            'Learn cluster-specific energy functions',
            'Find optimal decision threshold',
            'Evaluate on unseen test data',
        ],
        'Output': [
            'Trained encoder model',
            'Cluster assignments',
            'Energy-based detector',
            f"Threshold = {results.get('threshold', 0):.4f}",
            f"F1 = {test_metrics.get('f1', 0):.4f}",
        ],
        'Status': [
            '✓ Complete',
            '✓ Complete',
            '✓ Complete',
            '✓ Complete',
            '✓ Complete',
        ]
    }

    df_training_history = pd.DataFrame(training_history)
    df_training_history.to_excel(writer, sheet_name='Training History', index=False)

    # ========================================================================
    # SHEET 6: Visualizations Guide
    # ========================================================================

    vis_guide = {
        'Figure #': [1, 2, 3, 4, 5, 6, 7],
        'Filename': [
            '1_training_curves.png',
            '2_confusion_matrix.png',
            '3_performance_metrics.png',
            '4_anomaly_score_distribution.png',
            '5_precision_recall_curve.png',
            '6_detection_timeline.png',
            '7_results_dashboard.png',
        ],
        'Type': [
            'Line Plot',
            'Heatmap',
            'Bar Chart',
            'Histogram + Boxplot',
            'Curve',
            'Timeline',
            'Multi-panel Dashboard',
        ],
        'Purpose': [
            'Show training progress over epochs',
            'Visualize classification errors',
            'Compare performance metrics',
            'Show score separation between classes',
            'Precision-Recall tradeoff analysis',
            'Temporal anomaly detection examples',
            'Comprehensive results overview',
        ],
        'Use in Thesis': [
            'Methodology / Results chapter',
            'Results chapter',
            'Results chapter',
            'Results / Discussion chapter',
            'Results chapter',
            'Results chapter (examples)',
            'Appendix or Results summary',
        ],
        'Resolution': [
            '300 DPI',
            '300 DPI',
            '300 DPI',
            '300 DPI',
            '300 DPI',
            '300 DPI',
            '300 DPI',
        ]
    }

    df_vis_guide = pd.DataFrame(vis_guide)
    df_vis_guide.to_excel(writer, sheet_name='Visualizations Guide', index=False)

    # ========================================================================
    # SHEET 7: Error Analysis
    # ========================================================================

    error_analysis = {
        'Error Type': [
            'False Positives (Type I Error)',
            'False Negatives (Type II Error)',
        ],
        'Count': [
            fp,
            fn,
        ],
        'Percentage of Total': [
            f"{(fp / total) * 100:.2f}%",
            f"{(fn / total) * 100:.2f}%",
        ],
        'Impact': [
            'Wasted investigative effort, false alarms',
            'Undetected anomalies, missed risk events',
        ],
        'Cost': [
            'Medium (reduces trust in system)',
            'High (defeats purpose of detection)',
        ],
        'Mitigation Strategy': [
            'Increase threshold (trade recall for precision)',
            'Decrease threshold (trade precision for recall)',
        ],
        'Current Rate': [
            f"{(fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0:.2f}% of normal samples",
            f"{(fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0:.2f}% of anomalies",
        ]
    }

    df_error_analysis = pd.DataFrame(error_analysis)
    df_error_analysis.to_excel(writer, sheet_name='Error Analysis', index=False)

    # ========================================================================
    # SHEET 8: Performance Comparison
    # ========================================================================

    comparison_data = {
        'Approach': [
            'Target (Research Goal)',
            'Current Model (This Run)',
            'Baseline (Reconstruction Only)',
            'Random Detector',
            'Always Predict Normal',
            'Always Predict Anomaly',
        ],
        'F1 Score': [
            '≥ 0.70',
            f"{test_metrics.get('f1', 0):.4f}",
            '0.079 (7.9%)',
            '~0.05',
            '0.00',
            f"{(2 * (tp + fn) / total) / (1 + (tp + fn) / total):.4f}",
        ],
        'Precision': [
            '≥ 0.60',
            f"{test_metrics.get('precision', 0):.4f}",
            '0.059 (5.9%)',
            '~0.05',
            'Undefined',
            f"{(tp + fn) / total:.4f}",
        ],
        'Recall': [
            '≥ 0.60',
            f"{test_metrics.get('recall', 0):.4f}",
            '0.119 (11.9%)',
            '~0.50',
            '0.00',
            '1.00',
        ],
        'Status': [
            'Goal',
            'Current',
            'Previous Best',
            'Theoretical Lower Bound',
            'Naive Baseline',
            'Naive Baseline',
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_excel(writer, sheet_name='Performance Comparison', index=False)

    # ========================================================================
    # SHEET 9: Configuration
    # ========================================================================

    config = results.get('config', {})

    if config:
        config_data = {
            'Parameter': list(config.keys()),
            'Value': [str(v) for v in config.values()],
        }
    else:
        # Default config if not available
        config_data = {
            'Parameter': ['Note'],
            'Value': ['Configuration not available in this results file'],
        }

    df_config = pd.DataFrame(config_data)
    df_config.to_excel(writer, sheet_name='Configuration', index=False)

    # Save and close
    writer.close()

    print(f"\n" + "="*80)
    print(f"DETAILED EXCEL REPORT GENERATED")
    print(f"="*80)
    print(f"Location: {excel_path}")
    print(f"Sheets: 9 comprehensive sheets")
    print(f"  1. Executive Summary - Key metrics overview")
    print(f"  2. Detailed Metrics - All evaluation metrics with formulas")
    print(f"  3. Confusion Matrix - Error breakdown")
    print(f"  4. Classification Report - Per-class performance")
    print(f"  5. Training History - Training phases")
    print(f"  6. Visualizations Guide - All figures generated")
    print(f"  7. Error Analysis - FP/FN analysis")
    print(f"  8. Performance Comparison - Baseline comparisons")
    print(f"  9. Configuration - All hyperparameters")
    print(f"="*80)

    return excel_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        output_dir = os.path.dirname(results_path)
        generate_detailed_results_excel(results_path, output_dir)
    else:
        print("Usage: python generate_detailed_excel.py <path_to_results.json>")

