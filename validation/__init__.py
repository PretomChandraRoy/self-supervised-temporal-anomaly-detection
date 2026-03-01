"""Validation package for anomaly detection"""

from .synthetic_anomalies import (
    inject_price_spikes,
    inject_volatility_spikes,
    inject_trend_breaks,
    inject_combined_anomalies,
    evaluate_detection,
    print_evaluation_results,
    create_validation_report
)

__all__ = [
    'inject_price_spikes',
    'inject_volatility_spikes',
    'inject_trend_breaks',
    'inject_combined_anomalies',
    'evaluate_detection',
    'print_evaluation_results',
    'create_validation_report'
]

