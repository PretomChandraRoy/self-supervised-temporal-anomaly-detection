"""
Example: Validate Anomaly Detection with Synthetic Anomalies
This script demonstrates how to properly validate the anomaly detector
"""

import torch
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.models.temporal_transformer import SelfSupervisedTemporalModel
from anomaly_detection.models.clustering import DensityAwareClustering
from anomaly_detection.models.anomaly_detector import (
    EnergyBasedAnomalyDetector,
    ReconstructionBasedDetector,
    HybridAnomalyDetector
)
from anomaly_detection.training.trainer import AnomalyDetectionTrainer
from anomaly_detection.data.preprocessing import (
    FinancialDataPreprocessor,
    load_forex_data,
    split_train_test
)
from anomaly_detection.validation.synthetic_anomalies import (
    inject_combined_anomalies,
    evaluate_detection,
    print_evaluation_results,
    create_validation_report
)


def validate_with_synthetic_anomalies():
    """
    Validate anomaly detector using synthetic anomalies
    """

    print("="*80)
    print("VALIDATION: Synthetic Anomaly Detection")
    print("="*80)

    # Configuration
    DATA_PATH = '../forexPredictor/H4_EURUSD_2015.csv'
    CHECKPOINT_PATH = 'example_outputs/checkpoints/final_model.pt'
    WINDOW_SIZE = 60
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    ANOMALY_RATIO = 0.05  # 5% anomalies

    # Step 1: Load clean data
    print("\n[1/6] Loading clean data...")
    df = load_forex_data(DATA_PATH)

    # Use only recent data for faster validation
    df_validation = df.iloc[-5000:].copy()

    print(f"✓ Loaded {len(df_validation)} samples for validation")

    # Step 2: Inject synthetic anomalies
    print("\n[2/6] Injecting synthetic anomalies...")
    df_with_anomalies, ground_truth = inject_combined_anomalies(
        df_validation,
        anomaly_ratio=ANOMALY_RATIO,
        seed=42
    )

    print(f"✓ Ground truth anomalies: {ground_truth.sum()} / {len(ground_truth)}")

    # Step 3: Prepare data
    print("\n[3/6] Preprocessing data...")
    preprocessor = FinancialDataPreprocessor(
        window_size=WINDOW_SIZE,
        stride=1,
        add_technical_indicators=True,
        scaler_type='robust',
        clip_outliers=True
    )

    sequences, feature_names = preprocessor.prepare_data(df_with_anomalies, fit_scaler=True)
    n_features = sequences.shape[2]

    # Align ground truth with sequences
    # Each sequence ends at position i+window_size
    ground_truth_aligned = ground_truth[WINDOW_SIZE-1:][:len(sequences)]

    print(f"✓ Created {len(sequences)} sequences")
    print(f"✓ Aligned ground truth: {ground_truth_aligned.sum()} anomalies")

    # Step 4: Load trained model
    print("\n[4/6] Loading trained model...")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️  Checkpoint not found: {CHECKPOINT_PATH}")
        print("Please train the model first using example.py")
        return None

    model = SelfSupervisedTemporalModel(
        n_features=n_features,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
        reconstruction_weight=0.1,
        contrastive_weight=1.0
    )

    clustering = DensityAwareClustering(n_clusters=10, method='kmeans')
    energy_detector = EnergyBasedAnomalyDetector(embedding_dim=128, n_clusters=10)
    recon_detector = ReconstructionBasedDetector(model.reconstructor)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    if 'energy_detector_state_dict' in checkpoint:
        energy_detector.load_state_dict(checkpoint['energy_detector_state_dict'])

    model = model.to(DEVICE)
    model.eval()
    energy_detector = energy_detector.to(DEVICE)

    print("✓ Model loaded successfully")

    # Step 5: Detect anomalies
    print("\n[5/6] Running anomaly detection...")

    test_data = torch.FloatTensor(sequences).to(DEVICE)

    with torch.no_grad():
        # Extract embeddings
        test_embeddings = model.get_embeddings(test_data)

        # Fit clustering on validation embeddings
        cluster_labels = clustering.fit(test_embeddings.cpu().numpy())
        cluster_tensor = torch.LongTensor(cluster_labels).to(DEVICE)

    # Try hybrid detector first, fall back to reconstruction-only if it fails
    use_reconstruction_only = False

    try:
        # Create hybrid detector
        hybrid = HybridAnomalyDetector(
            energy_detector=energy_detector,
            reconstruction_detector=recon_detector,
            fusion_method='weighted_sum',
            energy_weight=0.5,
            reconstruction_weight=0.5
        )

        # Predict
        scores, is_anomaly, details = hybrid.predict(test_data, test_embeddings, cluster_tensor)

        # Check if scores are NaN
        if torch.isnan(scores).any():
            print("⚠️  Energy detector produced NaN - falling back to reconstruction-only")
            use_reconstruction_only = True
        else:
            print("✓ Using hybrid detection (energy + reconstruction)")

    except Exception as e:
        print(f"⚠️  Hybrid detector failed: {e}")
        print("⚠️  Falling back to reconstruction-only detection")
        use_reconstruction_only = True

    # Fall back to reconstruction-only if needed
    if use_reconstruction_only:
        print("\n=== Using Reconstruction-Only Detection ===")

        # Compute reconstruction errors
        all_errors = []
        for i in range(0, len(test_data), BATCH_SIZE):
            batch = test_data[i:i+BATCH_SIZE]
            reconstructed, _, _ = model.reconstructor(batch)
            errors = ((batch - reconstructed) ** 2).mean(dim=(1, 2))
            all_errors.append(errors.cpu())

        all_errors = torch.cat(all_errors)

        # Set threshold at 90th percentile
        threshold = torch.quantile(all_errors, 0.90)

        print(f"✓ Reconstruction threshold: {threshold:.4f}")
        print(f"✓ Error range: [{all_errors.min():.4f}, {all_errors.max():.4f}]")

        # Detect anomalies
        is_anomaly = all_errors > threshold
        scores = all_errors

    # Convert to numpy
    predictions = is_anomaly.cpu().numpy()
    anomaly_scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else scores.cpu().numpy()

    print(f"✓ Detected {predictions.sum()} anomalies ({predictions.sum()/len(predictions)*100:.2f}%)")

    if not np.isnan(anomaly_scores).any():
        print(f"✓ Score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
    else:
        print(f"⚠️  Score range: [nan, nan] - detection failed")

    # Step 6: Evaluate
    print("\n[6/6] Evaluating detection performance...")

    metrics = evaluate_detection(predictions, ground_truth_aligned)
    print_evaluation_results(metrics)

    # Save report
    os.makedirs('validation_outputs', exist_ok=True)
    create_validation_report(
        metrics,
        save_path='validation_outputs/synthetic_validation_report.txt'
    )

    # Additional analysis
    print("\n" + "="*80)
    print("ADDITIONAL ANALYSIS")
    print("="*80)

    # Score distribution for anomalies vs normal
    anomaly_scores_true = anomaly_scores[ground_truth_aligned]
    anomaly_scores_false = anomaly_scores[~ground_truth_aligned]

    if len(anomaly_scores_true) > 0:
        print(f"\nScores for TRUE anomalies:")
        print(f"  Mean: {anomaly_scores_true.mean():.4f}")
        print(f"  Std:  {anomaly_scores_true.std():.4f}")
        print(f"  Min:  {anomaly_scores_true.min():.4f}")
        print(f"  Max:  {anomaly_scores_true.max():.4f}")

    if len(anomaly_scores_false) > 0:
        print(f"\nScores for normal samples:")
        print(f"  Mean: {anomaly_scores_false.mean():.4f}")
        print(f"  Std:  {anomaly_scores_false.std():.4f}")
        print(f"  Min:  {anomaly_scores_false.min():.4f}")
        print(f"  Max:  {anomaly_scores_false.max():.4f}")

    # Recommendation
    print("\n" + "="*80)
    if metrics['f1'] >= 0.7:
        print("✅ Model validation PASSED - Good detection capability")
    elif metrics['f1'] >= 0.5:
        print("⚠️  Model validation MODERATE - Consider tuning threshold or retraining")
    else:
        print("❌ Model validation FAILED - Requires debugging or architecture changes")
    print("="*80)

    return metrics


if __name__ == '__main__':
    try:
        metrics = validate_with_synthetic_anomalies()

        if metrics and metrics['f1'] >= 0.5:
            print("\n✅ Validation complete - Model is ready for real-world testing")
        elif metrics:
            print("\n⚠️  Validation complete - Model needs improvement")
        else:
            print("\n❌ Validation failed - Please check errors above")

    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()

