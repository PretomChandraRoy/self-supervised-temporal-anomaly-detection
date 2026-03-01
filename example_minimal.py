    """
Minimal example without visualization dependencies
This version works even if matplotlib has compatibility issues
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


def example_usage():
    """
    Simple example of using the framework (no visualization)
    """

    print("="*80)
    print("Anomaly Detection Framework - Minimal Example")
    print("="*80)

    # Configuration
    DATA_PATH = '../forexPredictor/H4_EURUSD_2015.csv'
    WINDOW_SIZE = 60
    BATCH_SIZE = 32
    N_EPOCHS = 5  # Very small for quick testing
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 1: Load and preprocess data
    print("\n[1/6] Loading data...")
    print(f"Data path: {DATA_PATH}")

    try:
        df = load_forex_data(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: File not found: {DATA_PATH}")
        print("Please update DATA_PATH to point to your CSV file.")
        return None

    preprocessor = FinancialDataPreprocessor(
        window_size=WINDOW_SIZE,
        stride=1,
        add_technical_indicators=True
    )

    sequences, feature_names = preprocessor.prepare_data(df, fit_scaler=True)
    n_features = sequences.shape[2]

    print(f"✓ Loaded {len(sequences)} sequences with {n_features} features")

    # Step 2: Split data
    print("\n[2/6] Splitting train/test...")
    train_seq, test_seq = split_train_test(sequences, train_ratio=0.8)

    train_loader = preprocessor.create_dataloader(train_seq, batch_size=BATCH_SIZE)
    test_loader = preprocessor.create_dataloader(test_seq, batch_size=BATCH_SIZE, shuffle=False)

    print(f"✓ Train: {len(train_seq)}, Test: {len(test_seq)}")

    # Step 3: Initialize models
    print("\n[3/6] Initializing models...")

    model = SelfSupervisedTemporalModel(
        n_features=n_features,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dropout=0.1
    )

    clustering = DensityAwareClustering(n_clusters=10, method='kmeans')
    energy_detector = EnergyBasedAnomalyDetector(embedding_dim=128, n_clusters=10)
    recon_detector = ReconstructionBasedDetector(model.reconstructor)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Device: {DEVICE}")

    # Step 4: Train
    print("\n[4/6] Training...")
    print(f"Training for {N_EPOCHS} epochs (this will take a few minutes)...")

    trainer = AnomalyDetectionTrainer(
        model=model,
        clustering_model=clustering,
        energy_detector=energy_detector,
        reconstruction_detector=recon_detector,
        device=DEVICE
    )

    history = trainer.train(
        train_dataloader=train_loader,
        n_epochs=N_EPOCHS,
        save_dir='example_outputs/checkpoints'
    )

    print("✓ Training complete")

    # Step 5: Extract embeddings
    print("\n[5/6] Extracting embeddings...")

    model.eval()
    test_embeddings = []

    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(DEVICE)
            emb = model.get_embeddings(x)
            test_embeddings.append(emb.cpu())

    test_embeddings = torch.cat(test_embeddings, dim=0)
    print(f"✓ Extracted {len(test_embeddings)} embeddings")

    # Step 6: Detect anomalies
    print("\n[6/6] Detecting anomalies...")

    # Get cluster labels
    cluster_labels = clustering.predict(test_embeddings.numpy())

    # Create hybrid detector
    hybrid = HybridAnomalyDetector(
        energy_detector=energy_detector,
        reconstruction_detector=recon_detector
    )

    # Detect
    test_data = torch.FloatTensor(test_seq).to(DEVICE)
    test_emb = test_embeddings.to(DEVICE)
    cluster_tensor = torch.LongTensor(cluster_labels).to(DEVICE)

    scores, is_anomaly, details = hybrid.predict(test_data, test_emb, cluster_tensor)

    n_anomalies = is_anomaly.sum().item()
    anomaly_rate = (n_anomalies / len(is_anomaly)) * 100

    print(f"✓ Detected {n_anomalies} anomalies ({anomaly_rate:.2f}%)")

    # Print statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Total test samples: {len(is_anomaly)}")
    print(f"Detected anomalies: {n_anomalies} ({anomaly_rate:.2f}%)")
    print(f"Mean anomaly score: {scores.cpu().numpy().mean():.4f}")
    print(f"Max anomaly score:  {scores.cpu().numpy().max():.4f}")
    print(f"Min anomaly score:  {scores.cpu().numpy().min():.4f}")

    if n_anomalies > 0:
        anomaly_scores = scores[is_anomaly].cpu().numpy()
        print(f"\nAnomalous samples:")
        print(f"  Mean score: {anomaly_scores.mean():.4f}")
        print(f"  Max score:  {anomaly_scores.max():.4f}")
        print(f"  Min score:  {anomaly_scores.min():.4f}")

    # Save results
    print("\nSaving results...")
    os.makedirs('example_outputs', exist_ok=True)

    results_file = 'example_outputs/anomaly_results.txt'
    with open(results_file, 'w') as f:
        f.write("Anomaly Detection Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total samples: {len(is_anomaly)}\n")
        f.write(f"Anomalies: {n_anomalies} ({anomaly_rate:.2f}%)\n")
        f.write(f"Mean score: {scores.cpu().numpy().mean():.4f}\n")
        f.write(f"\nAnomaly indices:\n")
        anomaly_indices = np.where(is_anomaly.cpu().numpy())[0]
        for idx in anomaly_indices[:20]:  # First 20
            f.write(f"  Index {idx}: Score = {scores[idx].item():.4f}\n")
        if len(anomaly_indices) > 20:
            f.write(f"  ... and {len(anomaly_indices)-20} more\n")

    print(f"✓ Results saved to: {results_file}")

    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check 'example_outputs/anomaly_results.txt' for detailed results")
    print("  2. Check 'example_outputs/checkpoints/' for saved models")
    print("  3. Run main.py for full pipeline with visualization")
    print("  4. Adjust parameters and re-run with your own data")

    return {
        'scores': scores.cpu().numpy(),
        'is_anomaly': is_anomaly.cpu().numpy(),
        'embeddings': test_embeddings.numpy(),
        'cluster_labels': cluster_labels
    }


if __name__ == '__main__':
    try:
        results = example_usage()
        if results is not None:
            print("\n✅ Example completed without errors!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

