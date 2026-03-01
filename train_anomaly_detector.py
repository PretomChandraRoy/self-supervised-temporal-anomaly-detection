"""
Example script demonstrating how to use the anomaly detection framework
"""

import torch
import sys
import os

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
    Simple example of using the framework
    """

    print("="*80)
    print("Anomaly Detection Framework - Example Usage")
    print("="*80)

    # Configuration
    DATA_PATH = '../forexPredictor/H4_EURUSD_2015.csv'
    WINDOW_SIZE = 60
    BATCH_SIZE = 32
    N_EPOCHS = 10  # Small for example
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 1: Load and preprocess data
    print("\n[1/6] Loading data...")
    df = load_forex_data(DATA_PATH)

    preprocessor = FinancialDataPreprocessor(
        window_size=WINDOW_SIZE,
        stride=1,
        add_technical_indicators=True,
        scaler_type='robust',
        clip_outliers=True
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
        dropout=0.1,
        reconstruction_weight=0.1,  # Lowered from 1.0 to prevent loss explosion
        contrastive_weight=1.0
    )

    clustering = DensityAwareClustering(n_clusters=10, method='kmeans')
    energy_detector = EnergyBasedAnomalyDetector(embedding_dim=128, n_clusters=10)
    recon_detector = ReconstructionBasedDetector(model.reconstructor)

    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 4: Train
    print("\n[4/6] Training...")

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

    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)

    return {
        'scores': scores.detach().cpu().numpy(),
        'is_anomaly': is_anomaly.cpu().numpy(),
        'embeddings': test_embeddings.numpy(),
        'cluster_labels': cluster_labels
    }


if __name__ == '__main__':
    results = example_usage()

