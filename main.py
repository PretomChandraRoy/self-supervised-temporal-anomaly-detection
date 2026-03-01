"""
Main execution script for anomaly detection framework
Complete end-to-end pipeline from data loading to anomaly detection
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.models.temporal_transformer import SelfSupervisedTemporalModel
from anomaly_detection.models.clustering import DensityAwareClustering, LatentSpaceRegularizer
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
from anomaly_detection.utils.visualization import (
    plot_training_history,
    plot_anomaly_scores,
    plot_detected_anomalies,
    plot_embeddings
)
from anomaly_detection.utils.evaluation import (
    evaluate_anomaly_detection,
    compute_anomaly_statistics
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Self-Supervised Anomaly Detection for Financial Time-Series'
    )

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file with OHLC data')
    parser.add_argument('--window_size', type=int, default=60,
                        help='Length of time-series windows')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data')

    # Model arguments
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Clustering arguments
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters')
    parser.add_argument('--clustering_method', type=str, default='kmeans',
                        choices=['kmeans', 'gmm', 'dbscan'],
                        help='Clustering method')

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                        help='Weight for contrastive loss')
    parser.add_argument('--reconstruction_weight', type=float, default=1.0,
                        help='Weight for reconstruction loss')

    # Anomaly detection arguments
    parser.add_argument('--anomaly_threshold_percentile', type=int, default=95,
                        help='Percentile for anomaly threshold')
    parser.add_argument('--fusion_method', type=str, default='weighted_sum',
                        choices=['weighted_sum', 'max', 'product'],
                        help='Score fusion method')

    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--save_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'full'],
                        help='Execution mode')

    return parser.parse_args()


def main():
    """Main execution function"""

    args = parse_args()

    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*80)
    print("Self-Supervised Anomaly Detection for Financial Time-Series")
    print("="*80)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print()

    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
    viz_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # ===== 1. Load and Preprocess Data =====
    print("Step 1: Loading and preprocessing data...")
    print("-" * 80)

    df = load_forex_data(args.data_path)

    preprocessor = FinancialDataPreprocessor(
        window_size=args.window_size,
        stride=args.stride,
        add_technical_indicators=True,
        scaler_type='robust'
    )

    sequences, feature_names = preprocessor.prepare_data(df, fit_scaler=True)
    n_features = sequences.shape[2]

    print(f"Total sequences: {len(sequences)}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Number of features: {n_features}")
    print()

    # Split train/test
    train_sequences, test_sequences = split_train_test(
        sequences,
        train_ratio=args.train_ratio
    )

    # Create dataloaders
    train_loader = preprocessor.create_dataloader(
        train_sequences,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = preprocessor.create_dataloader(
        test_sequences,
        batch_size=args.batch_size,
        shuffle=False
    )

    # ===== 2. Initialize Models =====
    print("\nStep 2: Initializing models...")
    print("-" * 80)

    # Main self-supervised model
    model = SelfSupervisedTemporalModel(
        n_features=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        contrastive_weight=args.contrastive_weight,
        reconstruction_weight=args.reconstruction_weight
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Clustering model
    clustering_model = DensityAwareClustering(
        n_clusters=args.n_clusters,
        method=args.clustering_method
    )

    # Energy-based detector
    energy_detector = EnergyBasedAnomalyDetector(
        embedding_dim=args.d_model,
        n_clusters=args.n_clusters
    )

    # Reconstruction-based detector
    reconstruction_detector = ReconstructionBasedDetector(
        reconstructor=model.reconstructor,
        threshold_percentile=args.anomaly_threshold_percentile
    )

    # Hybrid detector
    hybrid_detector = HybridAnomalyDetector(
        energy_detector=energy_detector,
        reconstruction_detector=reconstruction_detector,
        fusion_method=args.fusion_method
    )

    print("All models initialized successfully")
    print()

    # ===== 3. Training or Loading Checkpoint =====
    if args.mode in ['train', 'full']:
        print("\nStep 3: Training models...")
        print("-" * 80)

        trainer = AnomalyDetectionTrainer(
            model=model,
            clustering_model=clustering_model,
            energy_detector=energy_detector,
            reconstruction_detector=reconstruction_detector,
            device=device,
            learning_rate=args.learning_rate
        )

        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        # Train
        history = trainer.train(
            train_dataloader=train_loader,
            n_epochs=args.n_epochs,
            save_dir=checkpoint_dir,
            save_every=10
        )

        # Plot training history
        plot_training_history(
            history,
            save_path=os.path.join(viz_dir, 'training_history.png')
        )

        print("\nTraining completed!")
        print()

    elif args.mode == 'test':
        print("\nStep 3: Loading trained model...")
        print("-" * 80)

        if not args.checkpoint:
            # Try to load final model
            checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pt')
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')

            if not os.path.exists(checkpoint_path):
                raise ValueError("No checkpoint found. Please specify --checkpoint or train first.")
        else:
            checkpoint_path = args.checkpoint

        trainer = AnomalyDetectionTrainer(
            model=model,
            clustering_model=clustering_model,
            energy_detector=energy_detector,
            reconstruction_detector=reconstruction_detector,
            device=device
        )

        checkpoint = trainer.load_checkpoint(checkpoint_path)

        # Load cluster labels if available
        if 'cluster_labels' in checkpoint:
            cluster_labels_dict = checkpoint['cluster_labels']
            print(f"Loaded {len(cluster_labels_dict)} cluster labels")

        print()

    # ===== 4. Extract Embeddings =====
    print("\nStep 4: Extracting embeddings...")
    print("-" * 80)

    model.eval()

    train_embeddings = []
    test_embeddings = []

    with torch.no_grad():
        for (x,) in train_loader:
            x = x.to(device)
            emb = model.get_embeddings(x)
            train_embeddings.append(emb.cpu())

        for (x,) in test_loader:
            x = x.to(device)
            emb = model.get_embeddings(x)
            test_embeddings.append(emb.cpu())

    train_embeddings = torch.cat(train_embeddings, dim=0).numpy()
    test_embeddings = torch.cat(test_embeddings, dim=0).numpy()

    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    print()

    # Visualize embeddings
    plot_embeddings(
        train_embeddings,
        save_path=os.path.join(viz_dir, 'embeddings_train.png'),
        title='Training Embeddings (t-SNE)'
    )

    # ===== 5. Anomaly Detection on Test Set =====
    print("\nStep 5: Detecting anomalies...")
    print("-" * 80)

    # Get cluster labels for test set
    test_cluster_labels = clustering_model.predict(test_embeddings)

    # Detect anomalies
    test_data_tensor = torch.FloatTensor(test_sequences).to(device)
    test_embeddings_tensor = torch.FloatTensor(test_embeddings).to(device)
    test_cluster_labels_tensor = torch.LongTensor(test_cluster_labels).to(device)

    anomaly_scores, is_anomaly, details = hybrid_detector.predict(
        test_data_tensor,
        test_embeddings_tensor,
        test_cluster_labels_tensor
    )

    # Convert to numpy
    anomaly_scores = anomaly_scores.cpu().numpy()
    is_anomaly = is_anomaly.cpu().numpy()

    # Compute statistics
    stats = compute_anomaly_statistics(anomaly_scores, is_anomaly)

    print(f"Total samples: {len(anomaly_scores)}")
    print(f"Detected anomalies: {is_anomaly.sum()} ({is_anomaly.mean()*100:.2f}%)")
    print(f"Mean anomaly score: {stats['mean_score']:.4f}")
    print(f"Std anomaly score: {stats['std_score']:.4f}")
    print(f"Max anomaly score: {stats['max_score']:.4f}")
    print()

    # ===== 6. Visualization =====
    print("\nStep 6: Generating visualizations...")
    print("-" * 80)

    # Plot anomaly scores
    plot_anomaly_scores(
        anomaly_scores,
        is_anomaly,
        save_path=os.path.join(viz_dir, 'anomaly_scores.png')
    )

    # Plot detected anomalies on price data
    if len(df) >= len(sequences):
        # Get corresponding price data for test sequences
        test_start_idx = len(train_sequences)
        test_prices = df['close'].iloc[test_start_idx:test_start_idx + len(test_sequences)].values

        plot_detected_anomalies(
            test_prices,
            is_anomaly,
            anomaly_scores,
            save_path=os.path.join(viz_dir, 'anomalies_on_prices.png')
        )

    print("Visualizations saved")
    print()

    # ===== 7. Save Results =====
    print("\nStep 7: Saving results...")
    print("-" * 80)

    results_df = pd.DataFrame({
        'anomaly_score': anomaly_scores,
        'is_anomaly': is_anomaly,
        'cluster_label': test_cluster_labels
    })

    results_path = os.path.join(args.save_dir, 'anomaly_results.csv')
    results_df.to_csv(results_path, index=False)

    print(f"Results saved to: {results_path}")
    print()

    print("="*80)
    print("Anomaly Detection Complete!")
    print("="*80)

    return results_df


if __name__ == '__main__':
    main()

