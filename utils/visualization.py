"""
Visualization utilities for anomaly detection
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_training_history(history, save_path=None):
    """
    Plot training loss curves
    Args:
        history: dictionary with loss values
        save_path: optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')

    # Total loss
    if 'total_loss' in history and len(history['total_loss']) > 0:
        axes[0, 0].plot(history['total_loss'], linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

    # Contrastive loss
    if 'contrastive_loss' in history and len(history['contrastive_loss']) > 0:
        axes[0, 1].plot(history['contrastive_loss'], color='orange', linewidth=2)
        axes[0, 1].set_title('Contrastive Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)

    # Reconstruction loss
    if 'reconstruction_loss' in history and len(history['reconstruction_loss']) > 0:
        axes[1, 0].plot(history['reconstruction_loss'], color='green', linewidth=2)
        axes[1, 0].set_title('Reconstruction Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)

    # Energy loss
    if 'energy_loss' in history and len(history['energy_loss']) > 0:
        axes[1, 1].plot(history['energy_loss'], color='red', linewidth=2)
        axes[1, 1].set_title('Energy Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.close()


def plot_embeddings(embeddings, labels=None, save_path=None, title='Embeddings Visualization', method='tsne'):
    """
    Visualize high-dimensional embeddings in 2D
    Args:
        embeddings: (n_samples, embedding_dim) array
        labels: optional cluster labels for coloring
        save_path: optional path to save figure
        title: plot title
        method: 'tsne' or 'pca'
    """
    # Reduce to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=42)

    embeddings_2d = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))

    if labels is not None:
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter, label='Cluster')
    else:
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6,
            s=20
        )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embeddings plot saved to {save_path}")

    plt.close()


def plot_anomaly_scores(scores, is_anomaly, save_path=None):
    """
    Plot anomaly score distribution and detected anomalies
    Args:
        scores: anomaly scores
        is_anomaly: boolean mask of anomalies
        save_path: optional path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Anomaly Detection Results', fontsize=16, fontweight='bold')

    # Time series of scores
    axes[0].plot(scores, linewidth=1, alpha=0.7, label='Anomaly Score')
    anomaly_indices = np.where(is_anomaly)[0]
    axes[0].scatter(
        anomaly_indices,
        scores[anomaly_indices],
        color='red',
        s=50,
        alpha=0.8,
        label='Detected Anomaly',
        zorder=5
    )
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Anomaly Score')
    axes[0].set_title('Anomaly Scores Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(scores[~is_anomaly], bins=50, alpha=0.7, label='Normal', color='blue')
    axes[1].hist(scores[is_anomaly], bins=30, alpha=0.7, label='Anomaly', color='red')
    axes[1].axvline(
        scores[is_anomaly].min() if is_anomaly.sum() > 0 else 0,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Threshold'
    )
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Score Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Anomaly scores plot saved to {save_path}")

    plt.close()


def plot_detected_anomalies(prices, is_anomaly, scores, save_path=None, max_points=5000):
    """
    Plot detected anomalies on price chart
    Args:
        prices: price time series
        is_anomaly: boolean mask
        scores: anomaly scores
        save_path: optional path to save
        max_points: maximum points to plot
    """
    # Subsample if too many points
    if len(prices) > max_points:
        step = len(prices) // max_points
        prices = prices[::step]
        is_anomaly = is_anomaly[::step]
        scores = scores[::step]

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Detected Anomalies on Price Chart', fontsize=16, fontweight='bold')

    # Price chart with anomalies
    axes[0].plot(prices, linewidth=1, color='black', alpha=0.7, label='Price')

    anomaly_indices = np.where(is_anomaly)[0]
    if len(anomaly_indices) > 0:
        axes[0].scatter(
            anomaly_indices,
            prices[anomaly_indices],
            color='red',
            s=100,
            alpha=0.8,
            marker='x',
            linewidths=2,
            label='Anomaly',
            zorder=5
        )

    axes[0].set_ylabel('Price')
    axes[0].set_title('Price with Detected Anomalies')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Anomaly scores
    axes[1].fill_between(
        range(len(scores)),
        scores,
        alpha=0.5,
        color='blue',
        label='Anomaly Score'
    )

    if len(anomaly_indices) > 0:
        axes[1].scatter(
            anomaly_indices,
            scores[anomaly_indices],
            color='red',
            s=50,
            alpha=0.8,
            label='Detected',
            zorder=5
        )

    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Anomaly Score')
    axes[1].set_title('Anomaly Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Anomalies on prices plot saved to {save_path}")

    plt.close()


def plot_cluster_analysis(embeddings, cluster_labels, cluster_densities, save_path=None):
    """
    Visualize clustering results
    Args:
        embeddings: 2D reduced embeddings
        cluster_labels: cluster assignments
        cluster_densities: density per cluster
        save_path: optional save path
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Cluster Analysis', fontsize=16, fontweight='bold')

    # Cluster visualization
    scatter = axes[0].scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=cluster_labels,
        cmap='tab10',
        alpha=0.6,
        s=20
    )
    axes[0].set_title('Cluster Assignments')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    plt.colorbar(scatter, ax=axes[0], label='Cluster')

    # Cluster density distribution
    clusters = sorted(cluster_densities.keys())
    densities = [cluster_densities[c] for c in clusters]

    axes[1].bar(clusters, densities, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Cluster Densities')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster analysis plot saved to {save_path}")

    plt.close()

