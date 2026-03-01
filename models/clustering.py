"""
Density-Aware Clustering for Normal Market Regime Discovery
Clusters embeddings to identify normal market behaviors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np


class DensityAwareClustering:
    """
    Density-aware clustering in latent embedding space
    Discovers normal market regimes through unsupervised clustering
    """

    def __init__(
        self,
        n_clusters=10,
        method='kmeans',
        density_threshold=0.1,
        min_cluster_size=10
    ):
        """
        Args:
            n_clusters: Number of clusters for KMeans/GMM
            method: 'kmeans', 'gmm', or 'dbscan'
            density_threshold: Minimum density for normal regime
            min_cluster_size: Minimum points per cluster
        """
        self.n_clusters = n_clusters
        self.method = method
        self.density_threshold = density_threshold
        self.min_cluster_size = min_cluster_size

        self.model = None
        self.cluster_centers = None
        self.cluster_densities = None
        self.normal_clusters = None
        self.labels_ = None  # Store labels after fitting

    def fit(self, embeddings):
        """
        Fit clustering model on embeddings
        Args:
            embeddings: (n_samples, embedding_dim) numpy array or tensor
        """
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()

        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            labels = self.model.fit_predict(embeddings)
            self.cluster_centers = self.model.cluster_centers_

        elif self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type='full',
                random_state=42
            )
            labels = self.model.fit_predict(embeddings)
            self.cluster_centers = self.model.means_

        elif self.method == 'dbscan':
            self.model = DBSCAN(
                eps=0.5,
                min_samples=self.min_cluster_size
            )
            labels = self.model.fit_predict(embeddings)

            # Compute cluster centers for DBSCAN
            unique_labels = set(labels)
            unique_labels.discard(-1)  # Remove noise label

            self.cluster_centers = []
            for label in unique_labels:
                mask = labels == label
                center = embeddings[mask].mean(axis=0)
                self.cluster_centers.append(center)

            if len(self.cluster_centers) > 0:
                self.cluster_centers = np.array(self.cluster_centers)
            else:
                self.cluster_centers = None

        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # Compute cluster densities
        self._compute_cluster_densities(embeddings, labels)

        # Identify normal clusters (high density)
        self._identify_normal_clusters()

        # Store labels for later access
        self.labels_ = labels

        return labels

    def _compute_cluster_densities(self, embeddings, labels):
        """
        Compute density for each cluster
        Density = number of samples / average distance to center
        """
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise

        self.cluster_densities = {}

        for label in unique_labels:
            mask = labels == label
            cluster_points = embeddings[mask]

            if len(cluster_points) < self.min_cluster_size:
                self.cluster_densities[label] = 0.0
                continue

            # Compute average distance to cluster center
            if self.cluster_centers is not None:
                center = self.cluster_centers[label]
                distances = np.linalg.norm(cluster_points - center, axis=1)
                avg_distance = distances.mean()

                # Density = size / avg_distance
                density = len(cluster_points) / (avg_distance + 1e-8)
                self.cluster_densities[label] = density
            else:
                self.cluster_densities[label] = len(cluster_points)

    def _identify_normal_clusters(self):
        """
        Identify normal market regime clusters based on density
        High-density clusters are considered normal
        """
        if not self.cluster_densities:
            self.normal_clusters = set()
            return

        # Compute density threshold
        densities = np.array(list(self.cluster_densities.values()))

        if len(densities) == 0:
            self.normal_clusters = set()
            return

        # Use percentile-based threshold
        density_threshold = np.percentile(densities, self.density_threshold * 100)

        # Mark clusters above threshold as normal
        self.normal_clusters = set()
        for label, density in self.cluster_densities.items():
            if density >= density_threshold:
                self.normal_clusters.add(label)

        print(f"Identified {len(self.normal_clusters)} normal clusters out of {len(self.cluster_densities)}")

    def predict(self, embeddings):
        """
        Predict cluster assignments for new embeddings
        Args:
            embeddings: (n_samples, embedding_dim)
        Returns:
            labels: cluster assignments
        """
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()

        if self.method == 'dbscan':
            # For DBSCAN, assign to nearest cluster center
            if self.cluster_centers is None or len(self.cluster_centers) == 0:
                return np.full(len(embeddings), -1)

            distances = np.linalg.norm(
                embeddings[:, np.newaxis, :] - self.cluster_centers[np.newaxis, :, :],
                axis=2
            )
            labels = distances.argmin(axis=1)
        else:
            labels = self.model.predict(embeddings)

        return labels

    def is_normal(self, labels):
        """
        Check if samples belong to normal clusters
        Args:
            labels: cluster assignments
        Returns:
            is_normal: boolean array
        """
        if isinstance(labels, (int, np.integer)):
            return labels in self.normal_clusters

        return np.array([label in self.normal_clusters for label in labels])

    def get_cluster_info(self):
        """
        Get information about identified clusters
        Returns:
            info: dictionary with cluster statistics
        """
        info = {
            'n_clusters': len(self.cluster_densities),
            'n_normal_clusters': len(self.normal_clusters),
            'cluster_densities': self.cluster_densities,
            'normal_clusters': list(self.normal_clusters),
            'cluster_sizes': {}
        }

        return info

    def compute_cluster_anomaly_scores(self, embeddings, labels=None):
        """
        Compute cluster-based anomaly scores combining:
        1. Distance to assigned cluster center (primary signal)
        2. Cluster membership (normal vs abnormal cluster)
        3. Distance relative to cluster spread

        Args:
            embeddings: (n_samples, embedding_dim) numpy array or tensor
            labels: pre-computed cluster labels (optional)

        Returns:
            scores: anomaly scores based on cluster membership and distance
        """
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()

        if labels is None:
            labels = self.predict(embeddings)

        scores = np.zeros(len(embeddings))

        if self.cluster_centers is None or len(self.cluster_centers) == 0:
            return scores

        # Compute distances to assigned cluster centers
        n_samples = len(embeddings)
        distances_to_assigned = np.zeros(n_samples)

        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            if label >= 0 and label < len(self.cluster_centers):
                distances_to_assigned[i] = np.linalg.norm(emb - self.cluster_centers[label])
            else:
                # Distance to nearest center
                dists = np.linalg.norm(emb - self.cluster_centers, axis=1)
                distances_to_assigned[i] = dists.min()

        # Compute per-cluster statistics for adaptive thresholding
        cluster_stats = {}
        for label in range(len(self.cluster_centers)):
            mask = labels == label
            if mask.sum() > 5:
                cluster_dists = distances_to_assigned[mask]
                cluster_stats[label] = {
                    'mean': np.mean(cluster_dists),
                    'std': np.std(cluster_dists),
                    'p90': np.percentile(cluster_dists, 90)
                }

        # Compute max density for normalization
        max_density = max(self.cluster_densities.values()) if self.cluster_densities else 1.0

        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            dist = distances_to_assigned[i]

            # 1. Distance-based score using per-cluster statistics
            if label in cluster_stats:
                stats = cluster_stats[label]
                # Z-score based: how many stds away from mean?
                z_score = (dist - stats['mean']) / (stats['std'] + 1e-8)
                # Points > 2 std away are suspicious
                distance_score = np.clip(z_score / 3.0, 0, 1)
            else:
                # Fallback: use global percentile
                distance_score = np.clip(dist / (np.percentile(distances_to_assigned, 95) + 1e-8), 0, 1)

            # 2. Cluster membership score (abnormal cluster = high score)
            if label not in self.normal_clusters:
                membership_score = 0.8  # Abnormal cluster
            else:
                # Normal cluster - score based on density
                density = self.cluster_densities.get(label, 0)
                # Lower density = higher anomaly score
                membership_score = 0.2 * (1.0 - density / (max_density + 1e-8))

            # 3. Outlier score: is this point an outlier within its cluster?
            if label in cluster_stats:
                # If distance > 90th percentile of cluster, it's an outlier
                if dist > cluster_stats[label]['p90']:
                    outlier_score = np.clip((dist - cluster_stats[label]['p90']) /
                                           (cluster_stats[label]['std'] + 1e-8), 0, 1)
                else:
                    outlier_score = 0.0
            else:
                outlier_score = 0.3

            # Combined score: distance is primary, membership secondary
            scores[i] = 0.50 * distance_score + 0.25 * membership_score + 0.25 * outlier_score

        return scores

    def get_density_dict(self):
        """Return cluster densities for external use"""
        return self.cluster_densities.copy() if self.cluster_densities else {}


class LatentSpaceRegularizer(nn.Module):
    """
    Regularize latent space to improve cluster separation
    Uses center loss and within-cluster compactness
    """

    def __init__(self, embedding_dim, n_clusters, alpha=0.5):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.alpha = alpha

        # Learnable cluster centers
        self.centers = nn.Parameter(torch.randn(n_clusters, embedding_dim))

    def forward(self, embeddings, cluster_labels):
        """
        Compute regularization loss
        Args:
            embeddings: (batch_size, embedding_dim)
            cluster_labels: (batch_size,) cluster assignments
        Returns:
            loss: regularization loss
        """
        batch_size = embeddings.size(0)

        # Get centers for each sample
        centers_batch = self.centers[cluster_labels]  # (batch_size, embedding_dim)

        # Center loss: pull embeddings toward cluster centers
        center_loss = F.mse_loss(embeddings, centers_batch)

        # Separation loss: push cluster centers apart
        if self.n_clusters > 1:
            # Compute pairwise distances between centers
            centers_expanded = self.centers.unsqueeze(1)  # (K, 1, D)
            centers_tiled = self.centers.unsqueeze(0)      # (1, K, D)

            distances = F.pairwise_distance(
                centers_expanded.expand(self.n_clusters, self.n_clusters, self.embedding_dim).reshape(-1, self.embedding_dim),
                centers_tiled.expand(self.n_clusters, self.n_clusters, self.embedding_dim).reshape(-1, self.embedding_dim)
            ).reshape(self.n_clusters, self.n_clusters)

            # Mask diagonal
            mask = 1 - torch.eye(self.n_clusters, device=distances.device)
            distances = distances * mask

            # Separation loss: negative of average pairwise distance
            separation_loss = -distances.sum() / (self.n_clusters * (self.n_clusters - 1))
        else:
            separation_loss = 0

        # Combined loss
        total_loss = center_loss + self.alpha * separation_loss

        return total_loss

    def update_centers(self, embeddings, cluster_labels):
        """
        Update cluster centers with exponential moving average
        Args:
            embeddings: (batch_size, embedding_dim)
            cluster_labels: (batch_size,)
        """
        with torch.no_grad():
            for k in range(self.n_clusters):
                mask = cluster_labels == k
                if mask.sum() > 0:
                    cluster_embeddings = embeddings[mask]
                    new_center = cluster_embeddings.mean(dim=0)

                    # EMA update
                    self.centers[k] = 0.9 * self.centers[k] + 0.1 * new_center

