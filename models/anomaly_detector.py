"""
Energy-Based and Reconstruction-Based Anomaly Detection
Identifies rare and structurally inconsistent temporal patterns
"""

import torch
import torch.nn as nn
import numpy as np


class EnergyBasedAnomalyDetector(nn.Module):
    """
    Energy-based anomaly scoring with cluster conditioning
    High energy = anomaly (far from normal cluster regions)
    """

    def __init__(self, embedding_dim, n_clusters, temperature=1.0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.temperature = temperature

        # Energy function (learned)
        self.energy_net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Cluster-specific energy normalization
        self.cluster_energy_means = nn.Parameter(torch.zeros(n_clusters))
        self.cluster_energy_stds = nn.Parameter(torch.ones(n_clusters))

    def compute_energy(self, embeddings):
        """
        Compute energy score for embeddings
        Args:
            embeddings: (batch_size, embedding_dim)
        Returns:
            energy: (batch_size,) - lower is more normal
        """
        energy = self.energy_net(embeddings).squeeze(-1)
        return energy

    def forward(self, embeddings, cluster_labels=None):
        """
        Compute anomaly score based on energy
        Args:
            embeddings: (batch_size, embedding_dim)
            cluster_labels: (batch_size,) optional cluster assignments
        Returns:
            anomaly_scores: (batch_size,) - higher = more anomalous
        """
        # Compute raw energy
        energy = self.compute_energy(embeddings)

        # Normalize by cluster statistics if available
        if cluster_labels is not None:
            normalized_energy = torch.zeros_like(energy)

            for k in range(self.n_clusters):
                mask = cluster_labels == k
                if mask.sum() > 0:
                    # Normalize by cluster-specific statistics
                    normalized_energy[mask] = (
                        (energy[mask] - self.cluster_energy_means[k]) /
                        (self.cluster_energy_stds[k] + 1e-8)
                    )

            return normalized_energy
        else:
            return energy

    def update_cluster_statistics(self, embeddings, cluster_labels):
        """
        Update cluster-specific energy statistics
        Args:
            embeddings: (batch_size, embedding_dim)
            cluster_labels: (batch_size,)
        """
        with torch.no_grad():
            energy = self.compute_energy(embeddings)

            for k in range(self.n_clusters):
                mask = cluster_labels == k
                if mask.sum() > 0:
                    cluster_energies = energy[mask]

                    # Update with EMA
                    self.cluster_energy_means[k] = (
                        0.9 * self.cluster_energy_means[k] +
                        0.1 * cluster_energies.mean()
                    )
                    self.cluster_energy_stds[k] = (
                        0.9 * self.cluster_energy_stds[k] +
                        0.1 * cluster_energies.std()
                    )


class ReconstructionBasedDetector:
    """
    Reconstruction-based anomaly detection
    Measures deviation between input and reconstruction
    """

    def __init__(
        self,
        reconstructor,
        threshold_percentile=95,
        use_mahalanobis=True,
        feature_weights=None
    ):
        """
        Args:
            reconstructor: MaskedTimeSeriesReconstructor model
            threshold_percentile: Percentile for anomaly threshold
            use_mahalanobis: Use Mahalanobis distance instead of Euclidean
            feature_weights: Optional tensor of per-feature importance weights
        """
        self.reconstructor = reconstructor
        self.threshold_percentile = threshold_percentile
        self.use_mahalanobis = use_mahalanobis
        self.feature_weights = feature_weights

        self.threshold = None
        self.mean = None
        self.cov_inv = None

    def compute_reconstruction_error(self, x, reconstructed):
        """
        Compute reconstruction error focusing on the single worst timestep.
        Uses max-over-features per timestep so that a spike in even one
        feature (e.g. close price) produces a high per-timestep error.
        Then takes max over timesteps to capture point anomalies.
        Args:
            x: (batch_size, seq_len, n_features) original
            reconstructed: (batch_size, seq_len, n_features) reconstructed
        Returns:
            errors: (batch_size,) per-sample error
        """
        # Point-wise squared error
        squared_error = (x - reconstructed) ** 2

        # Apply feature weights if available
        if hasattr(self, 'feature_weights') and self.feature_weights is not None:
            weights = self.feature_weights.to(squared_error.device)
            squared_error = squared_error * weights.unsqueeze(0).unsqueeze(0)

        # Per-timestep error: use MAX over features (not mean)
        # so a spike in any single feature is not diluted
        per_timestep_max = squared_error.max(dim=2)[0]   # (batch, seq_len)
        per_timestep_mean = squared_error.mean(dim=2)      # (batch, seq_len)

        # Combine: emphasize the feature-max but include mean as context
        per_timestep_error = 0.7 * per_timestep_max + 0.3 * per_timestep_mean

        # Over time: max captures the anomalous timestep
        max_error = per_timestep_error.max(dim=1)[0]       # (batch,)

        # Top-K: more robust than pure max
        k = min(3, per_timestep_error.shape[1])
        topk_error = per_timestep_error.topk(k, dim=1)[0].mean(dim=1)  # (batch,)

        # Combined: heavily weight the peak
        errors = 0.6 * max_error + 0.4 * topk_error

        return errors

    def compute_mahalanobis_distance(self, x, reconstructed):
        """
        Compute Mahalanobis distance for reconstruction error
        Args:
            x: (batch_size, seq_len, n_features)
            reconstructed: (batch_size, seq_len, n_features)
        Returns:
            distances: (batch_size,)
        """
        # Flatten to (batch_size, seq_len * n_features)
        x_flat = x.reshape(x.size(0), -1)
        reconstructed_flat = reconstructed.reshape(reconstructed.size(0), -1)

        # Compute residual
        residual = x_flat - reconstructed_flat

        if self.cov_inv is None:
            # Use simple Euclidean if covariance not fitted
            return (residual ** 2).mean(dim=1)

        # Mahalanobis distance: sqrt(r^T * Σ^-1 * r)
        if torch.is_tensor(residual):
            residual_np = residual.cpu().numpy()
        else:
            residual_np = residual

        distances = []
        for r in residual_np:
            d = np.sqrt(r @ self.cov_inv @ r)
            distances.append(d)

        return torch.tensor(distances, device=x.device)

    def fit(self, x):
        """
        Fit detector on normal data to establish threshold.
        Uses the same bottleneck approach as predict().
        Args:
            x: (n_samples, seq_len, n_features) normal training data
        """
        self.reconstructor.eval()

        with torch.no_grad():
            # Encode and bottleneck (same as predict)
            encoder_output = self.reconstructor.encoder(x)
            bottleneck = encoder_output.mean(dim=1, keepdim=True)
            bottleneck_expanded = bottleneck.expand_as(encoder_output)
            reconstructed = self.reconstructor.reconstruction_head(bottleneck_expanded)

            if self.use_mahalanobis:
                # Fit covariance on residuals
                x_flat = x.reshape(x.size(0), -1).cpu().numpy()
                rec_flat = reconstructed.reshape(reconstructed.size(0), -1).cpu().numpy()
                residuals = x_flat - rec_flat

                self.mean = residuals.mean(axis=0)
                cov = np.cov(residuals.T)

                # Add regularization to ensure invertibility
                cov += np.eye(cov.shape[0]) * 1e-6

                try:
                    self.cov_inv = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    print("Warning: Covariance matrix not invertible, using Euclidean distance")
                    self.use_mahalanobis = False
                    self.cov_inv = None

                errors = self.compute_mahalanobis_distance(x, reconstructed)
            else:
                errors = self.compute_reconstruction_error(x, reconstructed)

            # Set threshold at percentile
            if torch.is_tensor(errors):
                errors_np = errors.cpu().numpy()
            else:
                errors_np = errors

            self.threshold = np.percentile(errors_np, self.threshold_percentile)

            print(f"Reconstruction threshold set to: {self.threshold:.6f}")

    def predict(self, x):
        """
        Predict anomaly scores based on reconstruction error.
        Uses a BOTTLENECK approach: the encoder output is mean-pooled
        into a single vector and then expanded back to seq_len, forcing
        information loss.  This means anomalous timesteps cannot be
        trivially copied through, producing higher reconstruction error.
        Args:
            x: (batch_size, seq_len, n_features)
        Returns:
            scores: (batch_size,) anomaly scores
            is_anomaly: (batch_size,) boolean mask
        """
        self.reconstructor.eval()

        with torch.no_grad():
            # Encode full sequence
            encoder_output = self.reconstructor.encoder(x)  # (B, L, d_model)

            # Bottleneck: compress to single vector, then expand back
            bottleneck = encoder_output.mean(dim=1, keepdim=True)  # (B, 1, d_model)
            bottleneck_expanded = bottleneck.expand_as(encoder_output)  # (B, L, d_model)

            # Reconstruct from bottleneck (anomalous timesteps lose their signal)
            reconstructed = self.reconstructor.reconstruction_head(bottleneck_expanded)

            if self.use_mahalanobis and self.cov_inv is not None:
                scores = self.compute_mahalanobis_distance(x, reconstructed)
            else:
                scores = self.compute_reconstruction_error(x, reconstructed)

            # Determine anomalies
            if self.threshold is not None:
                is_anomaly = scores > self.threshold
            else:
                is_anomaly = torch.zeros_like(scores, dtype=torch.bool)

        return scores, is_anomaly


class HybridAnomalyDetector:
    """
    Combines energy-based and reconstruction-based detection
    Provides comprehensive anomaly scoring
    """

    def __init__(
        self,
        energy_detector,
        reconstruction_detector,
        energy_weight=0.5,
        reconstruction_weight=0.5,
        fusion_method='weighted_sum'
    ):
        """
        Args:
            energy_detector: EnergyBasedAnomalyDetector
            reconstruction_detector: ReconstructionBasedDetector
            energy_weight: Weight for energy score
            reconstruction_weight: Weight for reconstruction score
            fusion_method: 'weighted_sum', 'max', or 'product'
        """
        self.energy_detector = energy_detector
        self.reconstruction_detector = reconstruction_detector
        self.energy_weight = energy_weight
        self.reconstruction_weight = reconstruction_weight
        self.fusion_method = fusion_method

        self.threshold = None

    def normalize_scores(self, scores):
        """
        Normalize scores to [0, 1] range
        """
        if torch.is_tensor(scores):
            # Detach from computation graph before converting to numpy
            scores_detached = scores.detach()
            min_val = scores_detached.min()
            max_val = scores_detached.max()

            if max_val - min_val < 1e-8:
                return scores_detached * 0

            normalized = (scores_detached - min_val) / (max_val - min_val)
            return normalized
        else:
            min_val = scores.min()
            max_val = scores.max()

            if max_val - min_val < 1e-8:
                return scores * 0

            normalized = (scores - min_val) / (max_val - min_val)
            return normalized

    def predict(self, x, embeddings, cluster_labels=None):
        """
        Compute hybrid anomaly scores
        Args:
            x: (batch_size, seq_len, n_features) input sequences
            embeddings: (batch_size, embedding_dim) learned representations
            cluster_labels: (batch_size,) optional cluster assignments
        Returns:
            scores: (batch_size,) combined anomaly scores
            is_anomaly: (batch_size,) boolean mask
            details: dict with individual scores
        """
        # Get energy-based scores
        energy_scores = self.energy_detector(embeddings, cluster_labels)
        energy_scores = self.normalize_scores(energy_scores)

        # Get reconstruction-based scores
        reconstruction_scores, _ = self.reconstruction_detector.predict(x)
        reconstruction_scores = self.normalize_scores(reconstruction_scores)

        # Combine scores
        if self.fusion_method == 'weighted_sum':
            combined_scores = (
                self.energy_weight * energy_scores +
                self.reconstruction_weight * reconstruction_scores
            )
        elif self.fusion_method == 'max':
            combined_scores = torch.maximum(energy_scores, reconstruction_scores)
        elif self.fusion_method == 'product':
            combined_scores = energy_scores * reconstruction_scores
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Determine anomalies
        if self.threshold is not None:
            is_anomaly = combined_scores > self.threshold
        else:
            # Use adaptive threshold (mean + 2*std)
            mean_score = combined_scores.mean()
            std_score = combined_scores.std()
            adaptive_threshold = mean_score + 2 * std_score
            is_anomaly = combined_scores > adaptive_threshold

        details = {
            'energy_scores': energy_scores,
            'reconstruction_scores': reconstruction_scores,
            'combined_scores': combined_scores
        }

        return combined_scores, is_anomaly, details

    def fit_threshold(self, x, embeddings, cluster_labels=None, percentile=95):
        """
        Fit threshold on normal data
        Args:
            x: normal training data
            embeddings: corresponding embeddings
            cluster_labels: cluster assignments
            percentile: threshold percentile
        """
        with torch.no_grad():
            scores, _, _ = self.predict(x, embeddings, cluster_labels)

            if torch.is_tensor(scores):
                scores_np = scores.cpu().numpy()
            else:
                scores_np = scores

            self.threshold = np.percentile(scores_np, percentile)

            print(f"Hybrid anomaly threshold set to: {self.threshold:.6f}")

