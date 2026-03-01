"""
Training Pipeline for Self-Supervised Anomaly Detection
Orchestrates the complete training process
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import json


class AnomalyDetectionTrainer:
    """
    Complete training pipeline for self-supervised anomaly detection
    """

    def __init__(
        self,
        model,
        clustering_model,
        energy_detector,
        reconstruction_detector,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=1e-4,
        weight_decay=1e-5
    ):
        """
        Args:
            model: SelfSupervisedTemporalModel
            clustering_model: DensityAwareClustering
            energy_detector: EnergyBasedAnomalyDetector
            reconstruction_detector: ReconstructionBasedDetector
            device: training device
            learning_rate: optimizer learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.clustering_model = clustering_model
        self.energy_detector = energy_detector.to(device)
        self.reconstruction_detector = reconstruction_detector
        self.device = device

        # Optimizer for main model
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Optimizer for energy detector
        self.energy_optimizer = optim.AdamW(
            self.energy_detector.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate schedulers
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )

        self.energy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.energy_optimizer,
            T_max=100,
            eta_min=1e-6
        )

        self.training_history = {
            'total_loss': [],
            'contrastive_loss': [],
            'reconstruction_loss': [],
            'energy_loss': []
        }

    def train_epoch(self, dataloader, epoch):
        """
        Train for one epoch
        Args:
            dataloader: DataLoader with training data
            epoch: current epoch number
        Returns:
            metrics: dictionary with loss values
        """
        self.model.train()
        self.energy_detector.train()

        total_loss = 0
        contrastive_loss_sum = 0
        reconstruction_loss_sum = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

        for batch_idx, (x,) in enumerate(pbar):
            x = x.to(self.device)

            # Forward pass through main model
            self.optimizer.zero_grad()

            loss, losses = self.model(
                x,
                use_contrastive=True,
                use_reconstruction=True
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            contrastive_loss_sum += losses.get('contrastive', 0)
            reconstruction_loss_sum += losses.get('reconstruction', 0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'contrast': f"{losses.get('contrastive', 0):.4f}",
                'recon': f"{losses.get('reconstruction', 0):.4f}"
            })

        # Compute average losses
        n_batches = len(dataloader)
        metrics = {
            'total_loss': total_loss / n_batches,
            'contrastive_loss': contrastive_loss_sum / n_batches,
            'reconstruction_loss': reconstruction_loss_sum / n_batches
        }

        return metrics

    def train_energy_detector(self, dataloader, cluster_labels_dict, n_epochs=10):
        """
        Train energy-based detector after clustering
        Args:
            dataloader: DataLoader
            cluster_labels_dict: mapping from sample index to cluster label
            n_epochs: number of epochs
        """
        print("\n=== Training Energy-Based Detector ===")

        self.model.eval()
        self.energy_detector.train()

        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(n_epochs):
            total_loss = 0
            n_batches = 0

            pbar = tqdm(dataloader, desc=f'Energy Epoch {epoch+1}/{n_epochs}')

            for batch_idx, (x,) in enumerate(pbar):
                x = x.to(self.device)

                # Get embeddings
                with torch.no_grad():
                    embeddings = self.model.get_embeddings(x)

                # Get cluster labels for this batch
                batch_start = batch_idx * dataloader.batch_size
                batch_end = batch_start + x.size(0)

                batch_labels = []
                for i in range(batch_start, batch_end):
                    if i in cluster_labels_dict:
                        batch_labels.append(cluster_labels_dict[i])
                    else:
                        batch_labels.append(0)  # Default cluster

                batch_labels = torch.tensor(batch_labels, device=self.device)

                # Compute energy scores
                self.energy_optimizer.zero_grad()

                energy_scores = self.energy_detector(embeddings, batch_labels)

                # Energy loss: minimize energy for normal samples
                # Penalize high variance within clusters
                energy_loss = energy_scores.mean() + 0.1 * energy_scores.var()

                # Check for NaN
                if torch.isnan(energy_loss):
                    print(f"\n⚠️ NaN detected in energy loss at epoch {epoch+1}, batch {batch_idx}")
                    print("Stopping energy training early")
                    return

                energy_loss.backward()

                # ✅ GRADIENT CLIPPING
                torch.nn.utils.clip_grad_norm_(
                    self.energy_detector.parameters(),
                    max_norm=1.0
                )

                self.energy_optimizer.step()

                total_loss += energy_loss.item()
                n_batches += 1

                pbar.set_postfix({'energy_loss': f"{energy_loss.item():.4f}"})

            # Compute average loss
            avg_loss = total_loss / max(n_batches, 1)

            # Update cluster statistics
            with torch.no_grad():
                for batch_idx, (x,) in enumerate(dataloader):
                    x = x.to(self.device)
                    embeddings = self.model.get_embeddings(x)

                    batch_start = batch_idx * dataloader.batch_size
                    batch_end = batch_start + x.size(0)

                    batch_labels = []
                    for i in range(batch_start, batch_end):
                        batch_labels.append(cluster_labels_dict.get(i, 0))

                    batch_labels = torch.tensor(batch_labels, device=self.device)

                    self.energy_detector.update_cluster_statistics(embeddings, batch_labels)

            print(f"Energy Epoch {epoch+1}: Loss = {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs (no improvement)")
                break

    def fit_clustering(self, dataloader):
        """
        Fit clustering model on learned embeddings
        Args:
            dataloader: DataLoader with training data
        Returns:
            cluster_labels: cluster assignments
        """
        print("\n=== Fitting Clustering Model ===")

        self.model.eval()

        all_embeddings = []

        with torch.no_grad():
            for (x,) in tqdm(dataloader, desc='Extracting embeddings'):
                x = x.to(self.device)
                embeddings = self.model.get_embeddings(x)
                all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

        print(f"Clustering {all_embeddings.shape[0]} embeddings...")

        # Fit clustering
        cluster_labels = self.clustering_model.fit(all_embeddings)

        # Print cluster info
        info = self.clustering_model.get_cluster_info()
        print(f"Total clusters: {info['n_clusters']}")
        print(f"Normal clusters: {info['n_normal_clusters']}")

        return cluster_labels

    def fit_anomaly_detectors(self, dataloader):
        """
        Fit reconstruction-based detector threshold
        Args:
            dataloader: DataLoader with normal training data
        """
        print("\n=== Fitting Anomaly Detectors ===")

        self.model.eval()

        all_data = []

        with torch.no_grad():
            for (x,) in tqdm(dataloader, desc='Collecting data'):
                all_data.append(x)

        all_data = torch.cat(all_data, dim=0).to(self.device)

        # Fit reconstruction detector
        self.reconstruction_detector.fit(all_data)

        print("Anomaly detectors fitted successfully")

    def train(
        self,
        train_dataloader,
        n_epochs=100,
        save_dir='checkpoints',
        save_every=10
    ):
        """
        Complete training pipeline
        Args:
            train_dataloader: DataLoader with training data
            n_epochs: number of training epochs
            save_dir: directory to save checkpoints
            save_every: save checkpoint every N epochs
        """
        os.makedirs(save_dir, exist_ok=True)

        print("="*60)
        print("Starting Self-Supervised Training")
        print("="*60)

        best_loss = float('inf')

        # Phase 1: Self-supervised representation learning
        for epoch in range(1, n_epochs + 1):
            metrics = self.train_epoch(train_dataloader, epoch)

            # Update learning rate
            self.scheduler.step()

            # Log metrics
            for key, value in metrics.items():
                self.training_history[key].append(value)

            print(f"\nEpoch {epoch}/{n_epochs}:")
            print(f"  Total Loss: {metrics['total_loss']:.4f}")
            print(f"  Contrastive: {metrics['contrastive_loss']:.4f}")
            print(f"  Reconstruction: {metrics['reconstruction_loss']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            if epoch % save_every == 0 or metrics['total_loss'] < best_loss:
                if metrics['total_loss'] < best_loss:
                    best_loss = metrics['total_loss']
                    is_best = True
                else:
                    is_best = False

                self.save_checkpoint(
                    epoch,
                    metrics,
                    save_dir,
                    is_best=is_best
                )

        # Phase 2: Fit clustering
        cluster_labels = self.fit_clustering(train_dataloader)

        # Create label mapping
        cluster_labels_dict = {}
        for idx, label in enumerate(cluster_labels):
            cluster_labels_dict[idx] = int(label)

        # Phase 3: Train energy detector
        self.train_energy_detector(train_dataloader, cluster_labels_dict, n_epochs=20)

        # Phase 4: Fit anomaly detectors
        self.fit_anomaly_detectors(train_dataloader)

        # Save final model
        self.save_checkpoint(
            n_epochs,
            metrics,
            save_dir,
            is_best=False,
            final=True,
            cluster_labels=cluster_labels_dict
        )

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)

        return self.training_history

    def save_checkpoint(self, epoch, metrics, save_dir, is_best=False, final=False, cluster_labels=None):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'energy_detector_state_dict': self.energy_detector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'energy_optimizer_state_dict': self.energy_optimizer.state_dict(),
            'metrics': metrics,
            'training_history': self.training_history
        }

        if cluster_labels is not None:
            checkpoint['cluster_labels'] = cluster_labels

        if final:
            path = os.path.join(save_dir, 'final_model.pt')
        elif is_best:
            path = os.path.join(save_dir, 'best_model.pt')
        else:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.energy_detector.load_state_dict(checkpoint['energy_detector_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.energy_optimizer.load_state_dict(checkpoint['energy_optimizer_state_dict'])

        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")

        return checkpoint

