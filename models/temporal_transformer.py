"""
Transformer-based Temporal Encoder for Financial Time-Series
Captures long-range dependencies and temporal patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder for financial time-series
    Learns context-aware embeddings capturing market dynamics
    """

    def __init__(
        self,
        n_features,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=512
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, n_features)
            mask: Optional attention mask
        Returns:
            encoded: (batch_size, seq_len, d_model)
        """
        # Project input to model dimension
        x = self.input_projection(x)  # (B, L, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer
        encoded = self.transformer_encoder(x, mask=mask)

        # Normalize
        encoded = self.layer_norm(encoded)

        return encoded

    def get_sequence_embedding(self, x, mask=None):
        """
        Get global sequence embedding using mean pooling
        Args:
            x: (batch_size, seq_len, n_features)
        Returns:
            embedding: (batch_size, d_model)
        """
        encoded = self.forward(x, mask)  # (B, L, d_model)

        # Mean pooling over time dimension
        embedding = encoded.mean(dim=1)  # (B, d_model)

        return embedding


class MaskedTimeSeriesReconstructor(nn.Module):
    """
    Masked autoencoder for time-series reconstruction
    Learns to predict masked temporal segments
    """

    def __init__(
        self,
        encoder,
        mask_ratio=0.15,
        reconstruction_weight=1.0
    ):
        super().__init__()

        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.reconstruction_weight = reconstruction_weight

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.dim_feedforward if hasattr(encoder, 'dim_feedforward') else 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(encoder.dim_feedforward if hasattr(encoder, 'dim_feedforward') else 512, encoder.n_features)
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, encoder.d_model))

    def random_masking(self, x):
        """
        Randomly mask time steps
        Args:
            x: (batch_size, seq_len, n_features)
        Returns:
            masked_x: (batch_size, seq_len, n_features)
            mask: (batch_size, seq_len) - 1 for masked, 0 for unmasked
        """
        B, L, D = x.shape

        # Random mask
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(B, L, device=x.device)

        # Sort noise to get masking indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Binary mask: 0 is keep, 1 is remove
        mask = torch.ones(B, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def forward(self, x):
        """
        Forward pass with masking and reconstruction
        Args:
            x: (batch_size, seq_len, n_features)
        Returns:
            reconstructed: (batch_size, seq_len, n_features)
            mask: (batch_size, seq_len)
            loss: reconstruction loss
        """
        # Create mask
        mask = self.random_masking(x)  # (B, L)

        # Encode (encoder handles the actual input)
        encoded = self.encoder(x)  # (B, L, d_model)

        # Replace masked positions with mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
        encoded_masked = encoded * (1 - mask_expanded) + self.mask_token * mask_expanded

        # Reconstruct
        reconstructed = self.reconstruction_head(encoded_masked)  # (B, L, n_features)

        # Compute reconstruction loss only on masked positions
        loss = F.mse_loss(
            reconstructed[mask.bool()],
            x[mask.bool()],
            reduction='mean'
        ) * self.reconstruction_weight

        return reconstructed, mask, loss


class TemporalContrastiveLearning(nn.Module):
    """
    Temporal contrastive learning for self-supervised representation
    Creates positive pairs through temporal augmentation
    """

    def __init__(
        self,
        encoder,
        projection_dim=128,
        temperature=0.07
    ):
        super().__init__()

        self.encoder = encoder
        self.temperature = temperature

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.d_model),
            nn.ReLU(),
            nn.Linear(encoder.d_model, projection_dim)
        )

    def create_temporal_augmentations(self, x):
        """
        Create augmented views through temporal transformations
        Args:
            x: (batch_size, seq_len, n_features)
        Returns:
            aug1, aug2: Two augmented views
        """
        # Augmentation 1: Random time masking
        mask1 = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > 0.1
        aug1 = x * mask1

        # Augmentation 2: Add small Gaussian noise
        noise = torch.randn_like(x) * 0.01
        aug2 = x + noise

        return aug1, aug2

    def nt_xent_loss(self, z1, z2):
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
        Args:
            z1, z2: (batch_size, projection_dim)
        Returns:
            loss: contrastive loss
        """
        batch_size = z1.shape[0]

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Create labels (diagonal pairs are positive)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels])

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, n_features)
        Returns:
            loss: contrastive loss
        """
        # Create augmented views
        aug1, aug2 = self.create_temporal_augmentations(x)

        # Encode both views
        h1 = self.encoder.get_sequence_embedding(aug1)  # (B, d_model)
        h2 = self.encoder.get_sequence_embedding(aug2)  # (B, d_model)

        # Project
        z1 = self.projection_head(h1)  # (B, projection_dim)
        z2 = self.projection_head(h2)  # (B, projection_dim)

        # Compute contrastive loss
        loss = self.nt_xent_loss(z1, z2)

        return loss


class SelfSupervisedTemporalModel(nn.Module):
    """
    Combined self-supervised learning framework
    Integrates contrastive learning and masked reconstruction
    """

    def __init__(
        self,
        n_features,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        mask_ratio=0.15,
        temperature=0.07,
        contrastive_weight=1.0,
        reconstruction_weight=1.0
    ):
        super().__init__()

        # Shared encoder
        self.encoder = TemporalTransformerEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Contrastive learning module
        self.contrastive = TemporalContrastiveLearning(
            encoder=self.encoder,
            temperature=temperature
        )

        # Masked reconstruction module
        self.reconstructor = MaskedTimeSeriesReconstructor(
            encoder=self.encoder,
            mask_ratio=mask_ratio,
            reconstruction_weight=reconstruction_weight
        )

        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight

    def forward(self, x, use_contrastive=True, use_reconstruction=True):
        """
        Combined training with both objectives
        Args:
            x: (batch_size, seq_len, n_features)
            use_contrastive: whether to compute contrastive loss
            use_reconstruction: whether to compute reconstruction loss
        Returns:
            total_loss: combined loss
            losses_dict: individual losses
        """
        losses = {}
        total_loss = 0

        if use_contrastive:
            contrastive_loss = self.contrastive(x)
            losses['contrastive'] = contrastive_loss.item()
            total_loss += self.contrastive_weight * contrastive_loss

        if use_reconstruction:
            _, _, reconstruction_loss = self.reconstructor(x)
            losses['reconstruction'] = reconstruction_loss.item()
            total_loss += reconstruction_loss

        losses['total'] = total_loss.item()

        return total_loss, losses

    def get_embeddings(self, x):
        """
        Extract learned representations
        Args:
            x: (batch_size, seq_len, n_features)
        Returns:
            embeddings: (batch_size, d_model)
        """
        return self.encoder.get_sequence_embedding(x)

