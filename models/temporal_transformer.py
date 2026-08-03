"""
Transformer-based Temporal Encoder for Financial Time-Series
Captures long-range dependencies and temporal patterns

Extended with:
  - Multi-Scale Temporal Context (MSTC) front-end
  - Hierarchical Regime Guidance (HRG) post-encoder block
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


# =========================================================================
# Squeeze-and-Excitation (1-D, shared helper)
# =========================================================================

class SqueezeExcitation1d(nn.Module):
    """Channel-wise squeeze-and-excitation gate for 1-D feature maps."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, C, T)  ->  (B, C, T), channel-reweighted."""
        s = x.mean(dim=2)           # (B, C) -- global average pool over time
        s = self.fc(s).unsqueeze(2) # (B, C, 1)
        return x * s


# =========================================================================
# MSTC -- Multi-Scale Temporal Context
# =========================================================================

class _DepthwiseConv1dContinuousDilation(nn.Module):
    """Depthwise Conv1d with a *learnable* continuous dilation alpha.

    Implementation: run the same depthwise kernel at floor(alpha) and
    ceil(alpha) and linearly interpolate.  The kernel weights are shared,
    so alpha is the only extra parameter and gradients flow through the
    interpolation.
    """

    def __init__(self, channels, kernel_size=3, init_alpha=1.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        # Learnable continuous dilation
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        # Depthwise kernel: (channels, 1, k)
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(channels))

    def _conv_at_dilation(self, x, dilation):
        """Run depthwise conv at a specific integer dilation."""
        dilation = max(int(dilation), 1)
        pad = (self.kernel_size - 1) * dilation // 2
        return F.conv1d(x, self.weight, self.bias,
                        padding=pad, dilation=dilation,
                        groups=self.channels)

    def forward(self, x):
        """x: (B, C_group, T)"""
        alpha_clamped = self.alpha.clamp(1.0, 8.0)
        floor_d = alpha_clamped.floor()
        ceil_d = (floor_d + 1).clamp(max=8.0)
        beta = alpha_clamped - floor_d  # fractional part

        y_floor = self._conv_at_dilation(x, floor_d.item())
        y_ceil = self._conv_at_dilation(x, ceil_d.item())

        # Interpolate -- gradients flow to alpha via beta
        y = (1.0 - beta) * y_floor + beta * y_ceil
        return y


class MultiScaleTemporalContext(nn.Module):
    """MSTC front-end: splits channels into 3 groups, applies depthwise
    conv with learnable dilation per group, then fuses with pointwise
    conv + SE gate + residual.

    Input / output shape: (B, C, T)  with C = n_features, T = window_size.
    """

    def __init__(self, n_channels, se_reduction=8):
        super().__init__()
        self.n_channels = n_channels

        # Split into 3 groups as evenly as possible
        g1 = n_channels // 3
        g2 = n_channels // 3
        g3 = n_channels - g1 - g2
        self.groups = [g1, g2, g3]

        # One branch per group with different initial dilation
        self.branches = nn.ModuleList()
        for g_size, init_d in zip(self.groups, [1.0, 2.0, 3.0]):
            self.branches.append(nn.Sequential(
                _DepthwiseConv1dContinuousDilation(g_size, kernel_size=3,
                                                  init_alpha=init_d),
                nn.BatchNorm1d(g_size),
                nn.SiLU(inplace=True),
            ))

        # Pointwise fusion + SE
        self.pointwise = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.se = SqueezeExcitation1d(n_channels, reduction=se_reduction)

    # ---- public helpers for dilation logging ----
    def get_alphas(self):
        """Return current alpha values as a list of floats."""
        return [branch[0].alpha.item() for branch in self.branches]

    def forward(self, x):
        """x: (B, C, T)  ->  (B, C, T)"""
        # Split along channel dim
        splits = torch.split(x, self.groups, dim=1)
        outs = [branch(s) for branch, s in zip(self.branches, splits)]
        cat = torch.cat(outs, dim=1)       # (B, C, T)

        fused = self.se(self.pointwise(cat))  # (B, C, T)
        return x + fused                     # residual


# =========================================================================
# HRG -- Hierarchical Regime Guidance
# =========================================================================

class _HRGLevelBranch(nn.Module):
    """One level branch of HRG: depthwise conv -> BN -> SiLU -> pointwise -> BN -> SE."""

    def __init__(self, channels, dilation=1, se_reduction=8):
        super().__init__()
        pad = (3 - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad,
                      dilation=dilation, groups=channels),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.se = SqueezeExcitation1d(channels, reduction=se_reduction)

    def forward(self, x):
        return self.se(self.block(x))


class HierarchicalRegimeGuidance(nn.Module):
    """HRG post-encoder block.

    Input:  H in R^(B, C, T)  (C = d_model)
    Output: H' in R^(B, C, T), g in [0,1]^(B, 1, T)
    """

    L = 4  # number of hierarchy levels

    def __init__(self, d_model, se_reduction=8):
        super().__init__()
        C = d_model

        # L parallel level branches with increasing dilation
        self.levels = nn.ModuleList([
            _HRGLevelBranch(C, dilation=d, se_reduction=se_reduction)
            for d in [1, 2, 3, 4]
        ])

        # Fusion block: 4C -> 2C -> C
        self.fusion = nn.Sequential(
            nn.Conv1d(self.L * C, 2 * C, kernel_size=1),
            nn.BatchNorm1d(2 * C),
            nn.SiLU(inplace=True),
            nn.Conv1d(2 * C, C, kernel_size=1),
        )

        # Level predictor: pool over time -> 1x1 conv -> softmax
        self.level_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, C, 1)
        )
        self.level_fc = nn.Conv1d(C, self.L, kernel_size=1)  # (B, L, 1)

        # Guidance head: conv(k=3) -> BN -> ReLU -> 1x1 -> sigmoid
        self.guidance_head = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=3, padding=1),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
            nn.Conv1d(C, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, H):
        """
        Args:
            H: (B, C, T) encoder output
        Returns:
            H_prime: (B, C, T) refined encoder output
            g: (B, 1, T) temporal guidance map in [0, 1]
        """
        B, C, T = H.shape

        # --- level branches ---
        level_outs = [level(H) for level in self.levels]  # list of (B, C, T)

        # --- fusion -> reference tensor f ---
        cat = torch.cat(level_outs, dim=1)  # (B, 4C, T)
        f = self.fusion(cat)                # (B, C, T)

        # --- level predictor ---
        pooled = self.level_predictor(f)              # (B, C, 1)
        pi = self.level_fc(pooled).squeeze(2)         # (B, L)
        pi = F.softmax(pi, dim=1)                     # (B, L)

        # --- level-weighted mixture ---
        # h_bar = sum_l pi_l * h_l,  pi_l is scalar per sample
        stacked = torch.stack(level_outs, dim=1)       # (B, L, C, T)
        pi_expanded = pi.unsqueeze(2).unsqueeze(3)     # (B, L, 1, 1)
        h_bar = (stacked * pi_expanded).sum(dim=1)     # (B, C, T)

        # --- guidance head ---
        g = self.guidance_head(h_bar)  # (B, 1, T)

        # --- gated residual refinement ---
        H_prime = H + g * h_bar  # g broadcasts over C

        return H_prime, g


# =========================================================================
# Temporal Transformer Encoder (extended)
# =========================================================================

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
        max_seq_len=512,
        use_mstc=False,
        use_hrg=False,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.use_mstc = use_mstc
        self.use_hrg = use_hrg

        # Optional MSTC front-end (operates on raw features before projection)
        if use_mstc:
            self.mstc = MultiScaleTemporalContext(n_features, se_reduction=8)
        else:
            self.mstc = None

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

        # Optional HRG post-encoder block
        if use_hrg:
            self.hrg = HierarchicalRegimeGuidance(d_model, se_reduction=8)
        else:
            self.hrg = None

        # Pooling projection: mean + max pooled -> d_model
        self.pool_projection = nn.Linear(d_model * 2, d_model)

        # Store last guidance map for retrieval
        self._last_guidance_map = None

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, n_features)
            mask: Optional attention mask
        Returns:
            encoded: (batch_size, seq_len, d_model)
        """
        # --- MSTC front-end (before projection) ---
        if self.mstc is not None:
            # x is (B, T, C) -> need (B, C, T) for conv
            x = self.mstc(x.permute(0, 2, 1)).permute(0, 2, 1)  # back to (B, T, C)

        # Project input to model dimension
        x = self.input_projection(x)  # (B, L, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer
        encoded = self.transformer_encoder(x, mask=mask)

        # Normalize
        encoded = self.layer_norm(encoded)

        # --- HRG post-encoder ---
        if self.hrg is not None:
            # encoded is (B, T, C) -> need (B, C, T) for conv
            encoded_ct = encoded.permute(0, 2, 1)  # (B, C, T)
            refined, g = self.hrg(encoded_ct)
            encoded = refined.permute(0, 2, 1)      # back to (B, T, C)
            self._last_guidance_map = g              # (B, 1, T)
        else:
            self._last_guidance_map = None

        return encoded

    def get_sequence_embedding(self, x, mask=None):
        """
        Get global sequence embedding using mean + max pooling.
        Max-pooling preserves single-timestep anomaly peaks that
        mean-pooling would dilute by 1/seq_len.
        Args:
            x: (batch_size, seq_len, n_features)
        Returns:
            embedding: (batch_size, d_model)
        """
        encoded = self.forward(x, mask)  # (B, L, d_model)

        mean_pool = encoded.mean(dim=1)  # (B, d_model)
        max_pool = encoded.max(dim=1)[0]  # (B, d_model)

        combined = torch.cat([mean_pool, max_pool], dim=1)  # (B, d_model*2)
        embedding = self.pool_projection(combined)           # (B, d_model)

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
        reconstruction_weight=1.0,
        use_mstc=False,
        use_hrg=False,
    ):
        super().__init__()

        # Shared encoder
        self.encoder = TemporalTransformerEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_mstc=use_mstc,
            use_hrg=use_hrg,
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

    def get_guidance_map(self):
        """Retrieve the guidance map from the last forward pass (HRG).

        Returns:
            g: (B, 1, T) tensor in [0, 1], or None if HRG is disabled.
        """
        return self.encoder._last_guidance_map
