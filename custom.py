import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
import numpy as np
from scipy.signal import butter, sosfilt


########################################################################################
# Architecture Flow (Epilepsy Detection):
# Input EEG → [Theta (4-8 Hz - Ictal activity) + Alpha (8-13 Hz - Blocking)] → MRCNN → Spatial Attention → Dilated Causal Conv → Multi-Head Attention → TCE → Classification
########################################################################################


class BandpassFilter(nn.Module):
    """
    Bandpass Filter for EEG signal preprocessing.
    Uses Butterworth IIR filter for frequency domain filtering.
    """
    def __init__(self, sampling_rate=100, lowcut=0.5, highcut=4.0, order=4):
        super(BandpassFilter, self).__init__()
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        
        nyquist = sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        self.sos = butter(order, [low, high], btype='band', output='sos')
        
    def forward(self, x):
        device = x.device
        x_cpu = x.cpu().detach().numpy()
        
        batch_size, channels, seq_len = x_cpu.shape
        filtered = np.zeros_like(x_cpu)
        
        for b in range(batch_size):
            for c in range(channels):
                filtered[b, c, :] = sosfilt(self.sos, x_cpu[b, c, :])
        
        return torch.from_numpy(filtered).float().to(device)


class ThetaBandFilter(BandpassFilter):
    """Theta Band (4-8 Hz) - Sleep spindles & K-complexes"""
    def __init__(self, sampling_rate=100):
        super().__init__(sampling_rate, lowcut=4.0, highcut=8.0, order=4)


class AlphaBandFilter(BandpassFilter):
    """Alpha Band (8-13 Hz) - Awake relaxation & REM sleep"""
    def __init__(self, sampling_rate=100):
        super().__init__(sampling_rate, lowcut=8.0, highcut=13.0, order=4)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, time)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(concat)
        attention = self.sigmoid(attention)
        return x * attention


class DilatedCausalConv1d(nn.Module):
    """
    Dilated Causal 1D Convolution.
    - Uses dilation to capture longer-range dependencies
    - Maintains causality: no future information is used
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=2, stride=1):
        super(DilatedCausalConv1d, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.padding = padding
        
    def forward(self, x):
        x = self.conv(x)
        # Remove extra padding to maintain causality and sequence length
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class MRCNNWithDilatedConv(nn.Module):
    """
    Multi-Resolution CNN with Spatial Attention.
    Output of this module goes directly into Dilated Causal Conv.
    """
    def __init__(self, spatial_attn_reduced_cnn_size):
        super(MRCNNWithDilatedConv, self).__init__()
        drate = 0.5
        
        # Small-kernel branch
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        # Wide-kernel branch
        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.dropout = nn.Dropout(drate)
        
        # Spatial Attention after concatenation
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # Channel adaptation convolution
        self.channel_adapter = nn.Conv1d(128, spatial_attn_reduced_cnn_size, kernel_size=1)
        self.bn_adapter = nn.BatchNorm1d(spatial_attn_reduced_cnn_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        # Extract multi-resolution features
        x1 = self.features1(x)
        x2 = self.features2(x)
        
        # Concatenate along time dimension
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        
        # Apply spatial attention
        x_attn = self.spatial_attention(x_concat)
        
        # Adapt channels for next stage
        x_adapted = self.channel_adapter(x_attn)
        x_adapted = self.bn_adapter(x_adapted)
        x_adapted = self.gelu(x_adapted)
        
        return x_adapted


def attention(query, key, value, mask=None, dropout=None):
    """Scaled Dot-Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttentionWithDilatedInput(nn.Module):
    """
    Multi-Head Attention that takes input from Dilated Causal Conv.
    The input to this module is the output of the Dilated Causal Conv.
    """
    def __init__(self, h, d_model, spatial_attn_reduced_cnn_size, dropout=0.1):
        super(MultiHeadedAttentionWithDilatedInput, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.spatial_dim = spatial_attn_reduced_cnn_size
        self.d_model = d_model

        # Linear projections for Q, K, V (from dilated conv output)
        self.query_proj = nn.Linear(self.spatial_dim, d_model)
        self.key_proj = nn.Linear(self.spatial_dim, d_model)
        self.value_proj = nn.Linear(self.spatial_dim, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, self.spatial_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_dilated, return_attn=False):
        """
        Args:
            x_dilated: Output from Dilated Causal Conv
                      Shape: (batch, time, spatial_dim) or (batch, spatial_dim, time)
            return_attn: bool, whether to also return attention weights.
        
        Returns:
            attention output with same shape as input, optionally (output, attn_weights)
        """
        # Ensure proper shape: (batch, time, spatial_dim)
        if x_dilated.dim() == 3:
            if x_dilated.size(1) == self.spatial_dim:
                # Already (batch, spatial_dim, time) -> transpose
                x_dilated = x_dilated.transpose(1, 2)
            elif x_dilated.size(2) == self.spatial_dim:
                # Already (batch, time, spatial_dim) -> keep
                pass
            else:
                raise ValueError(f"Unexpected input shape: {x_dilated.shape}. Expected spatial_dim={self.spatial_dim} at dim 1 or 2")
        else:
            raise ValueError(f"Expected 3D input, got {x_dilated.dim()}D")
        
        batch_size, seq_len, feature_dim = x_dilated.size()
        assert feature_dim == self.spatial_dim, f"Feature dimension mismatch: {feature_dim} vs {self.spatial_dim}"
        
        # Project to Q, K, V
        query = self.query_proj(x_dilated).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = self.key_proj(x_dilated).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = self.value_proj(x_dilated).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attn_weights = attention(query, key, value, dropout=self.dropout)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.h * self.d_k)
        
        # Project back to spatial dimension
        x = self.output_proj(x)

        if return_attn:
            return x, attn_weights
        return x


class LayerNorm(nn.Module):
    """Layer Normalization"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.w2(self.dropout(self.gelu(self.w1(x))))


class TCEEncoderLayer(nn.Module):
    """
    Temporal Context Encoder Layer.
    Now includes: Residual Connection → LayerNorm → MHA → Residual Connection → LayerNorm → FFN
    """
    def __init__(self, spatial_dim, attention, feed_forward, dropout=0.1):
        super(TCEEncoderLayer, self).__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(spatial_dim)
        self.norm2 = LayerNorm(spatial_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Attention sublayer with residual
        attn_output = self.attention(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward sublayer with residual
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x


class TCE(nn.Module):
    """
    Temporal Context Encoder (stacked layers).
    Processes the output from Multi-Head Attention.
    """
    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.norm2.gamma.size(0))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EpilepsyWithDilatedConv(nn.Module):
    """
    ARCHITECTURE FLOW:
    1. Input EEG → MRCNN → Spatial Attention
    2. Output → Dilated Causal Conv
    3. Output → Multi-Head Attention (MHA)
    4. Output → Temporal Context Encoder (TCE)
    5. Output → Classification
    """
    def __init__(self, num_classes=5, dilation_rate=2, num_tce_layers=2, sampling_rate=100):
        super(EpilepsyWithDilatedConv, self).__init__()
        
        # Hyperparameters
        spatial_dim = 30  # Reduced CNN channel size
        d_model = 80      # Model dimension for attention
        d_ff = 120        # Feed-forward dimension
        num_heads = 5     # Number of attention heads
        dropout = 0.1
        
        # 0. Multi-Band Filters (Theta + Alpha for Epilepsy Detection)
        self.theta_filter = ThetaBandFilter(sampling_rate=sampling_rate)
        self.alpha_filter = AlphaBandFilter(sampling_rate=sampling_rate)
        
        # 1. Feature Extraction: MRCNN + Spatial Attention
        self.feature_extractor = MRCNNWithDilatedConv(spatial_dim)
        
        # 2. Dilated Causal Convolution
        self.dilated_conv = DilatedCausalConv1d(
            in_channels=spatial_dim,
            out_channels=spatial_dim,
            kernel_size=7,
            dilation=dilation_rate
        )
        self.dilated_norm = nn.BatchNorm1d(spatial_dim)
        self.dilated_activation = nn.GELU()
        self.dilated_dropout = nn.Dropout(dropout)
        
        # 3. Multi-Head Attention (takes output of dilated conv)
        self.multihead_attention = MultiHeadedAttentionWithDilatedInput(
            h=num_heads,
            d_model=d_model,
            spatial_attn_reduced_cnn_size=spatial_dim,
            dropout=dropout
        )
        self.attention_dropout = nn.Dropout(dropout)
        
        # 4. Temporal Context Encoder
        feed_forward = PositionwiseFeedForward(spatial_dim, d_ff, dropout)
        
        encoder_layer = TCEEncoderLayer(
            spatial_dim=spatial_dim,
            attention=self.multihead_attention,
            feed_forward=feed_forward,
            dropout=dropout
        )
        
        self.tce = TCE(encoder_layer, num_tce_layers)
        
        # 5. Classification Head
        # We need to determine the flattened size
        # This depends on the output shape from TCE
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, return_attn=False):
        """
        Forward pass with corrected flow.
        
        Args:
            x: Input EEG signal (batch_size, 1, sequence_length) or (batch_size, sequence_length)
            return_attn: bool, if True returns (output, attn_weights)
            
        Returns:
            output: Class logits (batch_size, num_classes)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size = x.size(0)
        
        # 0. Multi-Band Filtering (Theta + Alpha for Epilepsy Detection)
        x_theta = self.theta_filter(x)    # (batch, 1, time) - Ictal activity
        x_alpha = self.alpha_filter(x)    # (batch, 1, time) - Blocking patterns
        
        # Combine bands: average them
        x_input = (x_theta + x_alpha) / 2.0
        
        # 1. Feature Extraction
        x_features = self.feature_extractor(x_input)  # (batch, spatial_dim, time)
        
        # 2. Dilated Causal Convolution
        x_dilated = self.dilated_conv(x_features)
        x_dilated = self.dilated_norm(x_dilated)
        x_dilated = self.dilated_activation(x_dilated)
        x_dilated = self.dilated_dropout(x_dilated)
        
        # 3. Prepare for Multi-Head Attention
        # Transpose to (batch, time, spatial_dim)
        x_attn_input = x_dilated.transpose(1, 2)
        
        # 4. Multi-Head Attention
        x_attn, attn_weights = self.multihead_attention(x_attn_input, return_attn=True)
        x_attn = self.attention_dropout(x_attn)
        
        # 5. Temporal Context Encoder
        x_tce = self.tce(x_attn)  # (batch, time, spatial_dim)
        
        # 6. Global pooling and classification
        # Transpose back to (batch, spatial_dim, time) for pooling
        x_tce = x_tce.transpose(1, 2)
        x_pooled = self.global_pool(x_tce).squeeze(-1)  # (batch, spatial_dim)
        
        # 7. Classification
        output = self.classifier(x_pooled)  # (batch, num_classes)
        
        if return_attn:
            return output, attn_weights
        return output
