# Copyright 2025-26 StreamDiffusionV2 Authors
"""
TensorRT-compatible attention implementations.

Replaces FlexAttention with standard scaled_dot_product_attention for ONNX export.
Handles KV cache as explicit tensor inputs/outputs for TensorRT binding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


# Local RMSNorm implementation to avoid external dependencies
class WanRMSNorm(nn.Module):
    """RMSNorm implementation for TensorRT compatibility."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight



def create_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a causal attention mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def create_block_causal_mask(
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create a block-wise causal mask for video diffusion.
    
    Each frame block can only attend to itself and previous blocks.
    This replaces FlexAttention's dynamic block mask with a static tensor.
    """
    total_length = num_frames * frame_seqlen
    mask = torch.zeros(total_length, total_length, device=device, dtype=dtype)
    
    # Block-wise causal: each block attends to all previous blocks
    block_size = frame_seqlen * num_frame_per_block
    for i in range(0, total_length, block_size):
        block_end = min(i + block_size, total_length)
        # This block can attend to everything up to block_end
        mask[i:block_end, block_end:] = float('-inf')
    
    return mask


def trt_rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings - TensorRT compatible version.
    
    Uses real-number operations (sin/cos rotation) instead of complex multiplication
    for ONNX compatibility.
    
    Args:
        x: [B, L, num_heads, head_dim]
        grid_sizes: [B, 3] containing (F, H, W)
        freqs: [max_seq, head_dim // 2] - the cos/sin frequencies
    
    Returns:
        Tensor with RoPE applied, same shape as input
    """
    b, seq_len, n, d = x.shape
    half_d = d // 2
    
    # Split freqs for F, H, W dimensions  
    freq_splits = [half_d - 2 * (half_d // 3), half_d // 3, half_d // 3]
    freqs_split = freqs.split(freq_splits, dim=1)
    
    output = []
    for i in range(b):
        f, h, w = grid_sizes[i].tolist()
        actual_seq_len = int(f * h * w)
        
        # Get the relevant portion and convert to float32 for precision
        x_i = x[i, :actual_seq_len].float()  # [seq, n, d]
        
        # Build frequency tensor for this sample
        freqs_f = freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1)
        freqs_h = freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
        freqs_w = freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        freqs_i = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(actual_seq_len, 1, -1)  # [seq, 1, half_d]
        
        # Real-number rotary embedding (avoid complex numbers for ONNX)
        # Split x into two halves for rotation
        x_1, x_2 = x_i[..., :half_d], x_i[..., half_d:]  # [seq, n, half_d] each
        
        # Get cos and sin of frequencies
        cos_freqs = freqs_i.cos().expand(-1, n, -1)  # [seq, n, half_d]
        sin_freqs = freqs_i.sin().expand(-1, n, -1)  # [seq, n, half_d]
        
        # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        x_rotated_1 = x_1 * cos_freqs - x_2 * sin_freqs
        x_rotated_2 = x_1 * sin_freqs + x_2 * cos_freqs
        
        x_rotated = torch.cat([x_rotated_1, x_rotated_2], dim=-1)  # [seq, n, d]
        
        # Handle padding
        if actual_seq_len < seq_len:
            x_rotated = torch.cat([x_rotated, x[i, actual_seq_len:].float()], dim=0)
        
        output.append(x_rotated)
    
    return torch.stack(output).type_as(x)


def trt_causal_rope_apply(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary position embeddings for causal streaming inference.
    
    Uses real-number operations (sin/cos rotation) instead of complex multiplication
    for ONNX compatibility.
    
    Args:
        x: [B, L, num_heads, head_dim]
        grid_sizes: [B, 3] containing (F, H, W)  
        freqs: [max_seq, head_dim // 2]
        start_frame: [B] frame index offset for each batch item
    
    Returns:
        Tensor with RoPE applied
    """
    b, seq_len, n, d = x.shape
    half_d = d // 2
    
    freq_splits = [half_d - 2 * (half_d // 3), half_d // 3, half_d // 3]
    freqs_split = freqs.split(freq_splits, dim=1)
    
    output = []
    for i in range(b):
        f, h, w = grid_sizes[i].tolist()
        actual_seq_len = int(f * h * w)
        sf = int(start_frame[i].item()) if isinstance(start_frame, torch.Tensor) else int(start_frame)
        
        x_i = x[i, :actual_seq_len].float()  # [seq, n, d]
        
        # Build frequency tensor with offset
        freqs_f = freqs_split[0][sf:sf + f].view(f, 1, 1, -1).expand(f, h, w, -1)
        freqs_h = freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
        freqs_w = freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        freqs_i = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(actual_seq_len, 1, -1)  # [seq, 1, half_d]
        
        # Real-number rotary embedding (avoid complex numbers for ONNX)
        x_1, x_2 = x_i[..., :half_d], x_i[..., half_d:]  # [seq, n, half_d] each
        
        cos_freqs = freqs_i.cos().expand(-1, n, -1)
        sin_freqs = freqs_i.sin().expand(-1, n, -1)
        
        x_rotated_1 = x_1 * cos_freqs - x_2 * sin_freqs
        x_rotated_2 = x_1 * sin_freqs + x_2 * cos_freqs
        
        x_rotated = torch.cat([x_rotated_1, x_rotated_2], dim=-1)  # [seq, n, d]
        
        if actual_seq_len < seq_len:
            x_rotated = torch.cat([x_rotated, x[i, actual_seq_len:].float()], dim=0)
        
        output.append(x_rotated)
    
    return torch.stack(output).type_as(x)


class TRTCausalSelfAttention(nn.Module):
    """
    TensorRT-compatible causal self-attention.
    
    Replaces FlexAttention with scaled_dot_product_attention for ONNX export.
    KV cache is handled as explicit input/output tensors.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.scale = self.head_dim ** -0.5
        
        # Layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        kv_cache_k: Optional[torch.Tensor] = None,
        kv_cache_v: Optional[torch.Tensor] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        current_start: Optional[torch.Tensor] = None,
        current_end: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with optional KV cache.
        
        Args:
            x: Input tensor [B, L, C]
            seq_lens: Sequence lengths [B]
            grid_sizes: Grid sizes [B, 3]
            freqs: RoPE frequencies
            causal_mask: Pre-computed causal mask [L, L] or None
            kv_cache_k: Key cache [B, cache_len, num_heads, head_dim]
            kv_cache_v: Value cache [B, cache_len, num_heads, head_dim]
            cache_seqlens: Current cache lengths [B]
            current_start: Start indices for cache update [B]
            current_end: End indices for cache update [B]
        
        Returns:
            output: Attention output [B, L, C]
            new_kv_cache_k: Updated key cache
            new_kv_cache_v: Updated value cache
        """
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        
        # Compute Q, K, V
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        
        # Apply RoPE
        if kv_cache_k is None:
            # Non-streaming: full sequence RoPE
            q = trt_rope_apply(q, grid_sizes, freqs)
            k = trt_rope_apply(k, grid_sizes, freqs)
        else:
            # Streaming: offset RoPE based on current position
            frame_seqlen = grid_sizes[0, 1].item() * grid_sizes[0, 2].item()
            start_frame = current_start // frame_seqlen
            q = trt_causal_rope_apply(q, grid_sizes, freqs, start_frame)
            k = trt_causal_rope_apply(k, grid_sizes, freqs, start_frame)
        
        # Handle KV cache
        if kv_cache_k is not None and kv_cache_v is not None:
            # Update cache with new K, V
            new_kv_cache_k = kv_cache_k.clone()
            new_kv_cache_v = kv_cache_v.clone()
            
            for i in range(b):
                if current_start is not None:
                    # Ring buffer update using explicit start/end indices
                    start_idx = int(current_start[i].item()) if isinstance(current_start, torch.Tensor) else int(current_start)
                    # For simplicty in TRT, we assume contiguous update matchng input length
                    # The original PyTorch logic handles complex rolling, but here we assume
                    # the cache is large enough or managed as a ring buffer externally
                    end_idx = start_idx + s
                    
                    # Handle wrapping if implementing full ring buffer inside TRT?
                    # For now, simplistic slice update
                    if end_idx <= new_kv_cache_k.shape[1]:
                         new_kv_cache_k[i, start_idx:end_idx] = k[i]
                         new_kv_cache_v[i, start_idx:end_idx] = v[i]
                    else:
                        # naive wrap around handling if needed, or just clamp
                        valid_len = new_kv_cache_k.shape[1] - start_idx
                        if valid_len > 0:
                            new_kv_cache_k[i, start_idx:start_idx+valid_len] = k[i, :valid_len]
                            new_kv_cache_v[i, start_idx:start_idx+valid_len] = v[i, :valid_len]
                else:
                    # Append logic (non-streaming or simple growing cache)
                    start_idx = int(cache_seqlens[i].item()) if cache_seqlens is not None else 0
                    end_idx = start_idx + s
                    new_kv_cache_k[i, start_idx:end_idx] = k[i]
                    new_kv_cache_v[i, start_idx:end_idx] = v[i]
            
            # Use full cache for attention
            k_full = new_kv_cache_k[:, :end_idx]
            v_full = new_kv_cache_v[:, :end_idx]
        else:
            k_full = k
            v_full = v
            new_kv_cache_k = None
            new_kv_cache_v = None
        
        # Reshape for attention: [B, num_heads, L, head_dim]
        q = q.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)
        
        # Compute attention using SDPA (ONNX compatible)
        if causal_mask is not None:
            # Expand mask for batch and heads
            attn_mask = causal_mask[:s, :k_full.shape[2]]
            out = F.scaled_dot_product_attention(
                q, k_full, v_full,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False  # We provide explicit mask
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k_full, v_full,
                dropout_p=0.0,
                is_causal=(kv_cache_k is None)  # Causal only for non-cached
            )
        
        # Reshape back: [B, L, C]
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.o(out)
        
        return out, new_kv_cache_k, new_kv_cache_v


class TRTCrossAttention(nn.Module):
    """
    TensorRT-compatible cross-attention for text conditioning.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
        crossattn_cache_k: Optional[torch.Tensor] = None,
        crossattn_cache_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cross-attention with optional caching.
        
        Args:
            x: Query input [B, L, C]
            context: Key/Value context [B, ctx_len, C]
            context_lens: Context lengths [B]
            crossattn_cache_k: Cached keys [B, ctx_len, num_heads, head_dim]
            crossattn_cache_v: Cached values [B, ctx_len, num_heads, head_dim]
        
        Returns:
            output, cached_k, cached_v
        """
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        
        # Use cache if available, otherwise compute K, V
        if crossattn_cache_k is not None and crossattn_cache_v is not None:
            k = crossattn_cache_k
            v = crossattn_cache_v
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [B, n, s, d]
        k = k.transpose(1, 2)  # [B, n, ctx_len, d]
        v = v.transpose(1, 2)  # [B, n, ctx_len, d]
        
        # SDPA
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.o(out)
        
        # Return with cache (transposed back)
        return out, k.transpose(1, 2), v.transpose(1, 2)
