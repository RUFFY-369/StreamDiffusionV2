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
        # Use torch.unbind to preserve symbolic graph (avoid .tolist())
        # grid_sizes is [B, 3], so grid_sizes[i] is [3]
        f, h, w = torch.unbind(grid_sizes[i], dim=0)
        
        # Calculate seq len symbolically
        # Note: We cast to int for slicing, but slicing with specialized tensors usually works in TRT dynamic shapes
        # However, for looping in Python, we need values.
        # But here we are tracing.
        # Ideally, we used batch-aware ops, but loop is fine if B is small (1).
        
        # For tracing, slicing with tensor variables can be tricky.
        # But converting to int (.item()) BAKES the value.
        # We must keep it as tensor if possible, or assume it matches.
        
        # ACTUALLY: For TRT + ONNX, if we loop over B, and B is known (optimization profile), it unrolls.
        # But 'f', 'h', 'w' MUST be tensors to be dynamic.
        # If we slice x[i, :f*h*w], PyTorch ONNX exporter handles dynamic slice.
        
        actual_seq_len = f * h * w
        
        # Get the relevant portion
        # We use dynamic slicing
        x_i = x[i, :actual_seq_len].float()  # [seq, n, d]
        
        # Build frequency tensor for this sample
        freqs_f = freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1)
        freqs_h = freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
        freqs_w = freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        freqs_i = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(actual_seq_len, 1, -1)  # [seq, 1, half_d]
        
        # Real-number rotary embedding (avoid complex numbers for ONNX)
        # CORRECTION: WanModel uses adjacent pairs (view_as_complex on last dim)
        # So Real = x[0::2], Imag = x[1::2]
        x_real = x_i[..., 0::2]
        x_imag = x_i[..., 1::2]
        
        # freqs_i is [seq, 1, half_d] -> expand to [seq, n, half_d]
        cos_freqs = freqs_i.cos().expand(-1, n, -1)
        sin_freqs = freqs_i.sin().expand(-1, n, -1)
        
        # Rotate
        # (a + ib) * (cos + isin) = (acos - bsin) + i(asin + bcos)
        out_real = x_real * cos_freqs - x_imag * sin_freqs
        out_imag = x_real * sin_freqs + x_imag * cos_freqs
        
        # Interleave back: stack on last dim then flatten
        # Stack: [seq, n, half_d, 2] -> Flatten: [seq, n, d]
        # Use simple operations to avoid TRT shape overflow
        x_rotated = torch.cat([out_real.unsqueeze(-1), out_imag.unsqueeze(-1)], dim=-1)
        x_rotated = x_rotated.reshape(x_rotated.shape[:-2] + (d,))
        
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
        f, h, w = torch.unbind(grid_sizes[i], dim=0)
        actual_seq_len = f * h * w
        
        # start_frame should be tensor
        sf = start_frame[i]
        
        # Casting to int for Slicing indices:
        # If we use tensors for slicing: freqs[sf:sf+f]
        # PyTorch supports this in export.
        
        x_i = x[i, :actual_seq_len].float()  # [seq, n, d]
        
        # Build frequency tensor with offset
        # Use explicit dynamic indexing to prevent constant folding of 'sf'
        idx_f = torch.arange(f, device=x.device) + sf
        # freq_max must be a python int or tensor on same device
        freq_max = int(freqs_split[0].shape[0]) - 1
        idx_f = idx_f.long().clamp(max=freq_max) 
        
        freqs_f = freqs_split[0][idx_f].view(f, 1, 1, -1).expand(f, h, w, -1)
        freqs_h = freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
        freqs_w = freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        freqs_i = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(actual_seq_len, 1, -1)  # [seq, 1, half_d]
        
        # Real-number rotary embedding (avoid complex numbers for ONNX)
        # CORRECTION: WanModel uses adjacent pairs (view_as_complex on last dim)
        x_real = x_i[..., 0::2]
        x_imag = x_i[..., 1::2]
        
        cos_freqs = freqs_i.cos().expand(-1, n, -1)
        sin_freqs = freqs_i.sin().expand(-1, n, -1)
        
        # Rotate
        out_real = x_real * cos_freqs - x_imag * sin_freqs
        out_imag = x_real * sin_freqs + x_imag * cos_freqs
        
        # Interleave back
        x_rotated = torch.cat([out_real.unsqueeze(-1), out_imag.unsqueeze(-1)], dim=-1)
        x_rotated = x_rotated.reshape(x_rotated.shape[:-2] + (d,))
        
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
        start_frame: Optional[torch.Tensor] = None,
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
        # Compute Q, K, V
        # Use -1 for sequence length to ensure dynamic reshaping in ONNX
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(x)).view(b, -1, n, d)
        v = self.v(x).view(b, -1, n, d)
        
        # Apply RoPE
        if kv_cache_k is None:
            # Non-streaming: full sequence RoPE
            q = trt_rope_apply(q, grid_sizes, freqs)
            k = trt_rope_apply(k, grid_sizes, freqs)
        else:
            # Streaming: offset RoPE based on current position
            # Ensure frame_seqlen is tensor
            grid_i = grid_sizes[0] # Assume B=1 or same grid
            frame_seqlen = grid_i[1] * grid_i[2]
            
            # Symbolic division
            if start_frame is None and current_start is not None:
                start_frame = current_start // frame_seqlen
                
            q = trt_causal_rope_apply(q, grid_sizes, freqs, start_frame)
            k = trt_causal_rope_apply(k, grid_sizes, freqs, start_frame)
        
        # Handle KV cache
        if kv_cache_k is not None and kv_cache_v is not None:
            # Update cache with new K, V
            new_kv_cache_k = kv_cache_k.clone()
            new_kv_cache_v = kv_cache_v.clone()
            
            # Vectorized or tensor-friendly update
            # We assume batch size is small (usually 1 for streaming)
            if current_start is not None:
                # Use tensor operations to update cache to preserve graph connections
                indices = torch.arange(s, device=x.device).expand(b, s)
                start_indices = current_start.view(b, 1)
                target_indices = indices + start_indices
                
                # Careful scatter update
                # Since we update a slice [start:start+s], we can generate indices
                # New K/V are [B, S, H, D]
                # Cache is [B, MaxLen, H, D]
                
                # Handling batch loop purely with tensors is tricky for scatter if B > 1 and starts differ
                # But we can iterate B since it is a loop in graph (or unrolled if small)
                for i in range(b):
                    # Get start as tensor (0-d or 1-d)
                    start_t = current_start[i] 
                    idx = torch.arange(s, device=x.device) + start_t
                    
                    # Ensure indices are within bounds (clamp or mask)
                    # For streaming, we assume caller manages logic, but let's be safe for tracing
                    max_len = new_kv_cache_k.shape[1]
                    mask = idx < max_len
                    valid_idx = idx[mask]
                    
                    if valid_idx.numel() > 0:
                        # Index update: cache[i, valid_idx] = k[i, :num_valid]
                        # We must slice the source 'k' as well if we clamped
                        valid_src = k[i, :valid_idx.numel()]
                        
                        # Use index_put_ or simple indexing which traces to Scatter/IndexPut
                        new_kv_cache_k[i, valid_idx] = valid_src
                        new_kv_cache_v[i, valid_idx] = v[i, :valid_idx.numel()]
                
                # Define end_idx for slicing the full cache for attention
                # We use max() to handle batching, though usually B=1
                end_idx = (current_start + s).max()
            else:
                # Append logic (non-streaming legacy path)
                for i in range(b):
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
        # Compute attention using SDPA (ONNX compatible)
        # Dynamic Masking for Ring Buffer / Infinite Streaming
        # We cannot rely on the static 'causal_mask' input because:
        # 1. In Linear phase (Log >> 0), slicing mask[:s] uses rows 0..s instead of Log..Log+s
        # 2. In Ring phase, Physical ordering != Logical ordering
        
        # Strategy: 
        # - Default: Attend to everything (History is valid)
        # - Constraint: Within the current chunk (Physical: current_start...current_start+s), enforces causality.
        
        attn_mask = None
        if s > 1 and kv_cache_k is not None and current_start is not None:
            # Construct dynamic mask
            # Shape: [B, 1, s, Total_K] -> Broadcast over heads
            total_k = k_full.shape[2]
            
            # Start with all True (Attend to everything)
            # Use float mask for TRT compatibility often? boolean is fine for SDPA.
            # We use float -inf for masked, 0 for allowed if additive?
            # SDPA supports boolean: True = attend, False = mask.
            
            # Apply Causal constraint to the current writing block
            # USE TENSOR INDEXING to prevent constant folding
            c_start = current_start[0] # Tensor scalar-like
            c_end = c_start + s
            
            # We must use tensor-based masking
            # Create indices [0...Total_K]
            col_indices = torch.arange(total_k, device=q.device).view(1, 1, 1, total_k)
            
            # 1. Intra-chunk causality: mask [c_start : c_end] with tril
            # This is hard to do with pure vector logic without scattering.
            # But the local_causal pattern is static shape (s, s).
            # We can use previous logic IF slice assignment supports tensor indices?
            # Assigning to mask[:, :, :, c_start:c_end] works if c_start is int.
            # If c_start is tensor, we need:
            # mask.index_put_((slice(None), slice(None), slice(None), range_tensor), local_causal)
            # Range tensor:
            range_tensor = torch.arange(s, device=q.device) + c_start
            
            # Clamp range to be safe (though it should fit)
            max_val = torch.tensor(total_k - 1, device=q.device, dtype=torch.long)
            range_tensor = range_tensor.clamp(max=max_val)
            
            # Create expansion of local_causal to match mask dims?
            # We need to scatter the local_causal into the big mask.
            # TRT doesn't love scatter.
            # Alternative: Construct mask analytically.
            # A position (i, j) in (s, total_k) is ALLOWED if:
            #   (j < c_start) OR (j >= c_start AND j < c_end AND j <= (c_start + i))
            #   AND j < valid_end
            
            # Row indices [0...s]
            row_indices = torch.arange(s, device=q.device).view(1, 1, s, 1)
            
            # Condition 1: History (j < c_start) -> True
            cond_history = col_indices < c_start.view(1, 1, 1, 1)
            
            # Condition 2: Intra-chunk (c_start <= j < c_end AND j <= c_start + i)
            # j relative to chunk start: j_rel = j - c_start
            cond_intra_range = (col_indices >= c_start.view(1, 1, 1, 1)) & (col_indices < c_end.view(1, 1, 1, 1))
            cond_causal = col_indices <= (c_start.view(1, 1, 1, 1) + row_indices)
            cond_intra = cond_intra_range & cond_causal
            
            # Initial Allowed Mask (History + Intra-Causal)
            mask = cond_history | cond_intra
            
            # Condition 3: Valid Memory (Optimization for Zeros)
            # Check for valid usage of Ring Buffer
            f_g, h_g, w_g = torch.unbind(grid_sizes[0], dim=0)
            frame_len = f_g * h_g * w_g
            
            sf_val = start_frame[0] # Tensor
            logical_valid = (sf_val * frame_len) + s # Tensor
            
            # valid_end = min(logical_valid, total_k)
            valid_end = torch.min(logical_valid, torch.tensor(total_k, device=q.device))
            
            cond_valid = col_indices < valid_end.view(1, 1, 1, 1)
            
            # Final Mask = (History OR Intra) AND Valid_Memory
            mask = mask & cond_valid
            
            attn_mask = mask
        elif causal_mask is not None:
            # Fallback for non-streaming / static cases
            attn_mask = causal_mask[:s, :k_full.shape[2]]

        out = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        # Reshape back: [B, L, C]
        # Reshape back: [B, L, C]
        # Use -1 for sequence length
        out = out.transpose(1, 2).contiguous().view(b, -1, self.dim)
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
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(b, -1, self.dim)
        out = self.o(out)
        
        # Return with cache (transposed back)
        return out, k.transpose(1, 2), v.transpose(1, 2)
