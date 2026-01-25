# Copyright 2025-26 StreamDiffusionV2 Authors
"""
TensorRT-exportable variant of CausalWanModel.

Replaces FlexAttention with standard SDPA for ONNX compatibility.
KV cache is handled as explicit tensor inputs/outputs.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

from causvid.models.wan.wan_base.modules.model import (
    WanRMSNorm,
    WanLayerNorm,
    Head,
    MLPProj,
    sinusoidal_embedding_1d,
    rope_params,
)
from .attention import TRTCausalSelfAttention, TRTCrossAttention, trt_rope_apply


class TRTCrossAttentionT2V(nn.Module):
    """TensorRT-compatible T2V cross-attention."""
    
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
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
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        
        if cache_k is not None and cache_v is not None:
            k, v = cache_k, cache_v
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
        
        # SDPA expects [B, heads, L, dim]
        q = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k_t, v_t, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.o(out)
        
        return out, k, v


class TRTWanAttentionBlock(nn.Module):
    """TensorRT-compatible attention block."""
    
    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        
        # Layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = TRTCausalSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = TRTCrossAttentionT2V(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )
        
        # Modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor],
        kv_cache_k: Optional[torch.Tensor] = None,
        kv_cache_v: Optional[torch.Tensor] = None,
        cross_cache_k: Optional[torch.Tensor] = None,
        cross_cache_v: Optional[torch.Tensor] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        current_start: Optional[torch.Tensor] = None,
        current_end: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with explicit KV cache I/O.
        
        Returns:
            x, new_kv_k, new_kv_v, new_cross_k, new_cross_v
        """
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        
        # Modulation
        e_mod = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        
        # Self-attention
        x_normed = self.norm1(x).unflatten(1, (num_frames, frame_seqlen))
        x_modulated = (x_normed * (1 + e_mod[1]) + e_mod[0]).flatten(1, 2)
        
        y, new_kv_k, new_kv_v = self.self_attn(
            x_modulated, seq_lens, grid_sizes, freqs, causal_mask,
            kv_cache_k, kv_cache_v, cache_seqlens, current_start, current_end
        )
        
        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e_mod[2]).flatten(1, 2)
        
        # Cross-attention
        cross_out, new_cross_k, new_cross_v = self.cross_attn(
            self.norm3(x), context, context_lens, cross_cache_k, cross_cache_v
        )
        x = x + cross_out
        
        # FFN
        x_normed2 = self.norm2(x).unflatten(1, (num_frames, frame_seqlen))
        y = self.ffn((x_normed2 * (1 + e_mod[4]) + e_mod[3]).flatten(1, 2))
        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e_mod[5]).flatten(1, 2)
        
        return x, new_kv_k, new_kv_v, new_cross_k, new_cross_v


class CausalWanModelTRTExport(nn.Module):
    """
    TensorRT-exportable variant of CausalWanModel.
    
    Key differences from original:
    1. FlexAttention replaced with SDPA
    2. KV cache as explicit tensor inputs/outputs (not dicts)
    3. Causal mask as tensor (not BlockMask)
    4. No dynamic block_mask creation
    """
    
    def __init__(
        self,
        model_type: str = 't2v',
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 1536,
        ffn_dim: int = 8960,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 12,
        num_layers: int = 30,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )
        
        # Attention blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            TRTWanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps
            )
            for _ in range(num_layers)
        ])
        
        # Head (simplified for export)
        self.head_norm = WanLayerNorm(dim, eps)
        self.head_linear = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.head_modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        
        # RoPE frequencies
        d = dim // num_heads
        self.register_buffer('freqs', torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1))
        
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
    
    @classmethod
    def from_pretrained_model(cls, original_model) -> 'CausalWanModelTRTExport':
        """Create TRT-exportable model from pretrained CausalWanModel."""
        # Access attributes directly from model (not from config dict)
        trt_model = cls(
            model_type=original_model.model_type,
            patch_size=original_model.patch_size,
            text_len=original_model.text_len,
            in_dim=original_model.in_dim,
            dim=original_model.dim,
            ffn_dim=original_model.ffn_dim,
            freq_dim=original_model.freq_dim,
            text_dim=original_model.text_dim,
            out_dim=original_model.out_dim,
            num_heads=original_model.num_heads,
            num_layers=original_model.num_layers,
            window_size=original_model.window_size,
            qk_norm=original_model.qk_norm,
            cross_attn_norm=original_model.cross_attn_norm,
            eps=original_model.eps,
        )
        
        # Copy weights from original model
        trt_model.patch_embedding.load_state_dict(original_model.patch_embedding.state_dict())
        trt_model.text_embedding.load_state_dict(original_model.text_embedding.state_dict())
        trt_model.time_embedding.load_state_dict(original_model.time_embedding.state_dict())
        trt_model.time_projection.load_state_dict(original_model.time_projection.state_dict())
        
        # Copy attention block weights
        for i, (trt_block, orig_block) in enumerate(zip(trt_model.blocks, original_model.blocks)):
            # Copy normalization layers
            trt_block.norm1.load_state_dict(orig_block.norm1.state_dict())
            trt_block.norm2.load_state_dict(orig_block.norm2.state_dict())
            if hasattr(orig_block.norm3, 'weight'):
                trt_block.norm3.load_state_dict(orig_block.norm3.state_dict())
            
            # Copy self-attention weights
            trt_block.self_attn.q.load_state_dict(orig_block.self_attn.q.state_dict())
            trt_block.self_attn.k.load_state_dict(orig_block.self_attn.k.state_dict())
            trt_block.self_attn.v.load_state_dict(orig_block.self_attn.v.state_dict())
            trt_block.self_attn.o.load_state_dict(orig_block.self_attn.o.state_dict())
            trt_block.self_attn.norm_q.load_state_dict(orig_block.self_attn.norm_q.state_dict())
            trt_block.self_attn.norm_k.load_state_dict(orig_block.self_attn.norm_k.state_dict())
            
            # Copy cross-attention weights
            trt_block.cross_attn.q.load_state_dict(orig_block.cross_attn.q.state_dict())
            trt_block.cross_attn.k.load_state_dict(orig_block.cross_attn.k.state_dict())
            trt_block.cross_attn.v.load_state_dict(orig_block.cross_attn.v.state_dict())
            trt_block.cross_attn.o.load_state_dict(orig_block.cross_attn.o.state_dict())
            trt_block.cross_attn.norm_q.load_state_dict(orig_block.cross_attn.norm_q.state_dict())
            trt_block.cross_attn.norm_k.load_state_dict(orig_block.cross_attn.norm_k.state_dict())
            
            # Copy FFN
            trt_block.ffn.load_state_dict(orig_block.ffn.state_dict())
            
            # Copy modulation
            trt_block.modulation.data.copy_(orig_block.modulation.data)
        
        # Copy head weights
        trt_model.head_norm.load_state_dict(original_model.head.norm.state_dict())
        trt_model.head_linear.load_state_dict(original_model.head.head.state_dict())
        trt_model.head_modulation.data.copy_(original_model.head.modulation.data)
        
        # Copy RoPE frequencies
        trt_model.freqs.copy_(original_model.freqs)
        
        if hasattr(original_model, 'img_emb'):
            trt_model.img_emb.load_state_dict(original_model.img_emb.state_dict())
        
        return trt_model
    
    def forward_export(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified forward for ONNX export (no KV cache).
        
        This version runs a single forward pass without caching,
        suitable for exporting the core model to ONNX/TensorRT.
        The KV cache logic will be handled in the C++ runtime wrapper.
        
        Args:
            x: [B, F, C, H, W] noisy latents
            timestep: [B, F] timesteps
            context: [B, text_len, text_dim] text embeddings (raw, not embedded)
            grid_sizes: [B, 3] containing (F, H, W)
        
        Returns:
            output: [B, C, F', H', W'] denoised prediction
        """
        device = x.device
        dtype = x.dtype
        b = x.shape[0]
        
        # Embed context (text) - ensure dtype matches
        context_emb = self.text_embedding(context.to(dtype))  # [B, text_len, dim]
        
        # Embed patches: [B, F, C, H, W] -> [B, dim, F', H', W'] -> [B, L, dim]
        x = self.patch_embedding(x.permute(0, 2, 1, 3, 4))  # [B, C, F, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, L, dim]
        
        seq_lens = torch.tensor([x.shape[1]] * b, device=device, dtype=torch.long)
        
        # Time embeddings - cast sinusoidal output to model dtype
        from causvid.models.wan.wan_base.modules.model import sinusoidal_embedding_1d
        sin_emb = sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(dtype)
        t_emb = self.time_embedding(sin_emb)
        e = self.time_projection(t_emb).unflatten(1, (6, self.dim))
        e = e.unflatten(0, timestep.shape)  # [B, F, 6, dim]
        
        # Process through blocks (no caching for export)
        for block in self.blocks:
            x, _, _, _, _ = block(
                x, e, seq_lens, grid_sizes, self.freqs, context_emb, None,
                None, None, None, None, None, None, None, None
            )
        
        # Head
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e_head = t_emb.unflatten(0, timestep.shape).unsqueeze(2)  # [B, F, 1, dim]
        e_mod = (self.head_modulation.unsqueeze(1) + e_head).chunk(2, dim=2)
        
        x = self.head_norm(x).unflatten(1, (num_frames, frame_seqlen))
        x = self.head_linear(x * (1 + e_mod[1]) + e_mod[0])
        
        # Unpatchify
        x = self._unpatchify(x, grid_sizes)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: torch.Tensor,
        current_start: torch.Tensor,
        current_end: torch.Tensor,
        kv_caches: List[torch.Tensor],
        cross_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with explicit cache tensors.
        
        Args:
            x: [B, F, C, H, W] noisy latents
            timestep: [B, F] timesteps
            context: [B, text_len, dim] text embeddings (already embedded)
            grid_sizes: [B, 3] containing (F, H, W)
            current_start: [B] cache start positions
            current_end: [B] cache end positions
            kv_caches: List of [k, v] tensors for each layer
            cross_caches: List of [k, v] tensors for each layer
        
        Returns:
            output, new_kv_caches, new_cross_caches
        """
        device = x.device
        b = x.shape[0]
        
        # Embed patches: [B, F, C, H, W] -> [B, dim, F', H', W'] -> [B, L, dim]
        x = self.patch_embedding(x.permute(0, 2, 1, 3, 4))  # [B, C, F, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, L, dim]
        
        seq_lens = torch.tensor([x.shape[1]] * b, device=device, dtype=torch.long)
        
        # Time embeddings
        t_emb = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()))
        e = self.time_projection(t_emb).unflatten(1, (6, self.dim))
        e = e.unflatten(0, timestep.shape)  # [B, F, 6, dim]
        
        # Create causal mask (precomputed for TRT)
        # In practice, this would be a static input for the engine
        causal_mask = None  # Using is_causal=True in SDPA instead
        
        new_kv_caches = []
        new_cross_caches = []
        
        # Process through blocks
        for i, block in enumerate(self.blocks):
            kv_k = kv_caches[i * 2] if kv_caches else None
            kv_v = kv_caches[i * 2 + 1] if kv_caches else None
            cross_k = cross_caches[i * 2] if cross_caches else None
            cross_v = cross_caches[i * 2 + 1] if cross_caches else None
            
            x, new_kv_k, new_kv_v, new_cross_k, new_cross_v = block(
                x, e, seq_lens, grid_sizes, self.freqs, context, None,
                causal_mask, kv_k, kv_v, cross_k, cross_v, None,
                current_start, current_end
            )
            
            new_kv_caches.extend([new_kv_k, new_kv_v])
            new_cross_caches.extend([new_cross_k, new_cross_v])
        
        # Head
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e_head = t_emb.unflatten(0, timestep.shape).unsqueeze(2)  # [B, F, 1, dim]
        e_mod = (self.head_modulation.unsqueeze(1) + e_head).chunk(2, dim=2)
        
        x = self.head_norm(x).unflatten(1, (num_frames, frame_seqlen))
        x = self.head_linear(x * (1 + e_mod[1]) + e_mod[0])
        
        # Unpatchify
        x = self._unpatchify(x, grid_sizes)
        
        return x, new_kv_caches, new_cross_caches
    
    def _unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        """Reconstruct video from patches."""
        b = x.shape[0]
        c = self.out_dim
        outputs = []
        
        for i in range(b):
            f, h, w = grid_sizes[i].tolist()
            seq_len = f * h * w
            u = x[i, :seq_len].view(f, h, w, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, f * self.patch_size[0], h * self.patch_size[1], w * self.patch_size[2])
            outputs.append(u)
        
        return torch.stack(outputs)
