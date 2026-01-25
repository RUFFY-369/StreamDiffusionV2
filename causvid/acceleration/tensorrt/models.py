# Copyright 2025-26 StreamDiffusionV2 Authors
"""
TensorRT model definitions for ONNX export.

Defines input/output profiles for dynamic shape support.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn


class BaseModelTRT:
    """Base class for TensorRT model definitions."""
    
    def __init__(
        self,
        fp16: bool = True,
        device: str = "cuda",
        max_batch_size: int = 4,
        min_batch_size: int = 1,
    ):
        self.name = "BaseModel"
        self.fp16 = fp16
        self.device = device
        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
    
    def get_input_names(self) -> List[str]:
        raise NotImplementedError
    
    def get_output_names(self) -> List[str]:
        raise NotImplementedError
    
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        raise NotImplementedError
    
    def get_input_profile(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> Dict[str, Tuple]:
        """Get TensorRT optimization profile (min, opt, max) for each input."""
        raise NotImplementedError
    
    def get_sample_input(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> Tuple[torch.Tensor, ...]:
        """Get sample inputs for ONNX export."""
        raise NotImplementedError


class CausalWanModelTRT(BaseModelTRT):
    """
    TensorRT model definition for CausalWanModel (DiT backbone).
    
    Handles:
    - Dynamic batch size
    - Dynamic spatial resolution
    - Dynamic frame count
    - KV cache as explicit I/O tensors
    """
    
    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 12,
        num_layers: int = 30,
        text_len: int = 512,
        text_dim: int = 4096,
        freq_dim: int = 256,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_dim: int = 16,
        out_dim: int = 16,
        max_kv_cache_len: int = 32760,  # ~21 frames
        fp16: bool = True,
        device: str = "cuda",
        max_batch_size: int = 4,
        min_batch_size: int = 1,
        model_type: str = "T2V-1.3B",
    ):
        super().__init__(fp16, device, max_batch_size, min_batch_size)
        self.name = "CausalWanModel"
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.text_len = text_len
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_kv_cache_len = max_kv_cache_len
        self.head_dim = dim // num_heads
        self.model_type = model_type
        
        # Model-specific settings
        if model_type == "T2V-1.3B":
            self.dim = 1536
            self.num_heads = 12
            self.num_layers = 30
        elif model_type == "T2V-14B":
            self.dim = 5120
            self.num_heads = 40
            self.num_layers = 40
    
    def get_input_names(self) -> List[str]:
        names = [
            "x",               # Noisy latents [B, F, C, H, W]
            "timestep",        # Diffusion timestep [B, F]
            "context",         # Text embeddings [B, text_len, text_dim]
            "grid_sizes",      # [B, 3] for (F, H, W)
            "current_start",   # KV cache start [B]
            "current_end",     # KV cache end [B]
        ]
        
        # Add KV cache inputs for each layer
        for i in range(self.num_layers):
            names.extend([
                f"kv_cache_k_{i}",  # [B, cache_len, num_heads, head_dim]
                f"kv_cache_v_{i}",
                f"cross_cache_k_{i}",
                f"cross_cache_v_{i}",
            ])
        
        return names
    
    def get_output_names(self) -> List[str]:
        names = ["output"]  # Denoised latents [B, F, C, H, W]
        
        # Output updated KV caches
        for i in range(self.num_layers):
            names.extend([
                f"new_kv_cache_k_{i}",
                f"new_kv_cache_v_{i}",
                f"new_cross_cache_k_{i}",
                f"new_cross_cache_v_{i}",
            ])
        
        return names
    
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        axes = {
            "x": {0: "B", 1: "F"},
            "timestep": {0: "B", 1: "F"},
            "context": {0: "B"},
            "grid_sizes": {0: "B"},
            "current_start": {0: "B"},
            "current_end": {0: "B"},
            "output": {0: "B", 1: "F"},
        }
        
        for i in range(self.num_layers):
            axes[f"kv_cache_k_{i}"] = {0: "B", 1: "cache_len"}
            axes[f"kv_cache_v_{i}"] = {0: "B", 1: "cache_len"}
            axes[f"cross_cache_k_{i}"] = {0: "B"}
            axes[f"cross_cache_v_{i}"] = {0: "B"}
            axes[f"new_kv_cache_k_{i}"] = {0: "B", 1: "cache_len"}
            axes[f"new_kv_cache_v_{i}"] = {0: "B", 1: "cache_len"}
            axes[f"new_cross_cache_k_{i}"] = {0: "B"}
            axes[f"new_cross_cache_v_{i}"] = {0: "B"}
        
        return axes
    
    def get_input_profile(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> Dict[str, Tuple]:
        """Get optimization profiles for TensorRT."""
        # Latent dimensions
        lat_h = height // 8
        lat_w = width // 8
        frame_seqlen = (lat_h // self.patch_size[1]) * (lat_w // self.patch_size[2])
        
        min_b, opt_b, max_b = 1, batch_size, self.max_batch
        min_f, opt_f, max_f = 1, num_frames, 21  # Standard max frames
        
        profile = {
            "x": [
                (min_b, min_f, self.in_dim, lat_h, lat_w),
                (opt_b, opt_f, self.in_dim, lat_h, lat_w),
                (max_b, max_f, self.in_dim, lat_h, lat_w),
            ],
            "timestep": [
                (min_b, min_f),
                (opt_b, opt_f),
                (max_b, max_f),
            ],
            "context": [
                (min_b, self.text_len, self.text_dim),
                (opt_b, self.text_len, self.text_dim),
                (max_b, self.text_len, self.text_dim),
            ],
            "grid_sizes": [
                (min_b, 3),
                (opt_b, 3),
                (max_b, 3),
            ],
            "current_start": [(min_b,), (opt_b,), (max_b,)],
            "current_end": [(min_b,), (opt_b,), (max_b,)],
        }
        
        # KV cache profiles
        min_cache = frame_seqlen
        opt_cache = frame_seqlen * num_frames
        max_cache = self.max_kv_cache_len
        
        for i in range(self.num_layers):
            profile[f"kv_cache_k_{i}"] = [
                (min_b, min_cache, self.num_heads, self.head_dim),
                (opt_b, opt_cache, self.num_heads, self.head_dim),
                (max_b, max_cache, self.num_heads, self.head_dim),
            ]
            profile[f"kv_cache_v_{i}"] = profile[f"kv_cache_k_{i}"]
            
            profile[f"cross_cache_k_{i}"] = [
                (min_b, self.text_len, self.num_heads, self.head_dim),
                (opt_b, self.text_len, self.num_heads, self.head_dim),
                (max_b, self.text_len, self.num_heads, self.head_dim),
            ]
            profile[f"cross_cache_v_{i}"] = profile[f"cross_cache_k_{i}"]
        
        return profile
    
    def get_sample_input(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> Dict[str, torch.Tensor]:
        """Get sample inputs for ONNX export."""
        dtype = torch.float16 if self.fp16 else torch.float32
        device = self.device
        
        lat_h = height // 8
        lat_w = width // 8
        frame_seqlen = (lat_h // self.patch_size[1]) * (lat_w // self.patch_size[2])
        cache_len = frame_seqlen * num_frames
        
        inputs = {
            "x": torch.randn(batch_size, num_frames, self.in_dim, lat_h, lat_w, 
                           dtype=dtype, device=device),
            "timestep": torch.randint(0, 1000, (batch_size, num_frames), 
                                     device=device).long(),
            "context": torch.randn(batch_size, self.text_len, self.text_dim,
                                  dtype=dtype, device=device),
            "grid_sizes": torch.tensor([[num_frames, lat_h // self.patch_size[1], 
                                        lat_w // self.patch_size[2]]] * batch_size,
                                      device=device, dtype=torch.long),
            "current_start": torch.zeros(batch_size, dtype=torch.long, device=device),
            "current_end": torch.ones(batch_size, dtype=torch.long, device=device) * cache_len,
        }
        
        # Initialize KV caches
        for i in range(self.num_layers):
            inputs[f"kv_cache_k_{i}"] = torch.zeros(
                batch_size, cache_len, self.num_heads, self.head_dim,
                dtype=dtype, device=device
            )
            inputs[f"kv_cache_v_{i}"] = torch.zeros(
                batch_size, cache_len, self.num_heads, self.head_dim,
                dtype=dtype, device=device
            )
            inputs[f"cross_cache_k_{i}"] = torch.zeros(
                batch_size, self.text_len, self.num_heads, self.head_dim,
                dtype=dtype, device=device
            )
            inputs[f"cross_cache_v_{i}"] = torch.zeros(
                batch_size, self.text_len, self.num_heads, self.head_dim,
                dtype=dtype, device=device
            )
        
        return inputs


class VAEEncoderTRT(BaseModelTRT):
    """TensorRT model definition for WanVAE encoder."""
    
    def __init__(
        self,
        z_dim: int = 16,
        fp16: bool = True,
        device: str = "cuda",
        max_batch_size: int = 4,
        min_batch_size: int = 1,
    ):
        super().__init__(fp16, device, max_batch_size, min_batch_size)
        self.name = "VAEEncoder"
        self.z_dim = z_dim
    
    def get_input_names(self) -> List[str]:
        return ["video"]  # [B, C, T, H, W]
    
    def get_output_names(self) -> List[str]:
        return ["latent"]  # [B, z_dim, T//4, H//8, W//8]
    
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "video": {0: "B", 2: "T", 3: "H", 4: "W"},
            "latent": {0: "B", 2: "T_lat", 3: "H_lat", 4: "W_lat"},
        }
    
    def get_input_profile(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> Dict[str, Tuple]:
        min_b, opt_b, max_b = 1, batch_size, self.max_batch
        min_t, opt_t, max_t = 1, num_frames, 81  # Max frames for video
        min_h, opt_h, max_h = 256, height, 1024
        min_w, opt_w, max_w = 256, width, 1024
        
        return {
            "video": [
                (min_b, 3, min_t, min_h, min_w),
                (opt_b, 3, opt_t, opt_h, opt_w),
                (max_b, 3, max_t, max_h, max_w),
            ]
        }
    
    def get_sample_input(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> Tuple[torch.Tensor]:
        dtype = torch.float16 if self.fp16 else torch.float32
        video = torch.randn(batch_size, 3, num_frames, height, width,
                           dtype=dtype, device=self.device)
        return (video,)


class VAEDecoderTRT(BaseModelTRT):
    """TensorRT model definition for WanVAE decoder."""
    
    def __init__(
        self,
        z_dim: int = 16,
        fp16: bool = True,
        device: str = "cuda",
        max_batch_size: int = 4,
        min_batch_size: int = 1,
    ):
        super().__init__(fp16, device, max_batch_size, min_batch_size)
        self.name = "VAEDecoder"
        self.z_dim = z_dim
    
    def get_input_names(self) -> List[str]:
        return ["latent"]
    
    def get_output_names(self) -> List[str]:
        return ["video"]
    
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "latent": {0: "B", 2: "T_lat", 3: "H_lat", 4: "W_lat"},
            "video": {0: "B", 2: "T", 3: "H", 4: "W"},
        }
    
    def get_input_profile(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> Dict[str, Tuple]:
        lat_h, lat_w = height // 8, width // 8
        lat_t = (num_frames + 3) // 4  # Temporal compression
        
        min_b, opt_b, max_b = 1, batch_size, self.max_batch
        
        return {
            "latent": [
                (min_b, self.z_dim, 1, 32, 32),
                (opt_b, self.z_dim, lat_t, lat_h, lat_w),
                (max_b, self.z_dim, 21, 128, 128),
            ]
        }
    
    def get_sample_input(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> Tuple[torch.Tensor]:
        dtype = torch.float16 if self.fp16 else torch.float32
        lat_h, lat_w = height // 8, width // 8
        lat_t = (num_frames + 3) // 4
        
        latent = torch.randn(batch_size, self.z_dim, lat_t, lat_h, lat_w,
                            dtype=dtype, device=self.device)
        return (latent,)


class T5EncoderTRT(BaseModelTRT):
    """TensorRT model definition for T5 text encoder."""
    
    def __init__(
        self,
        text_len: int = 512,
        hidden_dim: int = 4096,
        fp16: bool = True,
        device: str = "cuda",
        max_batch_size: int = 4,
        min_batch_size: int = 1,
    ):
        super().__init__(fp16, device, max_batch_size, min_batch_size)
        self.name = "T5Encoder"
        self.text_len = text_len
        self.hidden_dim = hidden_dim
    
    def get_input_names(self) -> List[str]:
        return ["input_ids", "attention_mask"]
    
    def get_output_names(self) -> List[str]:
        return ["text_embeddings"]
    
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "B"},
            "attention_mask": {0: "B"},
            "text_embeddings": {0: "B"},
        }
    
    def get_input_profile(
        self,
        batch_size: int,
        **kwargs,
    ) -> Dict[str, Tuple]:
        min_b, opt_b, max_b = 1, batch_size, self.max_batch
        
        return {
            "input_ids": [
                (min_b, self.text_len),
                (opt_b, self.text_len),
                (max_b, self.text_len),
            ],
            "attention_mask": [
                (min_b, self.text_len),
                (opt_b, self.text_len),
                (max_b, self.text_len),
            ],
        }
    
    def get_sample_input(
        self,
        batch_size: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.randint(0, 32000, (batch_size, self.text_len),
                                 dtype=torch.long, device=self.device)
        attention_mask = torch.ones(batch_size, self.text_len,
                                   dtype=torch.long, device=self.device)
        return (input_ids, attention_mask)
