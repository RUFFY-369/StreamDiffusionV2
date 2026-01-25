# Copyright 2025-26 StreamDiffusionV2 Authors
"""
TensorRT runtime engine wrapper for CausalWanModel (DiT).

Provides drop-in replacement for the PyTorch model with:
- CUDA Graph acceleration
- External KV cache management
- Streaming inference support
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from cuda import cudart

from ..utilities import Engine

logger = logging.getLogger(__name__)


class CausalWanModelEngine:
    """
    TensorRT engine wrapper for CausalWanModel.
    
    Drop-in replacement for CausalWanModel.forward() in inference mode.
    KV cache is managed externally and passed to/from the engine.
    
    Usage:
        engine = CausalWanModelEngine(
            engine_path="./trt_engines/dit.engine",
            stream=cuda_stream,
            use_cuda_graph=True,
        )
        
        # Initialize KV cache
        kv_cache = engine.create_kv_cache(batch_size, max_cache_len)
        
        # Run inference
        output = engine(x, timestep, context, kv_cache, current_start, current_end)
    """
    
    def __init__(
        self,
        engine_path: str,
        stream,
        num_layers: int = 30,
        num_heads: int = 12,
        head_dim: int = 128,
        text_len: int = 512,
        use_cuda_graph: bool = True,
        device: str = "cuda",
    ):
        self.engine_path = engine_path
        self.stream = stream
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.text_len = text_len
        self.use_cuda_graph = use_cuda_graph
        self.device = device
        
        # Load engine
        self.engine = Engine(engine_path)
        self.engine.load()
        self.engine.activate()
        
        # Track buffer shapes
        self._last_shape_hash = None
        
        logger.info(f"Loaded DiT TensorRT engine: {engine_path}")
    
    def create_kv_cache(
        self,
        batch_size: int,
        max_cache_len: int,
        dtype: torch.dtype = torch.float16,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create KV cache structure for all transformer layers.
        
        Returns:
            List of cache dicts, one per layer, each containing:
            - k: [B, cache_len, num_heads, head_dim]
            - v: [B, cache_len, num_heads, head_dim]
            - global_end_index: [B]
            - local_end_index: [B]
        """
        kv_cache = []
        
        for _ in range(self.num_layers):
            kv_cache.append({
                "k": torch.zeros(
                    batch_size, max_cache_len, self.num_heads, self.head_dim,
                    dtype=dtype, device=self.device
                ),
                "v": torch.zeros(
                    batch_size, max_cache_len, self.num_heads, self.head_dim,
                    dtype=dtype, device=self.device
                ),
                "global_end_index": torch.zeros(batch_size, dtype=torch.long, device=self.device),
                "local_end_index": torch.zeros(batch_size, dtype=torch.long, device=self.device),
            })
        
        return kv_cache
    
    def create_cross_cache(
        self,
        batch_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> List[Dict[str, torch.Tensor]]:
        """Create cross-attention cache for text conditioning."""
        cross_cache = []
        
        for _ in range(self.num_layers):
            cross_cache.append({
                "k": torch.zeros(
                    batch_size, self.text_len, self.num_heads, self.head_dim,
                    dtype=dtype, device=self.device
                ),
                "v": torch.zeros(
                    batch_size, self.text_len, self.num_heads, self.head_dim,
                    dtype=dtype, device=self.device
                ),
                "is_init": False,
            })
        
        return cross_cache
    
    def _prepare_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: torch.Tensor,
        kv_cache: List[Dict],
        cross_cache: List[Dict],
        current_start: torch.Tensor,
        current_end: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Prepare input dict for TensorRT engine."""
        inputs = {
            "x": x.contiguous(),
            "timestep": timestep.contiguous(),
            "context": context.contiguous(),
            "grid_sizes": grid_sizes.contiguous(),
            "current_start": current_start.contiguous(),
            "current_end": current_end.contiguous(),
        }
        
        # Add KV cache inputs
        for i in range(self.num_layers):
            inputs[f"kv_cache_k_{i}"] = kv_cache[i]["k"].contiguous()
            inputs[f"kv_cache_v_{i}"] = kv_cache[i]["v"].contiguous()
            inputs[f"cross_cache_k_{i}"] = cross_cache[i]["k"].contiguous()
            inputs[f"cross_cache_v_{i}"] = cross_cache[i]["v"].contiguous()
        
        return inputs
    
    def _extract_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        kv_cache: List[Dict],
        cross_cache: List[Dict],
    ) -> torch.Tensor:
        """Extract output and update caches from engine output."""
        # Update KV cache with new values
        for i in range(self.num_layers):
            kv_cache[i]["k"].copy_(outputs[f"new_kv_cache_k_{i}"])
            kv_cache[i]["v"].copy_(outputs[f"new_kv_cache_v_{i}"])
            cross_cache[i]["k"].copy_(outputs[f"new_cross_cache_k_{i}"])
            cross_cache[i]["v"].copy_(outputs[f"new_cross_cache_v_{i}"])
            cross_cache[i]["is_init"] = True
        
        return outputs["output"]
    
    def _compute_shape_hash(self, x: torch.Tensor) -> int:
        """Compute hash for buffer reuse checking."""
        return hash((x.shape, x.dtype, x.device))
    
    def __call__(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: torch.Tensor,
        kv_cache: List[Dict],
        cross_cache: List[Dict],
        current_start: torch.Tensor,
        current_end: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run DiT inference through TensorRT engine.
        
        Args:
            x: [B, F, C, H, W] noisy latents
            timestep: [B, F] diffusion timesteps
            context: [B, text_len, dim] text embeddings
            grid_sizes: [B, 3] spatial dimensions
            kv_cache: Self-attention KV cache (modified in-place)
            cross_cache: Cross-attention cache (modified in-place)
            current_start: [B] cache start positions
            current_end: [B] cache end positions
        
        Returns:
            Denoised output [B, C, F, H, W]
        """
        # Prepare shape dict
        shape_hash = self._compute_shape_hash(x)
        if shape_hash != self._last_shape_hash:
            # Need to reallocate buffers
            shape_dict = {
                "x": x.shape,
                "timestep": timestep.shape,
                "context": context.shape,
                "grid_sizes": grid_sizes.shape,
                "current_start": current_start.shape,
                "current_end": current_end.shape,
            }
            
            for i in range(self.num_layers):
                shape_dict[f"kv_cache_k_{i}"] = kv_cache[i]["k"].shape
                shape_dict[f"kv_cache_v_{i}"] = kv_cache[i]["v"].shape
                shape_dict[f"cross_cache_k_{i}"] = cross_cache[i]["k"].shape
                shape_dict[f"cross_cache_v_{i}"] = cross_cache[i]["v"].shape
            
            self.engine.allocate_buffers(shape_dict, self.device)
            
            # Reset CUDA graph when shapes change
            if self.use_cuda_graph:
                self.engine.reset_cuda_graph()
            
            self._last_shape_hash = shape_hash
        
        # Prepare inputs
        inputs = self._prepare_inputs(
            x, timestep, context, grid_sizes,
            kv_cache, cross_cache, current_start, current_end
        )
        
        # Run inference
        outputs = self.engine.infer(inputs, self.stream, use_cuda_graph=self.use_cuda_graph)
        
        # Synchronize stream
        self.stream.synchronize()
        
        # Extract outputs and update caches
        output = self._extract_outputs(outputs, kv_cache, cross_cache)
        
        return output
    
    def reset_cuda_graph(self):
        """Reset CUDA graph for recapture."""
        self.engine.reset_cuda_graph()
        self._last_shape_hash = None


class CausalWanModelEngineLite:
    """
    Lightweight engine wrapper for streaming inference.
    
    Optimized for frame-by-frame processing with minimal overhead.
    """
    
    def __init__(
        self,
        engine_path: str,
        stream,
        num_layers: int = 30,
        use_cuda_graph: bool = True,
        device: str = "cuda",
    ):
        self.full_engine = CausalWanModelEngine(
            engine_path=engine_path,
            stream=stream,
            num_layers=num_layers,
            use_cuda_graph=use_cuda_graph,
            device=device,
        )
        
        # Pre-allocated tensors for single-frame inference
        self._cached_grid_sizes = None
        self._cached_current_start = None
        self._cached_current_end = None
    
    def prepare_single_frame(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype = torch.float16,
    ):
        """Pre-allocate tensors for single-frame streaming."""
        device = self.full_engine.device
        lat_h = height // 8
        lat_w = width // 8
        
        self._cached_grid_sizes = torch.tensor(
            [[1, lat_h // 2, lat_w // 2]] * batch_size,
            device=device, dtype=torch.long
        )
        self._cached_current_start = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._cached_current_end = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    def infer_single_frame(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        kv_cache: List[Dict],
        cross_cache: List[Dict],
        frame_index: int,
    ) -> torch.Tensor:
        """Inference for a single frame in streaming mode."""
        frame_seqlen = self._cached_grid_sizes[0, 1].item() * self._cached_grid_sizes[0, 2].item()
        
        # Update position indices
        self._cached_current_start.fill_(frame_index * frame_seqlen)
        self._cached_current_end.fill_((frame_index + 1) * frame_seqlen)
        
        return self.full_engine(
            x, timestep, context, self._cached_grid_sizes,
            kv_cache, cross_cache,
            self._cached_current_start, self._cached_current_end
        )
