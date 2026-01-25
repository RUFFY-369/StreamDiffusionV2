# Copyright 2025-26 StreamDiffusionV2 Authors
"""
TensorRT runtime engine wrapper for WanVAE.

Provides drop-in replacement for VAE encode/decode with:
- Separate encoder and decoder engines
- Streaming encode/decode support
- Feature cache handling
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from ..utilities import Engine

logger = logging.getLogger(__name__)


class WanVAEEngine:
    """
    TensorRT engine wrapper for WanVAE encoder and decoder.
    
    Provides same interface as WanVAEWrapper for drop-in replacement.
    
    Usage:
        vae_engine = WanVAEEngine(
            encoder_path="./trt_engines/vae_encoder.engine",
            decoder_path="./trt_engines/vae_decoder.engine",
            stream=cuda_stream,
        )
        
        latent = vae_engine.encode(video)
        video = vae_engine.decode(latent)
    """
    
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        stream,
        use_cuda_graph: bool = True,
        device: str = "cuda",
    ):
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        self.device = device
        
        # Load encoder engine
        self.encoder_engine = Engine(encoder_path)
        self.encoder_engine.load()
        self.encoder_engine.activate()
        
        # Load decoder engine  
        self.decoder_engine = Engine(decoder_path)
        self.decoder_engine.load()
        self.decoder_engine.activate()
        
        # Normalization parameters (same as WanVAEWrapper)
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        self.z_dim = 16
        
        # Shape tracking for buffer reuse
        self._encoder_last_shape = None
        self._decoder_last_shape = None
        
        # Streaming state
        self.first_encode = True
        self.first_decode = True
        
        logger.info(f"Loaded VAE TensorRT engines")
    
    def _normalize_latent(self, mu: torch.Tensor) -> torch.Tensor:
        """Apply latent normalization."""
        return (mu - self.mean.view(1, self.z_dim, 1, 1, 1)) * (1.0 / self.std.view(1, self.z_dim, 1, 1, 1))
    
    def _denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Remove latent normalization."""
        return z / (1.0 / self.std.view(1, self.z_dim, 1, 1, 1)) + self.mean.view(1, self.z_dim, 1, 1, 1)
    
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space.
        
        Args:
            video: [B, C, T, H, W] video tensor, values in [-1, 1]
        
        Returns:
            latent: [B, z_dim, T//4, H//8, W//8]
        """
        # Check if shapes changed
        if video.shape != self._encoder_last_shape:
            self.encoder_engine.allocate_buffers({"video": video.shape}, self.device)
            if self.use_cuda_graph:
                self.encoder_engine.reset_cuda_graph()
            self._encoder_last_shape = video.shape
        
        # Run encoder
        outputs = self.encoder_engine.infer(
            {"video": video.contiguous()},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        self.stream.synchronize()
        
        # Apply normalization
        latent = self._normalize_latent(outputs["latent"])
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video.
        
        Args:
            latent: [B, z_dim, T_lat, H_lat, W_lat] normalized latent
        
        Returns:
            video: [B, C, T, H, W] video tensor, values in [-1, 1]
        """
        # Denormalize
        z = self._denormalize_latent(latent)
        
        # Check if shapes changed
        if z.shape != self._decoder_last_shape:
            self.decoder_engine.allocate_buffers({"latent": z.shape}, self.device)
            if self.use_cuda_graph:
                self.decoder_engine.reset_cuda_graph()
            self._decoder_last_shape = z.shape
        
        # Run decoder
        outputs = self.decoder_engine.infer(
            {"latent": z.contiguous()},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        self.stream.synchronize()
        
        # Clamp output
        video = outputs["video"].clamp(-1, 1)
        
        return video
    
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to pixel space (compatible with WanVAEWrapper interface).
        
        Args:
            latent: [B, F, C, H, W] latent in batch-frames-first format
        
        Returns:
            video: [B, F, C, H, W] video in same format
        """
        # Convert to channels-first: [B, F, C, H, W] -> [B, C, F, H, W]
        z = latent.permute(0, 2, 1, 3, 4)
        
        # Decode
        video = self.decode(z)
        
        # Convert back: [B, C, F, H, W] -> [B, F, C, H, W]
        return video.permute(0, 2, 1, 3, 4)
    
    def stream_encode(self, video: torch.Tensor, is_scale: bool = True) -> torch.Tensor:
        """
        Streaming encode (compatible with WanVAEWrapper.stream_encode).
        
        Note: Feature caching for causal convolutions is not supported in TRT mode.
        Falls back to full encoding.
        """
        # TensorRT doesn't support internal state for causal conv caching
        # Use full encode
        if video.dim() == 4:
            video = video.unsqueeze(0)
        
        latent = self.encode(video)
        
        return latent
    
    def stream_decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Streaming decode (compatible with WanVAEWrapper.stream_decode_to_pixel).
        
        Note: Feature caching for causal convolutions is not supported in TRT mode.
        Falls back to full decoding.
        """
        # Convert format
        z = latent.permute(0, 2, 1, 3, 4)
        
        # Full decode
        video = self.decode(z)
        
        # Convert back
        return video.permute(0, 2, 1, 3, 4)
    
    def clear_cache(self):
        """Clear streaming state (no-op for TRT, maintained for interface compatibility)."""
        self.first_encode = True
        self.first_decode = True
    
    def reset_cuda_graph(self):
        """Reset CUDA graphs for recapture."""
        self.encoder_engine.reset_cuda_graph()
        self.decoder_engine.reset_cuda_graph()
        self._encoder_last_shape = None
        self._decoder_last_shape = None


class WanVAEEncoderOnlyEngine:
    """Encoder-only VAE engine for latent preprocessing."""
    
    def __init__(
        self,
        encoder_path: str,
        stream,
        use_cuda_graph: bool = True,
        device: str = "cuda",
    ):
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        self.device = device
        
        self.engine = Engine(encoder_path)
        self.engine.load()
        self.engine.activate()
        
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        self.z_dim = 16
        
        self._last_shape = None
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if video.shape != self._last_shape:
            self.engine.allocate_buffers({"video": video.shape}, self.device)
            if self.use_cuda_graph:
                self.engine.reset_cuda_graph()
            self._last_shape = video.shape
        
        outputs = self.engine.infer(
            {"video": video.contiguous()},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        self.stream.synchronize()
        
        mu = outputs["latent"]
        return (mu - self.mean.view(1, self.z_dim, 1, 1, 1)) * (1.0 / self.std.view(1, self.z_dim, 1, 1, 1))
