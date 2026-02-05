# Copyright 2025-26 StreamDiffusionV2 Authors
"""
Simplified TensorRT engine wrapper for CausalWanModel (DiT).

This wrapper matches the `forward_export` interface which was used to build
the TensorRT engine - it takes simplified inputs without explicit KV cache
tensors (caches are handled internally or omitted for full-context inference).
"""

import logging
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Lazy imports
_Engine = None
_cuda = None


def _get_engine_class():
    global _Engine
    if _Engine is None:
        from causvid.acceleration.tensorrt.utilities import Engine
        _Engine = Engine
    return _Engine


def _get_cuda():
    global _cuda
    if _cuda is None:
        from polygraphy import cuda
        _cuda = cuda
    return _cuda


class DiTEngineSimple:
    """
    Simplified TensorRT engine wrapper for DiT model.
    
    Matches the `forward_export` interface:
    - Input: x, timestep, context, grid_sizes
    - Output: denoised output tensor
    
    No explicit KV cache management (suitable for full-context inference).
    
    Usage:
        engine = DiTEngineSimple("./trt_engines/dit.engine")
        output = engine(x, timestep, context, grid_sizes)
    """
    
    def __init__(
        self,
        engine_path: str,
        use_cuda_graph: bool = True,
        device: str = "cuda",
    ):
        self.engine_path = engine_path
        self.use_cuda_graph = use_cuda_graph
        self.device = device
        
        # Lazy load on first use
        self._engine = None
        self._stream = None
        self._loaded = False
        self._last_shape_hash = None
        
        logger.info(f"DiTEngineSimple initialized: {engine_path}")
    
    def _ensure_loaded(self):
        """Lazy load the engine on first use."""
        if self._loaded:
            return
        
        Engine = _get_engine_class()
        cuda = _get_cuda()
        
        self._engine = Engine(self.engine_path)
        self._engine.load()
        self._engine.activate()
        self._stream = cuda.Stream()
        self._loaded = True
        
        logger.info(f"Loaded DiT engine: {self.engine_path}")
    
    def _shape_hash(self, x: torch.Tensor) -> int:
        return hash((x.shape, x.dtype))
    
    def __call__(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run DiT inference.
        
        Args:
            x: [B, F, C, H, W] noisy latents (float16)
            timestep: [B, F] diffusion timesteps (long)
            context: [B, text_len, text_dim] text embeddings (float16)
            grid_sizes: [B, 3] containing (F, H', W') patch grid sizes
        
        Returns:
            output: [B, C, F', H', W'] denoised prediction
        """
        self._ensure_loaded()
        
        # Check if we need to reallocate buffers
        shape_hash = self._shape_hash(x)
        if shape_hash != self._last_shape_hash:
            shape_dict = {
                "x": x.shape,
                "timestep": timestep.shape,
                "context": context.shape,
                "grid_sizes": grid_sizes.shape,
            }
            self._engine.allocate_buffers(shape_dict, self.device)
            
            if self.use_cuda_graph:
                self._engine.reset_cuda_graph()
            
            self._last_shape_hash = shape_hash
        
        # Run inference
        outputs = self._engine.infer(
            {
                "x": x.contiguous(),
                "timestep": timestep.contiguous(),
                "context": context.contiguous(),
                "timestep": timestep.contiguous(),
                "context": context.contiguous(),
                "grid_sizes": grid_sizes.contiguous().cpu(), # Force to CPU for Shape Tensor usage
            },
            self._stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        
        # Get output (name may vary)
        output = outputs.get("output")
        if output is None:
            # Try to find the first output tensor
            for name, tensor in outputs.items():
                if tensor.dim() == 5:  # [B, C, F, H, W]
                    output = tensor
                    break
        
        return output
    
    def reset_cuda_graph(self):
        """Reset CUDA graph for recapture."""
        if self._engine is not None:
            self._engine.reset_cuda_graph()
        self._last_shape_hash = None


class DiTEngineStreaming:
    """
    Streaming TensorRT engine wrapper for DiT model.
    
    Supports explicit KV cache inputs/outputs for stateful inference.
    """
    def __init__(
        self,
        engine_path: str,
        use_cuda_graph: bool = True,
        device: str = "cuda",
    ):
        self.engine_path = engine_path
        self.use_cuda_graph = use_cuda_graph
        self.device = device
        
        self._engine = None
        self._stream = None
        self._loaded = False
        self._last_shape_hash = None
        
        logger.info(f"DiTEngineStreaming initialized: {engine_path}")
    
    def _ensure_loaded(self):
        if self._loaded:
            return
            
        Engine = _get_engine_class()
        cuda = _get_cuda()
        
        self._engine = Engine(self.engine_path)
        self._engine.load()
        self._engine.activate()
        self._stream = cuda.Stream()
        self._loaded = True
        logger.info(f"Loaded DiT Streaming engine: {self.engine_path}")
    
    def _shape_hash(self, x: torch.Tensor, kv_caches: tuple) -> int:
        # Tuple of 5 tensor shapes + x
        return hash((x.shape, x.dtype, tuple(k.shape for k in kv_caches)))
    
    def __call__(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        kv_caches: list, # List of 5 tensors
        current_start: torch.Tensor,
        start_frame_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, list]:
        """
        Run Streaming DiT inference with split caches.
        
        Args:
            kv_caches: List of 5 tensors [LayerGroup, 2, B, Seq, H, D]
        Returns:
            output: Denoised latents
            new_kv_caches: Updated KV cache tensors (same list, in-place update)
        """
        self._ensure_loaded()
        
        shape_hash = self._shape_hash(x, kv_caches)
        if shape_hash != self._last_shape_hash:
            shape_dict = {
                "x": x.shape,
                "timestep": timestep.shape,
                "context": context.shape,
                "current_start": current_start.shape,
                "start_frame_idx": start_frame_idx.shape,
            }
            # Add cache shapes
            for i, c in enumerate(kv_caches):
                shape_dict[f"kv_cache_{i}"] = c.shape
            
            # Skip allocation for KV caches (external)
            ext_tensors = []
            for i in range(len(kv_caches)):
                 ext_tensors.append(f"kv_cache_{i}")
                 ext_tensors.append(f"new_kv_cache_{i}")
            
            self._engine.allocate_buffers(
                shape_dict, 
                self.device,
                external_tensors=ext_tensors
            )
            if self.use_cuda_graph:
                self._engine.reset_cuda_graph()
            self._last_shape_hash = shape_hash
        
        # Build inference dict
        feed_dict = {
            "x": x.contiguous(),
            "timestep": timestep.contiguous(),
            "context": context.contiguous(),
            "current_start": current_start.contiguous(),
            "start_frame_idx": start_frame_idx.contiguous(),
        }
        for i, c in enumerate(kv_caches):
             feed_dict[f"kv_cache_{i}"] = c # Zero copy
             feed_dict[f"new_kv_cache_{i}"] = c # In-place
        
        outputs = self._engine.infer(
            feed_dict,
            self._stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        
        return outputs["output"], kv_caches


class VAEEngineSimple:
    """
    Simplified TensorRT engine wrapper for VAE encoder/decoder.
    
    Usage:
        vae = VAEEngineSimple(
            encoder_path="./trt_engines/vae_encoder.engine",
            decoder_path="./trt_engines/vae_decoder.engine",
        )
        latent = vae.encode(video)
        video = vae.decode(latent)
    """
    
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        use_cuda_graph: bool = True,
        device: str = "cuda",
    ):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.use_cuda_graph = use_cuda_graph
        self.device = device
        
        # Lazy load
        self._encoder = None
        self._decoder = None
        self._stream = None
        self._loaded = False
        
        self._encoder_last_shape = None
        self._decoder_last_shape = None
        
        # Normalization constants
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self._mean = torch.tensor(mean, dtype=torch.float32).view(1, 16, 1, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).view(1, 16, 1, 1, 1)
        self.z_dim = 16
        
        logger.info(f"VAEEngineSimple initialized")
    
    def _ensure_loaded(self):
        if self._loaded:
            return
        
        Engine = _get_engine_class()
        cuda = _get_cuda()
        
        self._encoder = Engine(self.encoder_path)
        self._encoder.load()
        self._encoder.activate()
        
        self._decoder = Engine(self.decoder_path)
        self._decoder.load()
        self._decoder.activate()
        
        self._stream = cuda.Stream()
        
        self._mean = self._mean.to(self.device)
        self._std = self._std.to(self.device)
        
        self._loaded = True
        logger.info("Loaded VAE encoder and decoder engines")
    
    def _normalize(self, mu: torch.Tensor) -> torch.Tensor:
        """Apply latent normalization."""
        return (mu - self._mean) / self._std
    
    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        """Remove latent normalization."""
        return z * self._std + self._mean
    
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space.
        
        Args:
            video: [B, C, T, H, W] video tensor, values in [-1, 1]
        
        Returns:
            latent: [B, z_dim, T//4, H//8, W//8] normalized latent
        """
        self._ensure_loaded()
        
        if video.shape != self._encoder_last_shape:
            self._encoder.allocate_buffers({"video": video.shape}, self.device)
            if self.use_cuda_graph:
                self._encoder.reset_cuda_graph()
            self._encoder_last_shape = video.shape
        
        outputs = self._encoder.infer(
            {"video": video.contiguous()},
            self._stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        
        mu = outputs["latent"]
        return self._normalize(mu.float()).to(video.dtype)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video.
        
        Args:
            latent: [B, z_dim, T_lat, H_lat, W_lat] normalized latent
        
        Returns:
            video: [B, C, T, H, W] video tensor, values in [-1, 1]
        """
        self._ensure_loaded()
        
        z = self._denormalize(latent.float()).to(latent.dtype)
        
        if z.shape != self._decoder_last_shape:
            self._decoder.allocate_buffers({"latent": z.shape}, self.device)
            if self.use_cuda_graph:
                self._decoder.reset_cuda_graph()
            self._decoder_last_shape = z.shape
        
        outputs = self._decoder.infer(
            {"latent": z.contiguous()},
            self._stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        
        return outputs["video"].clamp(-1, 1)


class TRTAcceleratedPipeline:
    """
    TensorRT-accelerated inference pipeline.
    
    Combines DiT TensorRT engine with optional VAE acceleration.
    T5 text encoder runs in PyTorch (runs once per prompt, minimal overhead).
    
    VAE can be:
    - TensorRT (if vae_encoder_path/vae_decoder_path provided)
    - PyTorch (if pytorch_vae provided, or as fallback)
    
    Usage:
        # Option 1: TRT DiT + PyTorch VAE (recommended)
        pipeline = TRTAcceleratedPipeline(
            dit_engine_path="./trt_engines/dit.engine",
            pytorch_vae=original_vae,  # PyTorch WanVAE
            text_encoder=original_text_encoder,
        )
        
        # Option 2: Full TRT (if VAE engines work)
        pipeline = TRTAcceleratedPipeline(
            dit_engine_path="./trt_engines/dit.engine",
            vae_encoder_path="./trt_engines/vae_encoder.engine",
            vae_decoder_path="./trt_engines/vae_decoder.engine",
            text_encoder=original_text_encoder,
        )
    """
    
    def __init__(
        self,
        dit_engine_path: str,
        vae_encoder_path: Optional[str] = None,
        vae_decoder_path: Optional[str] = None,
        pytorch_vae=None,
        text_encoder=None,
        use_cuda_graph: bool = True,
        device: str = "cuda",
        streaming: bool = False,
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.streaming = streaming
        
        # Initialize TensorRT DiT engine
        if streaming:
             self.dit = DiTEngineStreaming(
                dit_engine_path,
                use_cuda_graph=use_cuda_graph,
                device=device,
            )
        else:
            self.dit = DiTEngineSimple(
                dit_engine_path,
                use_cuda_graph=use_cuda_graph,
                device=device,
            )
        
        # Initialize VAE (TRT or PyTorch)
        self._use_trt_vae = False
        self.vae = None
        self._pytorch_vae = None
        
        if vae_encoder_path and vae_decoder_path:
            try:
                self.vae = VAEEngineSimple(
                    vae_encoder_path,
                    vae_decoder_path,
                    use_cuda_graph=use_cuda_graph,
                    device=device,
                )
                self._use_trt_vae = True
                logger.info("Using TensorRT VAE")
            except Exception as e:
                logger.warning(f"Failed to load TRT VAE: {e}, falling back to PyTorch")
                self._use_trt_vae = False
        
        if not self._use_trt_vae:
            if pytorch_vae is not None:
                self._pytorch_vae = pytorch_vae
                logger.info("Using PyTorch VAE (recommended - faster than current TRT VAE)")
            else:
                logger.warning("No VAE provided! encode_video/decode_latent will fail.")
        
        # Keep PyTorch text encoder
        self.text_encoder = text_encoder
        
        logger.info("TRTAcceleratedPipeline initialized")
    
    def encode_text(self, prompt: str) -> torch.Tensor:
        """
        Encode text prompt to context embedding.
        
        Uses PyTorch T5 encoder (runs once per prompt).
        """
        if self.text_encoder is None:
            raise ValueError("Text encoder not provided")
        return self.text_encoder.forward(prompt)
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to latent space."""
        if self._use_trt_vae:
            return self.vae.encode(video)
        elif self._pytorch_vae is not None:
            # Use PyTorch VAE
            with torch.no_grad():
                return self._pytorch_vae.encode(video)
        else:
            raise ValueError("No VAE available for encoding")
    
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to video."""
        if self._use_trt_vae:
            return self.vae.decode(latent)
        elif self._pytorch_vae is not None:
            # Use PyTorch VAE
            with torch.no_grad():
                return self._pytorch_vae.decode(latent)
        else:
            raise ValueError("No VAE available for decoding")
    
    def denoise_step(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single denoising step using TensorRT DiT.
        
        Args:
            noisy_latent: [B, F, C, H, W]
            timestep: [B, F]
            context: [B, text_len, text_dim]
            grid_sizes: [B, 3]
        """
        return self.dit(noisy_latent, timestep, context, grid_sizes)

