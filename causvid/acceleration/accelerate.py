# Copyright 2025-26 StreamDiffusionV2 Authors
"""
Main acceleration entry point for TensorRT.

Provides a single function to accelerate the full StreamDiffusionV2 pipeline.

Note: Requires TensorRT and polygraphy to be installed.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import torch

# TensorRT dependencies are optional at module level
TRT_AVAILABLE = False
try:
    from polygraphy import cuda
    TRT_AVAILABLE = True
except ImportError:
    cuda = None

logger = logging.getLogger(__name__)


def _check_trt_available():
    """Check if TensorRT is available, raise clear error if not."""
    if not TRT_AVAILABLE:
        raise ImportError(
            "TensorRT acceleration requires additional dependencies. "
            "Please install: pip install tensorrt polygraphy onnx onnx-graphsurgeon"
        )


# Lazy imports for TensorRT components
def _get_engine_builder():
    _check_trt_available()
    from .tensorrt.builder import EngineBuilder
    return EngineBuilder


def _get_engines():
    _check_trt_available()
    from .tensorrt.engines import CausalWanModelEngine, WanVAEEngine, T5EncoderEngine
    return CausalWanModelEngine, WanVAEEngine, T5EncoderEngine





def accelerate_with_tensorrt(
    pipeline,
    engine_dir: str,
    model_type: str = "T2V-1.3B",
    batch_size: int = 1,
    height: int = 480,
    width: int = 832,
    num_frames: int = 21,
    use_cuda_graph: bool = True,
    force_rebuild: bool = False,
    fp16: bool = True,
    accelerate_dit: bool = True,
    accelerate_vae: bool = True,
    accelerate_t5: bool = True,
):
    """
    Accelerate CausalStreamInferencePipeline with TensorRT.
    
    This function:
    1. Builds TensorRT engines for DiT, VAE, and T5 (if not already built)
    2. Replaces PyTorch models with TensorRT engine wrappers
    3. Returns the accelerated pipeline
    
    Args:
        pipeline: CausalStreamInferencePipeline instance
        engine_dir: Directory to store/load TensorRT engines
        model_type: Model variant ("T2V-1.3B" or "T2V-14B")
        batch_size: Optimization batch size
        height: Video height in pixels
        width: Video width in pixels
        num_frames: Number of video frames
        use_cuda_graph: Enable CUDA Graphs for reduced kernel launch overhead
        force_rebuild: Force engine rebuild even if cached
        fp16: Use FP16 precision
        accelerate_dit: Accelerate DiT model
        accelerate_vae: Accelerate VAE encoder/decoder
        accelerate_t5: Accelerate T5 text encoder
    
    Returns:
        Accelerated pipeline (same interface, faster inference)
    
    Example:
        from causvid.acceleration import accelerate_with_tensorrt
        
        pipeline = CausalStreamInferencePipeline(args, device)
        pipeline = accelerate_with_tensorrt(
            pipeline,
            engine_dir="./trt_engines",
            height=480,
            width=832,
        )
        
        # Use pipeline as normal - now accelerated!
        output = pipeline.inference_stream(noise, ...)
    """
    # Check TensorRT availability
    _check_trt_available()
    
    # Get lazy imports
    EngineBuilder = _get_engine_builder()
    CausalWanModelEngine, WanVAEEngine, T5EncoderEngine = _get_engines()
    
    logger.info("=" * 60)
    logger.info("TensorRT Acceleration for StreamDiffusionV2")
    logger.info("=" * 60)
    logger.info(f"Engine directory: {engine_dir}")
    logger.info(f"Configuration: {model_type}, {batch_size}x{height}x{width}x{num_frames}")
    logger.info(f"CUDA Graphs: {use_cuda_graph}, FP16: {fp16}")
    
    engine_dir = Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)
    
    device = pipeline.device
    
    # Create CUDA stream for async execution
    stream = cuda.Stream()
    
    # Model-specific settings
    if model_type == "T2V-1.3B":
        num_layers = 30
        num_heads = 12
    elif model_type == "T2V-14B":
        num_layers = 40
        num_heads = 40
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # === Build engines if needed ===
    builder = EngineBuilder(
        engine_dir=str(engine_dir),
        model_path=str(Path(pipeline.generator.model.config._name_or_path).parent 
                      if hasattr(pipeline.generator.model.config, '_name_or_path') 
                      else "./wan_models"),
        model_type=model_type,
        fp16=fp16,
        device=str(device),
        force_rebuild=force_rebuild,
    )
    
    # Build DiT engine
    dit_engine_path = engine_dir / "dit.engine"
    if accelerate_dit:
        if not dit_engine_path.exists() or force_rebuild:
            logger.info("Building DiT TensorRT engine (this may take 10-30 minutes)...")
            builder.build_dit(pipeline, batch_size, height, width, num_frames)
    
    # Build VAE engines
    vae_encoder_path = engine_dir / "vae_encoder.engine"
    vae_decoder_path = engine_dir / "vae_decoder.engine"
    if accelerate_vae:
        if not vae_encoder_path.exists() or force_rebuild:
            logger.info("Building VAE encoder TensorRT engine...")
            builder.build_vae_encoder(pipeline.vae, batch_size, height, width, num_frames)
        if not vae_decoder_path.exists() or force_rebuild:
            logger.info("Building VAE decoder TensorRT engine...")
            builder.build_vae_decoder(pipeline.vae, batch_size, height, width, num_frames)
    
    # Build T5 engine
    t5_engine_path = engine_dir / "t5_encoder.engine"
    if accelerate_t5:
        if not t5_engine_path.exists() or force_rebuild:
            logger.info("Building T5 encoder TensorRT engine...")
            builder.build_t5_encoder(pipeline.text_encoder, batch_size)
    
    # Clean up builder
    del builder
    gc.collect()
    torch.cuda.empty_cache()
    
    # === Load and attach engines ===
    logger.info("Loading TensorRT engines...")
    
    # Create accelerated pipeline wrapper
    accelerated = AcceleratedPipeline(
        original_pipeline=pipeline,
        stream=stream,
        device=device,
    )
    
    # Load DiT engine
    if accelerate_dit and dit_engine_path.exists():
        logger.info("Loading DiT engine...")
        accelerated.dit_engine = CausalWanModelEngine(
            engine_path=str(dit_engine_path),
            stream=stream,
            num_layers=num_layers,
            num_heads=num_heads,
            use_cuda_graph=use_cuda_graph,
            device=str(device),
        )
    
    # Load VAE engines
    if accelerate_vae and vae_encoder_path.exists() and vae_decoder_path.exists():
        logger.info("Loading VAE engines...")
        accelerated.vae_engine = WanVAEEngine(
            encoder_path=str(vae_encoder_path),
            decoder_path=str(vae_decoder_path),
            stream=stream,
            use_cuda_graph=use_cuda_graph,
            device=str(device),
        )
    
    # Load T5 engine
    if accelerate_t5 and t5_engine_path.exists():
        logger.info("Loading T5 engine...")
        accelerated.t5_engine = T5EncoderEngine(
            engine_path=str(t5_engine_path),
            tokenizer=pipeline.text_encoder.tokenizer,
            stream=stream,
            use_cuda_graph=use_cuda_graph,
            device=str(device),
        )
    
    logger.info("TensorRT acceleration enabled!")
    logger.info("=" * 60)
    
    return accelerated


class AcceleratedPipeline:
    """
    TensorRT-accelerated wrapper for CausalStreamInferencePipeline.
    
    Provides same interface as original pipeline but uses TensorRT engines
    for accelerated inference.
    """
    
    def __init__(
        self,
        original_pipeline,
        stream,
        device: str,
    ):
        self.original = original_pipeline
        self.stream = stream
        self.device = device
        
        # Engine slots (populated by accelerate_with_tensorrt)
        self.dit_engine = None  # CausalWanModelEngine
        self.vae_engine = None  # WanVAEEngine  
        self.t5_engine = None   # T5EncoderEngine
        
        # Copy attributes from original
        self.args = original_pipeline.args
        self.num_transformer_blocks = original_pipeline.num_transformer_blocks
        self.num_heads = original_pipeline.num_heads
        self.frame_seq_length = original_pipeline.frame_seq_length
        self.kv_cache_length = original_pipeline.kv_cache_length
        self.denoising_step_list = original_pipeline.denoising_step_list
        self.scheduler = original_pipeline.generator.get_scheduler()
    
    @property
    def generator(self):
        """For compatibility with code that accesses pipeline.generator."""
        return self.original.generator
    
    @property
    def text_encoder(self):
        """Return T5 engine if available, else original."""
        if self.t5_engine is not None:
            return self.t5_engine
        return self.original.text_encoder
    
    @property
    def vae(self):
        """Return VAE engine if available, else original."""
        if self.vae_engine is not None:
            return self.vae_engine
        return self.original.vae
    
    def _use_tensorrt_dit(self) -> bool:
        """Check if DiT TensorRT engine should be used."""
        return self.dit_engine is not None
    
    def prepare(self, *args, **kwargs):
        """Prepare for inference (delegates to original)."""
        return self.original.prepare(*args, **kwargs)
    
    def inference_stream(self, *args, **kwargs):
        """
        Streaming inference using TensorRT.
        
        Falls back to PyTorch if TensorRT engine not available.
        """
        if self._use_tensorrt_dit():
            return self._inference_stream_trt(*args, **kwargs)
        return self.original.inference_stream(*args, **kwargs)
    
    def _inference_stream_trt(
        self,
        noise: torch.Tensor,
        current_start: int,
        current_end: int,
        current_step: int,
    ) -> torch.Tensor:
        """TensorRT-accelerated streaming inference."""
        # Update state
        self.original.hidden_states[1:] = self.original.hidden_states[:-1].clone()
        self.original.hidden_states[0] = noise[0]
        
        self.original.kv_cache_starts[1:] = self.original.kv_cache_starts[:-1].clone()
        self.original.kv_cache_starts[0] = current_start
        
        self.original.kv_cache_ends[1:] = self.original.kv_cache_ends[:-1].clone()
        self.original.kv_cache_ends[0] = current_end
        
        if current_step is not None:
            self.original.timestep[0] = current_step
        
        # Prepare grid sizes
        batch_size = self.original.hidden_states.shape[0]
        num_frames = self.original.hidden_states.shape[1]
        lat_h = self.original.hidden_states.shape[3]
        lat_w = self.original.hidden_states.shape[4]
        
        grid_sizes = torch.tensor(
            [[num_frames, lat_h // 2, lat_w // 2]] * batch_size,
            device=self.device, dtype=torch.long
        )
        
        # Run DiT through TensorRT
        output = self.dit_engine(
            x=self.original.hidden_states,
            timestep=self.original.timestep.unsqueeze(1).expand(-1, num_frames),
            context=self.original.conditional_dict['prompt_embeds'],
            grid_sizes=grid_sizes,
            kv_cache=self.original.kv_cache1,
            cross_cache=self.original.crossattn_cache,
            current_start=self.original.kv_cache_starts,
            current_end=self.original.kv_cache_ends,
        )
        
        self.original.hidden_states = output
        
        # Add noise for next step
        for i in range(len(self.denoising_step_list) - 1):
            self.original.hidden_states[[i]] = self.scheduler.add_noise(
                self.original.hidden_states[[i]],
                torch.randn_like(self.original.hidden_states[[i]]),
                self.denoising_step_list[i + 1] * torch.ones([1], device="cuda", dtype=torch.long)
            )
        
        return self.original.hidden_states
    
    def inference(self, *args, **kwargs):
        """Regular inference (delegates to TRT or original)."""
        if self._use_tensorrt_dit():
            # For now, fall back to original for block-mode inference
            # TODO: Implement TRT version
            pass
        return self.original.inference(*args, **kwargs)
    
    def inference_wo_batch(self, *args, **kwargs):
        """Non-batched inference (delegates to original)."""
        return self.original.inference_wo_batch(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward unknown attributes to original pipeline."""
        return getattr(self.original, name)
