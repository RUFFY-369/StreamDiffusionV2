# Copyright 2025-26 StreamDiffusionV2 Authors
"""
TensorRT engine builder for StreamDiffusionV2 models.
"""

import os
import gc
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .utilities import Engine, export_onnx, optimize_onnx, build_engine
from .models import CausalWanModelTRT, VAEEncoderTRT, VAEDecoderTRT, T5EncoderTRT

logger = logging.getLogger(__name__)


class EngineBuilder:
    """
    Orchestrates the full ONNX export → optimize → TensorRT build pipeline.
    
    Usage:
        builder = EngineBuilder(
            engine_dir="./trt_engines",
            model_path="./wan_models/Wan2.1-T2V-1.3B",
        )
        builder.build_all(height=480, width=832, num_frames=21)
    """
    
    def __init__(
        self,
        engine_dir: str,
        model_path: str,
        model_type: str = "T2V-1.3B",
        fp16: bool = True,
        device: str = "cuda",
        force_rebuild: bool = False,
    ):
        self.engine_dir = Path(engine_dir)
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.fp16 = fp16
        self.device = device
        self.force_rebuild = force_rebuild
        
        # Create directory structure
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_dir = self.engine_dir / "onnx"
        self.onnx_dir.mkdir(exist_ok=True)
    
    def _get_engine_path(self, component: str) -> Path:
        return self.engine_dir / f"{component}.engine"
    
    def _get_onnx_path(self, component: str) -> Path:
        return self.onnx_dir / f"{component}.onnx"
    
    def _get_onnx_opt_path(self, component: str) -> Path:
        return self.onnx_dir / f"{component}.opt.onnx"
    
    def _engine_exists(self, component: str) -> bool:
        return self._get_engine_path(component).exists() and not self.force_rebuild
    
    def build_dit(
        self,
        pipeline,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 21,
        skip_onnx_optimize: bool = False,
    ) -> Optional[Engine]:
        """
        Build TensorRT engine for DiT model.
        
        Args:
            pipeline: CausalStreamInferencePipeline containing the model
            batch_size: Optimization batch size
            height: Video height
            width: Video width
            num_frames: Number of frames
        
        Returns:
            Built TensorRT Engine or None if already exists
        """
        component = "dit"
        engine_path = self._get_engine_path(component)
        
        if self._engine_exists(component):
            logger.info(f"DiT engine exists: {engine_path}")
            return None
        
        logger.info("Building DiT TensorRT engine...")
        
        # Get the underlying model and convert to TRT-compatible version
        from .causal_model_trt import CausalWanModelTRTExport
        
        original_model = pipeline.generator.model
        trt_model = CausalWanModelTRTExport.from_pretrained_model(original_model)
        
        # CRITICAL: Delete original model to free GPU memory before export
        # The ONNX export needs a lot of memory for intermediate tensors
        del original_model
        pipeline.generator.model = None  # Prevent access to deleted model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Freed original model memory for ONNX export")
        
        trt_model.eval().to(self.device)
        
        # Convert to fp16 if specified
        if self.fp16:
            trt_model = trt_model.half()
        
        # Create export wrapper that calls forward_export (simpler, no KV cache lists)
        class DiTExportWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x, timestep, context, grid_sizes):
                return self.model.forward_export(x, timestep, context, grid_sizes)
        
        export_model = DiTExportWrapper(trt_model).eval()
        
        # Model definition for profiles (simplified without caches)
        model_def = CausalWanModelTRT(
            model_type=self.model_type,
            fp16=self.fp16,
            device=self.device,
        )
        
        # Get simplified sample inputs (without caches)
        lat_h, lat_w = height // 8, width // 8
        dtype = torch.float16 if self.fp16 else torch.float32
        
        sample_inputs = {
            "x": torch.randn(batch_size, num_frames, 16, lat_h, lat_w, device=self.device, dtype=dtype),
            "timestep": torch.randint(0, 1000, (batch_size, num_frames), device=self.device),
            "context": torch.randn(batch_size, 512, 4096, device=self.device, dtype=dtype),
            "grid_sizes": torch.tensor([[num_frames, lat_h // 2, lat_w // 2]] * batch_size, device=self.device, dtype=torch.long),
        }
        
        # Simplified input/output names
        input_names = ["x", "timestep", "context", "grid_sizes"]
        output_names = ["output"]
        
        # Dynamic axes for variable batch and resolution
        dynamic_axes = {
            "x": {0: "batch", 1: "frames", 3: "height", 4: "width"},
            "timestep": {0: "batch", 1: "frames"},
            "context": {0: "batch"},
            "grid_sizes": {0: "batch"},
            "output": {0: "batch", 2: "frames", 3: "height", 4: "width"},
        }
        
        # Export to ONNX - use dynamo_export for memory efficiency
        onnx_path = str(self._get_onnx_path(component))
        export_onnx(
            export_model,
            onnx_path,
            tuple(sample_inputs.values()),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            use_dynamo=True,  # More memory efficient
        )
        
        # Optimize ONNX (optional - skip to save RAM)
        if skip_onnx_optimize:
            logger.info("Skipping ONNX optimization (--skip_onnx_optimize)")
            onnx_opt_path = onnx_path  # Use unoptimized ONNX
        else:
            onnx_opt_path = str(self._get_onnx_opt_path(component))
            optimize_onnx(onnx_path, onnx_opt_path)
        
        # Build TensorRT engine with simplified profiles
        # Note: grid_sizes is folded as constant during ONNX tracing, so only x, timestep, context are inputs
        # Use max(1, ...) to ensure MIN <= OPT <= MAX even with small frame counts
        opt_frames = max(1, num_frames)
        max_frames = max(2, num_frames * 2)
        input_profile = {
            "x": (
                (1, 1, 16, lat_h // 2, lat_w // 2),  # min
                (batch_size, opt_frames, 16, lat_h, lat_w),  # opt
                (batch_size * 2, max_frames, 16, lat_h * 2, lat_w * 2),  # max
            ),
            "timestep": (
                (1, 1),
                (batch_size, opt_frames),
                (batch_size * 2, max_frames),
            ),
            "context": (
                (1, 512, 4096),
                (batch_size, 512, 4096),
                (batch_size * 2, 512, 4096),
            ),
            # grid_sizes is folded as constant during ONNX export (not a dynamic input)
        }
        
        engine = build_engine(
            str(engine_path),
            onnx_opt_path,
            input_profile,
            fp16=self.fp16,
        )
        
        # Cleanup
        del trt_model, export_model, sample_inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"DiT engine built successfully: {engine_path}")
        return engine
    
    def build_vae_encoder(
        self,
        vae_model,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 21,
    ) -> Optional[Engine]:
        """Build TensorRT engine for VAE encoder."""
        component = "vae_encoder"
        engine_path = self._get_engine_path(component)
        
        if self._engine_exists(component):
            logger.info(f"VAE encoder engine exists: {engine_path}")
            return None
        
        logger.info("Building VAE encoder TensorRT engine...")
        
        # Create encoder-only wrapper
        class VAEEncoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.encoder = vae.model.encoder
                self.conv1 = vae.model.conv1
                
            def forward(self, video):
                x = self.encoder(video)
                mu, _ = self.conv1(x).chunk(2, dim=1)
                return mu
        
        encoder = VAEEncoderWrapper(vae_model).eval().to(self.device)
        
        model_def = VAEEncoderTRT(fp16=self.fp16, device=self.device)
        sample_inputs = model_def.get_sample_input(batch_size, height, width, num_frames)
        
        onnx_path = str(self._get_onnx_path(component))
        export_onnx(
            encoder,
            onnx_path,
            sample_inputs,
            input_names=model_def.get_input_names(),
            output_names=model_def.get_output_names(),
            dynamic_axes=model_def.get_dynamic_axes(),
        )
        
        onnx_opt_path = str(self._get_onnx_opt_path(component))
        optimize_onnx(onnx_path, onnx_opt_path)
        
        input_profile = model_def.get_input_profile(batch_size, height, width, num_frames)
        engine = build_engine(str(engine_path), onnx_opt_path, input_profile, fp16=self.fp16)
        
        del encoder, sample_inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"VAE encoder engine built: {engine_path}")
        return engine
    
    def build_vae_decoder(
        self,
        vae_model,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 21,
    ) -> Optional[Engine]:
        """Build TensorRT engine for VAE decoder."""
        component = "vae_decoder"
        engine_path = self._get_engine_path(component)
        
        if self._engine_exists(component):
            logger.info(f"VAE decoder engine exists: {engine_path}")
            return None
        
        logger.info("Building VAE decoder TensorRT engine...")
        
        class VAEDecoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.decoder = vae.model.decoder
                self.conv2 = vae.model.conv2
                
            def forward(self, latent):
                x = self.conv2(latent)
                return self.decoder(x)
        
        decoder = VAEDecoderWrapper(vae_model).eval().to(self.device)
        
        model_def = VAEDecoderTRT(fp16=self.fp16, device=self.device)
        sample_inputs = model_def.get_sample_input(batch_size, height, width, num_frames)
        
        onnx_path = str(self._get_onnx_path(component))
        export_onnx(
            decoder,
            onnx_path,
            sample_inputs,
            input_names=model_def.get_input_names(),
            output_names=model_def.get_output_names(),
            dynamic_axes=model_def.get_dynamic_axes(),
        )
        
        onnx_opt_path = str(self._get_onnx_opt_path(component))
        optimize_onnx(onnx_path, onnx_opt_path)
        
        input_profile = model_def.get_input_profile(batch_size, height, width, num_frames)
        engine = build_engine(str(engine_path), onnx_opt_path, input_profile, fp16=self.fp16)
        
        del decoder, sample_inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"VAE decoder engine built: {engine_path}")
        return engine
    
    def build_t5_encoder(
        self,
        text_encoder,
        batch_size: int = 1,
    ) -> Optional[Engine]:
        """Build TensorRT engine for T5 text encoder."""
        component = "t5_encoder"
        engine_path = self._get_engine_path(component)
        
        if self._engine_exists(component):
            logger.info(f"T5 encoder engine exists: {engine_path}")
            return None
        
        logger.info("Building T5 encoder TensorRT engine...")
        
        class T5EncoderWrapper(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder.text_encoder
                
            def forward(self, input_ids, attention_mask):
                return self.encoder(input_ids, attention_mask)
        
        encoder = T5EncoderWrapper(text_encoder).eval().to(self.device)
        
        model_def = T5EncoderTRT(fp16=self.fp16, device=self.device)
        sample_inputs = model_def.get_sample_input(batch_size)
        
        onnx_path = str(self._get_onnx_path(component))
        export_onnx(
            encoder,
            onnx_path,
            sample_inputs,
            input_names=model_def.get_input_names(),
            output_names=model_def.get_output_names(),
            dynamic_axes=model_def.get_dynamic_axes(),
        )
        
        onnx_opt_path = str(self._get_onnx_opt_path(component))
        optimize_onnx(onnx_path, onnx_opt_path)
        
        input_profile = model_def.get_input_profile(batch_size)
        engine = build_engine(str(engine_path), onnx_opt_path, input_profile, fp16=self.fp16)
        
        del encoder, sample_inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"T5 encoder engine built: {engine_path}")
        return engine
    
    def build_all(
        self,
        pipeline,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 21,
        skip_onnx_optimize: bool = False,
    ) -> Dict[str, Path]:
        """
        Build all TensorRT engines.
        
        Args:
            pipeline: CausalStreamInferencePipeline
            batch_size: Optimization batch size
            height: Video height
            width: Video width
            num_frames: Number of frames
            skip_onnx_optimize: Skip ONNX optimization step
        
        Returns:
            Dictionary of component name -> engine path
        """
        logger.info(f"Building all TensorRT engines in {self.engine_dir}")
        logger.info(f"Configuration: {batch_size}x{height}x{width}, {num_frames} frames")
        
        engines = {}
        
        # Free up memory by deleting unused pipeline components before DiT export
        # DiT export is the most memory-intensive
        vae = pipeline.vae  # Save reference
        text_encoder = pipeline.text_encoder  # Save reference
        
        # Clear any cached tensors
        gc.collect()
        torch.cuda.empty_cache()
        
        # Build DiT first (most memory intensive)
        self.build_dit(pipeline, batch_size, height, width, num_frames, skip_onnx_optimize)
        engines["dit"] = self._get_engine_path("dit")
        
        # Build VAE
        self.build_vae_encoder(vae, batch_size, height, width, num_frames)
        self.build_vae_decoder(vae, batch_size, height, width, num_frames)
        engines["vae_encoder"] = self._get_engine_path("vae_encoder")
        engines["vae_decoder"] = self._get_engine_path("vae_decoder")
        
        # Build T5
        self.build_t5_encoder(text_encoder, batch_size)
        engines["t5_encoder"] = self._get_engine_path("t5_encoder")
        
        logger.info("All TensorRT engines built successfully!")
        return engines
        return engines
