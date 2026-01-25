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
        trt_model.eval().to(self.device)
        
        # Model definition for profiles
        model_def = CausalWanModelTRT(
            model_type=self.model_type,
            fp16=self.fp16,
            device=self.device,
        )
        
        # Get sample inputs
        sample_inputs = model_def.get_sample_input(batch_size, height, width, num_frames)
        
        # Export to ONNX
        onnx_path = str(self._get_onnx_path(component))
        export_onnx(
            trt_model,
            onnx_path,
            tuple(sample_inputs.values()),
            input_names=model_def.get_input_names(),
            output_names=model_def.get_output_names(),
            dynamic_axes=model_def.get_dynamic_axes(),
        )
        
        # Optimize ONNX
        onnx_opt_path = str(self._get_onnx_opt_path(component))
        optimize_onnx(onnx_path, onnx_opt_path)
        
        # Build TensorRT engine
        input_profile = model_def.get_input_profile(batch_size, height, width, num_frames)
        engine = build_engine(
            str(engine_path),
            onnx_opt_path,
            input_profile,
            fp16=self.fp16,
        )
        
        # Cleanup
        del trt_model, sample_inputs
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
    ) -> Dict[str, Path]:
        """
        Build all TensorRT engines.
        
        Args:
            pipeline: CausalStreamInferencePipeline
            batch_size: Optimization batch size
            height: Video height
            width: Video width
            num_frames: Number of frames
        
        Returns:
            Dictionary of component name -> engine path
        """
        logger.info(f"Building all TensorRT engines in {self.engine_dir}")
        logger.info(f"Configuration: {batch_size}x{height}x{width}, {num_frames} frames")
        
        engines = {}
        
        # Build DiT
        self.build_dit(pipeline, batch_size, height, width, num_frames)
        engines["dit"] = self._get_engine_path("dit")
        
        # Build VAE
        self.build_vae_encoder(pipeline.vae, batch_size, height, width, num_frames)
        self.build_vae_decoder(pipeline.vae, batch_size, height, width, num_frames)
        engines["vae_encoder"] = self._get_engine_path("vae_encoder")
        engines["vae_decoder"] = self._get_engine_path("vae_decoder")
        
        # Build T5
        self.build_t5_encoder(pipeline.text_encoder, batch_size)
        engines["t5_encoder"] = self._get_engine_path("t5_encoder")
        
        logger.info("All TensorRT engines built successfully!")
        return engines
