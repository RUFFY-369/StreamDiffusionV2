# Copyright 2025-26 StreamDiffusionV2 Authors
# TensorRT Acceleration Module
"""
TensorRT acceleration for StreamDiffusionV2 pipeline.

This module provides:
- TensorRT-compatible model variants
- ONNX export utilities  
- TensorRT engine building and runtime
- Drop-in accelerated pipeline

Usage:
    from causvid.acceleration import accelerate_with_tensorrt
    
    pipeline = CausalStreamInferencePipeline(args, device)
    pipeline = accelerate_with_tensorrt(pipeline, engine_dir="./trt_engines")
"""

from .accelerate import accelerate_with_tensorrt

__all__ = [
    "accelerate_with_tensorrt",
]
