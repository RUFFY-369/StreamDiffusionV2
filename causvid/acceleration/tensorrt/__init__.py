# Copyright 2025-26 StreamDiffusionV2 Authors
# TensorRT Module
"""
TensorRT utilities for model compilation and runtime.

Note: TensorRT dependencies (tensorrt, polygraphy, onnx) are optional.
The module can be imported without them, but building engines requires them.
"""

import logging

logger = logging.getLogger(__name__)

# Lazy imports - only load when accessed
_lazy_imports = {}

def _get_utilities():
    if 'utilities' not in _lazy_imports:
        from . import utilities
        _lazy_imports['utilities'] = utilities
    return _lazy_imports['utilities']

def _get_builder():
    if 'builder' not in _lazy_imports:
        from . import builder
        _lazy_imports['builder'] = builder
    return _lazy_imports['builder']

def _get_models():
    if 'models' not in _lazy_imports:
        from . import models
        _lazy_imports['models'] = models
    return _lazy_imports['models']

def _get_causal_model_trt():
    if 'causal_model_trt' not in _lazy_imports:
        from . import causal_model_trt
        _lazy_imports['causal_model_trt'] = causal_model_trt
    return _lazy_imports['causal_model_trt']

# Re-export with lazy loading
def __getattr__(name):
    """Lazy attribute loading for TensorRT dependencies."""
    if name == "Engine":
        return _get_utilities().Engine
    elif name == "export_onnx":
        return _get_utilities().export_onnx
    elif name == "optimize_onnx":
        return _get_utilities().optimize_onnx
    elif name == "build_engine":
        return _get_utilities().build_engine
    elif name == "EngineBuilder":
        return _get_builder().EngineBuilder
    elif name == "CausalWanModelTRT":
        return _get_models().CausalWanModelTRT
    elif name == "VAEEncoderTRT":
        return _get_models().VAEEncoderTRT
    elif name == "VAEDecoderTRT":
        return _get_models().VAEDecoderTRT
    elif name == "T5EncoderTRT":
        return _get_models().T5EncoderTRT
    # Inference wrappers
    elif name == "CausalWanModelTRTInference":
        return _get_causal_model_trt().CausalWanModelTRTInference
    elif name == "load_trt_model":
        return _get_causal_model_trt().load_trt_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Utilities
    "Engine",
    "export_onnx", 
    "optimize_onnx",
    "build_engine",
    "EngineBuilder",
    # Model definitions
    "CausalWanModelTRT",
    "VAEEncoderTRT",
    "VAEDecoderTRT", 
    "T5EncoderTRT",
    # Inference wrappers
    "CausalWanModelTRTInference",
    "load_trt_model",
]


