# Copyright 2025-26 StreamDiffusionV2 Authors
"""
TensorRT utilities for engine building and runtime.

Forked from NVIDIA TensorRT demo utilities with modifications for video diffusion.
"""

import gc
import os
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from cuda import cudart

logger = logging.getLogger(__name__)

# Check for TensorRT availability
try:
    import tensorrt as trt
    import onnx
    import onnx_graphsurgeon as gs
    from polygraphy import cuda
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import (
        CreateConfig,
        Profile,
        engine_from_bytes,
        engine_from_network,
        network_from_onnx_path,
        save_engine,
    )
    from polygraphy.backend.onnx.loader import fold_constants
    TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
except ImportError as e:
    TRT_AVAILABLE = False
    logger.warning(f"TensorRT not available: {e}")

# NumPy to PyTorch dtype mapping
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


def CUASSERT(cuda_ret):
    """CUDA error checking utility."""
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: "
            "https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class Engine:
    """
    TensorRT engine wrapper with CUDA Graph support.
    
    Handles engine loading, buffer allocation, and inference execution.
    """
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None
        self.graph = None
        
        # Buffer reuse optimization
        self._last_shape_dict = None
        self._last_device = None
    
    def __del__(self):
        if not hasattr(self, 'buffers'):
            return
        
        # Clean up CUDA graph
        if hasattr(self, 'cuda_graph_instance') and self.cuda_graph_instance is not None:
            try:
                CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            except:
                pass
        if hasattr(self, 'graph') and self.graph is not None:
            try:
                CUASSERT(cudart.cudaGraphDestroy(self.graph))
            except:
                pass
        
        # Clean up engine resources
        if hasattr(self, 'engine'):
            del self.engine
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'buffers'):
            del self.buffers
        if hasattr(self, 'tensors'):
            del self.tensors
    
    def build(
        self,
        onnx_path: str,
        fp16: bool = True,
        input_profile: Optional[Dict] = None,
        enable_refit: bool = False,
        enable_all_tactics: bool = False,
        timing_cache: Optional[str] = None,
        workspace_size: int = 0,
    ):
        """Build TensorRT engine from ONNX model."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")
        
        logger.info(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3, f"Profile for {name} must have (min, opt, max)"
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])
        
        config_kwargs = {}
        if workspace_size > 0:
            config_kwargs["memory_pool_limits"] = {
                trt.MemoryPoolType.WORKSPACE: workspace_size
            }
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []
        
        engine = engine_from_network(
            network_from_onnx_path(
                onnx_path, 
                flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
            ),
            config=CreateConfig(
                fp16=fp16,
                refittable=enable_refit,
                profiles=[p],
                load_timing_cache=timing_cache,
                **config_kwargs
            ),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)
    
    def load(self):
        """Load serialized TensorRT engine from disk."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")
        
        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))
    
    def activate(self, reuse_device_memory: Optional[int] = None):
        """Create execution context."""
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()
    
    def allocate_buffers(
        self, 
        shape_dict: Optional[Dict[str, Tuple]] = None, 
        device: str = "cuda"
    ):
        """Allocate GPU buffers for engine I/O."""
        # Check if we can reuse existing buffers
        if self._can_reuse_buffers(shape_dict, device):
            return
        
        self.tensors.clear()
        
        # Reset CUDA graph when buffers change
        if self.cuda_graph_instance is not None:
            CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            self.cuda_graph_instance = None
            if hasattr(self, 'graph') and self.graph is not None:
                CUASSERT(cudart.cudaGraphDestroy(self.graph))
                self.graph = None
        
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
            
            dtype_np = trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            
            tensor = torch.empty(
                tuple(shape),
                dtype=numpy_to_torch_dtype_dict[dtype_np]
            ).to(device=device)
            self.tensors[name] = tensor
        
        self._last_shape_dict = shape_dict.copy() if shape_dict else None
        self._last_device = device
    
    def _can_reuse_buffers(
        self, 
        shape_dict: Optional[Dict] = None, 
        device: str = "cuda"
    ) -> bool:
        """Check if existing buffers can be reused."""
        if not self.tensors:
            return False
        if self._last_device != device:
            return False
        if shape_dict is None and self._last_shape_dict is None:
            return True
        if shape_dict is None or self._last_shape_dict is None:
            return False
        if len(shape_dict) != len(self._last_shape_dict):
            return False
        
        for name, new_shape in shape_dict.items():
            cached_shape = self._last_shape_dict.get(name)
            if cached_shape is None:
                return False
            if tuple(cached_shape) != tuple(new_shape):
                return False
        
        return True
    
    def reset_cuda_graph(self):
        """Reset CUDA graph for recapture."""
        if self.cuda_graph_instance is not None:
            CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            self.cuda_graph_instance = None
        if hasattr(self, 'graph') and self.graph is not None:
            CUASSERT(cudart.cudaGraphDestroy(self.graph))
            self.graph = None
    
    def infer(
        self,
        feed_dict: Dict[str, torch.Tensor],
        stream,
        use_cuda_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute inference.
        
        Args:
            feed_dict: Input tensors
            stream: CUDA stream
            use_cuda_graph: Enable CUDA graph for reduced overhead
        
        Returns:
            Output tensors
        """
        # Copy inputs to pre-allocated buffers
        for name, buf in feed_dict.items():
            if name in self.tensors:
                self.tensors[name].copy_(buf)
        
        # Bind tensor addresses
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        
        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                # Fast path: replay captured graph
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))
            else:
                # First run: capture graph
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror:
                    raise ValueError("TensorRT inference failed during graph capture")
                
                CUASSERT(cudart.cudaStreamBeginCapture(
                    stream.ptr,
                    cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
                ))
                self.context.execute_async_v3(stream.ptr)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                self.cuda_graph_instance = CUASSERT(
                    cudart.cudaGraphInstantiate(self.graph, 0)
                )
        else:
            noerror = self.context.execute_async_v3(stream.ptr)
            if not noerror:
                raise ValueError("TensorRT inference failed")
        
        return self.tensors


def export_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    sample_inputs: Tuple,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict] = None,
    opset_version: int = 18,  # Use opset 18 for _upsample_nearest_exact2d support
    use_dynamo: bool = False,
):
    """
    Export PyTorch model to ONNX.
    
    Args:
        model: PyTorch model
        onnx_path: Output path
        sample_inputs: Sample input tensors
        input_names: Names for inputs
        output_names: Names for outputs
        dynamic_axes: Dynamic axis specifications
        opset_version: ONNX opset version
        use_dynamo: Use torch.onnx.dynamo_export (more memory efficient)
    """
    logger.info(f"Exporting model to ONNX: {onnx_path}")
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Force garbage collection before export
    gc.collect()
    torch.cuda.empty_cache()
    
    if use_dynamo:
        # PyTorch 2.x dynamo export - more memory efficient
        try:
            logger.info("Using torch.onnx.dynamo_export (memory efficient)")
            export_output = torch.onnx.dynamo_export(
                model,
                *sample_inputs,
            )
            export_output.save(onnx_path)
            logger.info(f"Dynamo export successful: {onnx_path}")
            return
        except Exception as e:
            logger.warning(f"Dynamo export failed: {e}, falling back to classic export")
    
    # Classic export with memory optimizations
    with torch.inference_mode():
        # Disable gradient tracking completely
        for param in model.parameters():
            param.requires_grad = False
        
        torch.onnx.export(
            model,
            sample_inputs,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    
    # Handle large models (>2GB)
    onnx_model = onnx.load(onnx_path)
    if onnx_model.ByteSize() > 2147483648:
        logger.info("Model exceeds 2GB, using external data format")
        onnx.save_model(
            onnx_model,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
    
    del onnx_model
    gc.collect()
    torch.cuda.empty_cache()


def optimize_onnx(onnx_path: str, onnx_opt_path: str):
    """
    Optimize ONNX model using graph surgery.
    
    Args:
        onnx_path: Input ONNX path
        onnx_opt_path: Output optimized ONNX path
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT/ONNX tools not available")
    
    logger.info(f"Optimizing ONNX: {onnx_path} -> {onnx_opt_path}")
    
    # Check for external data
    onnx_dir = os.path.dirname(onnx_path)
    external_data_files = [f for f in os.listdir(onnx_dir) if f.endswith('.pb')]
    uses_external_data = len(external_data_files) > 0
    
    if uses_external_data:
        onnx_model = onnx.load(onnx_path, load_external_data=True)
    else:
        onnx_model = onnx.load(onnx_path)
    
    # Optimize using polygraphy
    graph = gs.import_onnx(onnx_model)
    graph.cleanup().toposort()
    
    # Fold constants
    onnx_opt = fold_constants(gs.export_onnx(graph), allow_onnxruntime_shape_inference=True)
    graph = gs.import_onnx(onnx_opt)
    graph.cleanup().toposort()
    
    opt_model = gs.export_onnx(graph)
    
    os.makedirs(os.path.dirname(onnx_opt_path), exist_ok=True)
    
    if uses_external_data or opt_model.ByteSize() > 2147483648:
        onnx.save_model(
            opt_model,
            onnx_opt_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
        )
    else:
        onnx.save(opt_model, onnx_opt_path)
    
    del graph, opt_model, onnx_model
    gc.collect()
    torch.cuda.empty_cache()


def build_engine(
    engine_path: str,
    onnx_opt_path: str,
    input_profile: Dict[str, Tuple],
    fp16: bool = True,
    enable_refit: bool = False,
):
    """
    Build TensorRT engine from optimized ONNX.
    
    Args:
        engine_path: Output engine path
        onnx_opt_path: Optimized ONNX path
        input_profile: Dynamic shape profiles {name: (min, opt, max)}
        fp16: Enable FP16 mode
        enable_refit: Enable weight refitting
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available")
    
    # Calculate workspace size
    _, free_mem, _ = cudart.cudaMemGetInfo()
    GiB = 2**30
    if free_mem > 6 * GiB:
        max_workspace_size = free_mem - 4 * GiB
    else:
        max_workspace_size = 0
    
    engine = Engine(engine_path)
    engine.build(
        onnx_opt_path,
        fp16=fp16,
        input_profile=input_profile,
        enable_refit=enable_refit,
        workspace_size=max_workspace_size,
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return engine
