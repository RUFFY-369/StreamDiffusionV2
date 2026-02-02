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
        
        # Use native TensorRT loading instead of polygraphy for TRT 10 compatibility
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {self.engine_path}")
            
        # Register binding names, dtypes, and indices
        self.binding_names = []
        self.tensor_dtypes = {}
        self.binding_indices = {}  # Map name -> index manually
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            self.binding_names.append(name)
            self.binding_indices[name] = idx
            dtype_np = trt.nptype(self.engine.get_tensor_dtype(name))
            self.tensor_dtypes[name] = numpy_to_torch_dtype_dict.get(dtype_np, torch.float32)
    
    def activate(self, reuse_device_memory: Optional[int] = None):
        """
        Create execution context for dynamic shape engine.
        
        Uses manual memory management to bypass TRT 10's automatic allocation
        which can fail with very large memory requests.
        """
        # For TRT 10+ with dynamic shapes, use manual memory management
        # to completely bypass TRT's internal allocation
        # Debug logging of bindings
        logger.info("Engine Bindings and Types:")
        for name, dtype in self.tensor_dtypes.items():
            logger.info(f"  {name}: {dtype}")
            
        self.context = self.engine.create_execution_context_without_device_memory()
        
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
    
    def allocate_buffers(
        self, 
        shape_dict: Optional[Dict[str, Tuple]] = None, 
        device: str = "cuda",
        external_tensors: Optional[List[str]] = None
    ):
        """
        Allocate GPU buffers for engine I/O.
        
        Args:
            shape_dict: Input shapes
            device: Device name
            external_tensors: List of tensor names to skip allocation for (Zero-Copy)
        """
        # Check if we can reuse existing buffers (skip for simplicity if external_tensors used)
        if self._can_reuse_buffers(shape_dict, device) and not external_tensors:
            return
        
        self.tensors.clear()
        
        external_tensors = set(external_tensors or [])
        
        # Reset CUDA graph when buffers change
        if self.cuda_graph_instance is not None:
            CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            self.cuda_graph_instance = None
            if hasattr(self, 'graph') and self.graph is not None:
                CUASSERT(cudart.cudaGraphDestroy(self.graph))
                self.graph = None
        
        # CRITICAL: For dynamic shape engines, set ALL input shapes FIRST
        # before querying output shapes or allocating memory
        if shape_dict:
            # Use binding_indices if available, else try fallbacks (iteration is safer)
            for name, shape in shape_dict.items():
                if name in self.binding_indices:
                    idx = self.binding_indices[name]
                    mode = self.engine.get_tensor_mode(name)
                    if mode == trt.TensorIOMode.INPUT:
                        # Convert to tuple if it's a torch.Size
                        if hasattr(shape, '__iter__'):
                            shape = tuple(shape)
                        self.context.set_input_shape(name, shape)
        
        # Allocate device memory for the context AFTER setting input shapes
        # For TRT 10 with dynamic shapes, device_memory_size_v2 can return garbage
        # We need to query the actual required size after setting input shapes
        try:
            # TRT 10: Use update_device_memory_size_for_shapes or infer from output shapes
            # First, try to get actual memory requirement from context
            try:
                if hasattr(self.context, 'update_device_memory_size_for_shapes'):
                    self.context.update_device_memory_size_for_shapes()
            except Exception as e:
                # Handle shape calculation overflow (e.g., large KV caches)
                logger.warning(f"TensorRT shape memory calculation failed (likely overflow): {e}")
            
            # Get device memory size based on actual input shapes
            device_mem_size = self.engine.get_device_memory_size_for_profile(0)
        except (AttributeError, TypeError):
            # Fallback: Calculate based on I/O tensor sizes
            device_mem_size = 0
        
        # For dynamic shapes, we can also estimate memory as a reasonable multiple
        # of I/O buffer sizes. VAE encoder needs ~3GB, so use 4GB as safe default.
        if device_mem_size <= 0 or device_mem_size > 100 * (1024**3):  # >100GB is garbage
            device_mem_size = 4 * 1024 * 1024 * 1024  # 4 GB default workspace
            logger.warning(f"Using default device memory size: {device_mem_size / (1024**3):.1f} GB")
        
        if device_mem_size > 0:
            # Free old memory if size changed
            if hasattr(self, '_device_memory') and self._device_memory is not None:
                if hasattr(self, '_device_memory_size') and self._device_memory_size != device_mem_size:
                    cudart.cudaFree(self._device_memory)
                    self._device_memory = None
            
            # Allocate CUDA memory for TRT's internal use
            if not hasattr(self, '_device_memory') or self._device_memory is None:
                err, self._device_memory = cudart.cudaMalloc(device_mem_size)
                if err != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"Failed to allocate {device_mem_size} bytes for TRT device memory")
                self._device_memory_size = device_mem_size
            self.context.device_memory = self._device_memory
        
        # Now allocate tensors for all I/O
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            # Allocation loop continues...
        
        # Iterate bindings
        for name in self.binding_names:
            if name in external_tensors:
                continue # Skip allocation for external tensors
                
            idx = self.binding_indices[name]
            dtype = self.tensor_dtypes[name]
            
            # Get shape from context (for dynamic output shapes)
            shape = self.context.get_tensor_shape(name)
            
            # Allocate
            vol = 1
            for d in shape:
                vol *= d
            
            # Handle negative dims (error)
            if vol < 0:
                logger.warning(f"Warning: Tensor {name} has dynamic shape {shape}, cannot allocate yet.")
                continue
                
            # Allocate buffer
            try:
                self.tensors[name] = torch.empty(
                    tuple(shape), 
                    dtype=dtype, 
                    device=device
                )
            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM allocating {name} ({vol} elements)")
                raise
        
        self._last_shape_dict = shape_dict.copy() if shape_dict else None
        self._last_device = device

    def _can_reuse_buffers(self, shape_dict: Optional[Dict] = None, device: str = "cuda") -> bool:
        """Check if existing buffers can be reused."""
        if not self.tensors:
            return False
        if getattr(self, '_last_device', None) != device:
            return False
        last_shape = getattr(self, '_last_shape_dict', None)
        if shape_dict is None and last_shape is None:
            return True
        if shape_dict is None or last_shape is None:
            return False
        if len(shape_dict) != len(last_shape):
            return False
        
        for name, new_shape in shape_dict.items():
            cached_shape = last_shape.get(name)
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
        stream: torch.cuda.Stream,
        use_cuda_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference (Zero-Copy compatible).
        """
        # 1. Bind provided tensors (Zero-Copy)
        for name, tensor in feed_dict.items():
            if name in self.binding_indices:
                # FIX: Removed the logic that forced 'current_start' to CPU.
                # In streaming DiT, 'current_start' is an execution tensor (cache index)
                # and must reside on the GPU. Forcing it to CPU causes Illegal Memory Access.
                
                if self.engine.is_shape_inference_io(name):
                    # Shape Tensor MUST be on Host (CPU)
                    if tensor.device.type != 'cpu':
                        tensor = tensor.cpu()
                    
                    # CRITICAL: Keep reference to CPU tensor to prevent GC!
                    # Otherwise data_ptr becomes invalid before execution.
                    tensor = tensor.contiguous()
                    feed_dict[name] = tensor 
                    
                    self.context.set_tensor_address(name, tensor.data_ptr())
                    
                    # Also update dimensions if dynamic
                    self.context.set_input_shape(name, tensor.shape)
                else:
                    # Execution Tensor: MUST be on Device (GPU)
                    # If input is on CPU but needs to be on GPU, warn or error?
                    # For now, we assume user provides correct device, or we could strict check.
                    # tensor = tensor.cuda() # Uncomment if auto-move is desired
                    
                    self.context.set_tensor_address(name, tensor.data_ptr())
        
        # 2. Bind internal tensors (for missing outputs/scratch)
        for name, tensor in self.tensors.items():
            if name not in feed_dict:
                self.context.set_tensor_address(name, tensor.data_ptr())
        
        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))
            else:
                # Capture
                self.context.execute_async_v3(stream.ptr)
                
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
            self.context.execute_async_v3(stream.ptr)
        
        return {**self.tensors, **feed_dict}


def export_onnx(
    model: torch.nn.Module,
    file_path: Union[str, Path],
    sample_inputs: Tuple[Any, ...],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    opset_version: int = 17,
    use_dynamo: bool = False,
):
    """
    Export PyTorch model to ONNX.
    Wrapper around torch.onnx.export with support for large models.
    """
    file_path = str(file_path)
    logger.info(f"Exporting ONNX to {file_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Handle dynamo export if requested, but fallback to script for reliability
    if use_dynamo:
        logger.info("Note: use_dynamo=True requested, but using standard torch.onnx.export for stability.")
    
    # Use standard export
    torch.onnx.export(
        model,
        sample_inputs,
        file_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        verbose=False
    )
    logger.info("ONNX export successful")


def optimize_onnx(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
):
    """
    Optimize ONNX model using Polygraphy (constant folding).
    """
    input_path = str(input_path)
    output_path = str(output_path)
    logger.info(f"Optimizing ONNX: {input_path} -> {output_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input ONNX file not found: {input_path}")
        
    onx = onnx.load(input_path)
    onx = fold_constants(onx)
    
    onnx.save(onx, output_path)
    logger.info("ONNX optimization complete")


def build_engine(
    engine_path: Union[str, Path],
    onnx_path: Union[str, Path],
    input_profile: Dict[str, Tuple[Tuple, Tuple, Tuple]],
    fp16: bool = True,
    timing_cache: Optional[str] = None,
    workspace_size: int = 0,
) -> Engine:
    """
    Build TensorRT engine from ONNX model.
    """
    engine_path = str(engine_path)
    onnx_path = str(onnx_path)
    
    logger.info(f"Building TensorRT Engine: {engine_path}")
    
    # Setup builder config
    p = Profile()
    for name, (min_shape, opt_shape, max_shape) in input_profile.items():
        p.add(name, min=min_shape, opt=opt_shape, max=max_shape)
    
    config_kwargs = {}
    if workspace_size > 0:
        config_kwargs["memory_pool_limits"] = {
            trt.MemoryPoolType.WORKSPACE: workspace_size
        }
        
    config = CreateConfig(
        fp16=fp16,
        profiles=[p],
        load_timing_cache=timing_cache,
        **config_kwargs
    )
    
    # Build
    engine = engine_from_network(
        network_from_onnx_path(onnx_path),
        config=config,
        save_timing_cache=timing_cache
    )
    
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
        
    save_engine(engine, path=engine_path)
    
    # Return Engine wrapper
    return Engine(engine_path)
