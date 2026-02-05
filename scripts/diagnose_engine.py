#!/usr/bin/env python3
"""Diagnose TensorRT engine optimization profiles."""

import sys
import tensorrt as trt

def inspect_engine(engine_path):
    """Print optimization profile info from engine."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {engine_path}")
    print(f"{'='*60}")
    
    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("ERROR: Failed to load engine")
        return
    
    print(f"TensorRT version: {trt.__version__}")
    print(f"Num I/O tensors: {engine.num_io_tensors}")
    print(f"Num optimization profiles: {engine.num_optimization_profiles}")
    
    # Check device memory requirement
    try:
        mem_size = engine.device_memory_size_v2
        print(f"Device memory size (v2): {mem_size:,} bytes ({mem_size / (1024**3):.2f} GB)")
        if mem_size > 100 * (1024**3):  # More than 100 GB is suspicious
            print(f"⚠️  WARNING: Device memory size is suspiciously large!")
    except:
        try:
            mem_size = engine.device_memory_size
            print(f"Device memory size: {mem_size:,} bytes ({mem_size / (1024**3):.2f} GB)")
        except:
            print("Could not get device memory size")
    
    # Print each tensor
    print(f"\nTensor Info:")
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"  [{idx}] {name}: shape={shape}, dtype={dtype}, mode={mode}")
    
    # Print optimization profile details
    print(f"\nOptimization Profile Details:")
    for profile_idx in range(engine.num_optimization_profiles):
        print(f"\n  Profile {profile_idx}:")
        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            mode = engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                try:
                    # TRT 10 API: get_tensor_profile_shape returns (min, opt, max) tuple
                    shapes = engine.get_tensor_profile_shape(name, profile_idx)
                    if isinstance(shapes, tuple) and len(shapes) == 3:
                        min_shape, opt_shape, max_shape = shapes
                    else:
                        # Fallback for different API
                        min_shape = opt_shape = max_shape = shapes
                    print(f"    {name}:")
                    print(f"      MIN: {min_shape}")
                    print(f"      OPT: {opt_shape}")
                    print(f"      MAX: {max_shape}")
                    
                    # Check for suspicious sizes
                    for dim_idx, (mn, op, mx) in enumerate(zip(min_shape, opt_shape, max_shape)):
                        if mx > 100000:
                            print(f"      ⚠️  WARNING: MAX dim[{dim_idx}] = {mx} is very large!")
                except Exception as e:
                    print(f"    {name}: Error getting profile - {e}")

if __name__ == "__main__":
    engine_paths = sys.argv[1:] or ["./trt_engines/vae_encoder.engine"]
    for path in engine_paths:
        inspect_engine(path)

