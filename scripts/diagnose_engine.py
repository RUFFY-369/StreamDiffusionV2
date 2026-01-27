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
    
    print(f"Num I/O tensors: {engine.num_io_tensors}")
    print(f"Num optimization profiles: {engine.num_optimization_profiles}")
    
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
                    min_shape = engine.get_tensor_profile_shape(name, profile_idx, opt=trt.ProfileSelector.MIN)
                    opt_shape = engine.get_tensor_profile_shape(name, profile_idx, opt=trt.ProfileSelector.OPT)
                    max_shape = engine.get_tensor_profile_shape(name, profile_idx, opt=trt.ProfileSelector.MAX)
                    print(f"    {name}:")
                    print(f"      MIN: {min_shape}")
                    print(f"      OPT: {opt_shape}")
                    print(f"      MAX: {max_shape}")
                except Exception as e:
                    print(f"    {name}: Error getting profile - {e}")

if __name__ == "__main__":
    engine_paths = sys.argv[1:] or ["./trt_engines/vae_encoder.engine"]
    for path in engine_paths:
        inspect_engine(path)
