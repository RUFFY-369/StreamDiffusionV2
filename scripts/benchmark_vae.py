#!/usr/bin/env python3
"""Benchmark VAE engine to diagnose slowness."""

import time
import torch
from polygraphy import cuda

import sys
sys.path.insert(0, ".")

from causvid.acceleration.tensorrt.utilities import Engine

def benchmark_vae_encoder(engine_path: str, warmup_runs: int = 3, bench_runs: int = 5):
    """Benchmark VAE encoder with warmup."""
    print(f"Loading engine: {engine_path}")
    
    engine = Engine(engine_path)
    engine.load()
    engine.activate()
    
    stream = cuda.Stream()
    device = "cuda"
    
    # Test with smaller input first
    batch_size = 1
    num_frames = 5  # Smaller than 21
    height, width = 480, 832
    
    video = torch.randn(batch_size, 3, num_frames, height, width,
                       dtype=torch.float16, device=device)
    
    shape_dict = {"video": video.shape}
    
    print(f"\nInput shape: {video.shape}")
    print(f"Running {warmup_runs} warmup + {bench_runs} benchmark runs...")
    
    # Allocate buffers once
    engine.allocate_buffers(shape_dict, device)
    
    times = []
    for i in range(warmup_runs + bench_runs):
        engine.tensors["video"].copy_(video)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Get tensor addresses
        for name, tensor in engine.tensors.items():
            engine.context.set_tensor_address(name, tensor.data_ptr())
        
        success = engine.context.execute_async_v3(stream.ptr)
        stream.synchronize()
        torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        run_type = "WARMUP" if i < warmup_runs else "BENCH"
        print(f"  [{run_type}] Run {i+1}: {elapsed*1000:.2f} ms (success={success})")
    
    bench_times = times[warmup_runs:]
    avg = sum(bench_times) / len(bench_times)
    print(f"\nBenchmark average: {avg*1000:.2f} ms")
    print(f"Min: {min(bench_times)*1000:.2f} ms, Max: {max(bench_times)*1000:.2f} ms")

if __name__ == "__main__":
    benchmark_vae_encoder("./trt_engines/vae_encoder.engine")
