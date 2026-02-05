#!/usr/bin/env python3
"""Diagnose VAE slowness - check GPU usage and execution details."""

import time
import torch
import subprocess
import threading
from polygraphy import cuda

import sys
sys.path.insert(0, ".")

from causvid.acceleration.tensorrt.utilities import Engine

def monitor_gpu(stop_event):
    """Monitor GPU usage in background."""
    while not stop_event.is_set():
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        print(f"  [GPU] {result.stdout.strip()}")
        time.sleep(2)

def test_pytorch_baseline():
    """Test PyTorch VAE encode speed for comparison."""
    print("\n=== PyTorch Baseline (random convolutions) ===")
    
    # Simulate VAE-like operations
    conv = torch.nn.Conv3d(3, 96, kernel_size=3, padding=1).half().cuda()
    x = torch.randn(1, 3, 5, 480, 832, dtype=torch.float16, device="cuda")
    
    # Warmup
    for _ in range(3):
        _ = conv(x)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(5):
        _ = conv(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 5
    
    print(f"PyTorch Conv3d avg: {elapsed*1000:.2f} ms")

def test_trt_vae():
    """Test TRT VAE with GPU monitoring."""
    print("\n=== TensorRT VAE Encoder ===")
    
    engine = Engine("./trt_engines/vae_encoder.engine")
    engine.load()
    engine.activate()
    
    stream = cuda.Stream()
    device = "cuda"
    
    # Very small input
    video = torch.randn(1, 3, 1, 256, 256, dtype=torch.float16, device=device)
    shape_dict = {"video": video.shape}
    
    print(f"Input shape (small): {video.shape}")
    
    engine.allocate_buffers(shape_dict, device)
    engine.tensors["video"].copy_(video)
    
    for name, tensor in engine.tensors.items():
        engine.context.set_tensor_address(name, tensor.data_ptr())
    
    # Start GPU monitor
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_event,))
    monitor_thread.start()
    
    print("Running TRT inference (watch GPU usage)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    success = engine.context.execute_async_v3(stream.ptr)
    stream.synchronize()
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    stop_event.set()
    monitor_thread.join()
    
    print(f"\nResult: {elapsed*1000:.2f} ms (success={success})")
    print(f"Output shape: {engine.tensors['latent'].shape}")

if __name__ == "__main__":
    test_pytorch_baseline()
    test_trt_vae()
