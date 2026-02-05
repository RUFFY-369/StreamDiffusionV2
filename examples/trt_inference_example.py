#!/usr/bin/env python3
# Copyright 2025-26 StreamDiffusionV2 Authors
"""
Example: TensorRT-accelerated inference with StreamDiffusionV2.

This example shows how to use the pre-built TensorRT engines
for fast video generation inference.

Usage:
    python examples/trt_inference_example.py --engine_dir ./trt_engines

Requirements:
    - Built TensorRT engines (dit.engine, vae_encoder.engine, vae_decoder.engine)
    - TensorRT and related packages installed
"""

import argparse
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def benchmark_dit(dit_engine, num_runs=10, warmup_runs=3):
    """Benchmark DiT inference speed."""
    batch_size = 1
    num_frames = 21
    height, width = 480, 832
    lat_h, lat_w = height // 8, width // 8
    
    x = torch.randn(batch_size, num_frames, 16, lat_h, lat_w,
                   dtype=torch.float16, device="cuda")
    timestep = torch.randint(0, 1000, (batch_size, num_frames), device="cuda")
    context = torch.randn(batch_size, 512, 4096, dtype=torch.float16, device="cuda")
    grid_sizes = torch.tensor([[num_frames, lat_h // 2, lat_w // 2]] * batch_size,
                             device="cuda", dtype=torch.long)
    
    # Warmup
    logger.info(f"Warming up with {warmup_runs} runs...")
    for _ in range(warmup_runs):
        _ = dit_engine(x, timestep, context, grid_sizes)
        torch.cuda.synchronize()
    
    # Benchmark
    logger.info(f"Benchmarking with {num_runs} runs...")
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = dit_engine(x, timestep, context, grid_sizes)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms
    
    avg = np.mean(times)
    std = np.std(times)
    p50 = np.percentile(times, 50)
    p99 = np.percentile(times, 99)
    
    return {
        "avg_ms": avg,
        "std_ms": std,
        "p50_ms": p50,
        "p99_ms": p99,
    }


def benchmark_vae(vae_engine, num_runs=10, warmup_runs=3):
    """Benchmark VAE encode/decode speed."""
    batch_size = 1
    num_frames = 21
    height, width = 480, 832
    
    video = torch.randn(batch_size, 3, num_frames, height, width,
                       dtype=torch.float16, device="cuda")
    
    # Warmup encode
    logger.info("Warming up VAE encoder...")
    for _ in range(warmup_runs):
        _ = vae_engine.encode(video)
        torch.cuda.synchronize()
    
    # Benchmark encode
    encode_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        latent = vae_engine.encode(video)
        torch.cuda.synchronize()
        encode_times.append((time.perf_counter() - start) * 1000)
    
    # Warmup decode
    logger.info("Warming up VAE decoder...")
    for _ in range(warmup_runs):
        _ = vae_engine.decode(latent)
        torch.cuda.synchronize()
    
    # Benchmark decode
    decode_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = vae_engine.decode(latent)
        torch.cuda.synchronize()
        decode_times.append((time.perf_counter() - start) * 1000)
    
    return {
        "encode_avg_ms": np.mean(encode_times),
        "encode_std_ms": np.std(encode_times),
        "decode_avg_ms": np.mean(decode_times),
        "decode_std_ms": np.std(decode_times),
    }


def main():
    parser = argparse.ArgumentParser(description="TensorRT Inference Example")
    parser.add_argument("--engine_dir", type=str, default="./trt_engines",
                       help="Directory containing TensorRT engines")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmarks")
    parser.add_argument("--num_runs", type=int, default=10,
                       help="Number of benchmark runs")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("TensorRT Accelerated Inference Example")
    logger.info("=" * 60)
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Engine directory: {args.engine_dir}")
    
    # Check engines exist
    dit_path = os.path.join(args.engine_dir, "dit.engine")
    vae_enc_path = os.path.join(args.engine_dir, "vae_encoder.engine")
    vae_dec_path = os.path.join(args.engine_dir, "vae_decoder.engine")
    
    engines_found = []
    for name, path in [("DiT", dit_path), ("VAE Encoder", vae_enc_path), ("VAE Decoder", vae_dec_path)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            engines_found.append(name)
            logger.info(f"  ✓ {name}: {path} ({size_mb:.1f} MB)")
        else:
            logger.warning(f"  ✗ {name}: NOT FOUND at {path}")
    
    if not engines_found:
        logger.error("No engines found!")
        sys.exit(1)
    
    logger.info("=" * 60)
    
    # Import and initialize engines
    from causvid.acceleration.tensorrt.engines.simple_engines import (
        DiTEngineSimple, VAEEngineSimple
    )
    
    results = {}
    
    # Test DiT
    if "DiT" in engines_found:
        logger.info("\n--- DiT Engine ---")
        dit_engine = DiTEngineSimple(dit_path, use_cuda_graph=True)
        
        if args.benchmark:
            stats = benchmark_dit(dit_engine, num_runs=args.num_runs)
            results["dit"] = stats
            logger.info(f"DiT: {stats['avg_ms']:.2f} ± {stats['std_ms']:.2f} ms")
            logger.info(f"  P50: {stats['p50_ms']:.2f} ms, P99: {stats['p99_ms']:.2f} ms")
    
    # Test VAE
    if "VAE Encoder" in engines_found and "VAE Decoder" in engines_found:
        logger.info("\n--- VAE Engine ---")
        vae_engine = VAEEngineSimple(vae_enc_path, vae_dec_path, use_cuda_graph=True)
        
        if args.benchmark:
            stats = benchmark_vae(vae_engine, num_runs=args.num_runs)
            results["vae"] = stats
            logger.info(f"VAE Encode: {stats['encode_avg_ms']:.2f} ± {stats['encode_std_ms']:.2f} ms")
            logger.info(f"VAE Decode: {stats['decode_avg_ms']:.2f} ± {stats['decode_std_ms']:.2f} ms")
    
    # Summary
    if args.benchmark and results:
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)
        
        total_per_step = 0
        if "dit" in results:
            total_per_step += results["dit"]["avg_ms"]
            logger.info(f"DiT per step:       {results['dit']['avg_ms']:>8.2f} ms")
        
        if "vae" in results:
            logger.info(f"VAE encode:         {results['vae']['encode_avg_ms']:>8.2f} ms")
            logger.info(f"VAE decode:         {results['vae']['decode_avg_ms']:>8.2f} ms")
        
        if "dit" in results:
            # Estimate for N denoising steps
            for n_steps in [2, 4, 8]:
                total_time = n_steps * results["dit"]["avg_ms"]
                if "vae" in results:
                    total_time += results["vae"]["encode_avg_ms"]
                    total_time += results["vae"]["decode_avg_ms"]
                fps = 1000.0 / total_time
                logger.info(f"Est. {n_steps}-step total: {total_time:>8.2f} ms ({fps:.1f} FPS)")
        
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
