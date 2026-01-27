#!/usr/bin/env python3
# Copyright 2025-26 StreamDiffusionV2 Authors
"""
Test script to verify TensorRT engines work correctly.

Usage:
    python tests/test_trt_engines.py --engine_dir ./trt_engines
"""

import argparse
import logging
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_vae_encoder(engine_dir: str, device: str = "cuda"):
    """Test VAE encoder engine."""
    from polygraphy import cuda
    from causvid.acceleration.tensorrt.utilities import Engine
    
    engine_path = os.path.join(engine_dir, "vae_encoder.engine")
    if not os.path.exists(engine_path):
        logger.warning(f"VAE encoder engine not found: {engine_path}")
        return False
    
    logger.info(f"Testing VAE encoder: {engine_path}")
    
    # Load engine
    engine = Engine(engine_path)
    engine.load()
    engine.activate()
    
    # Create stream
    stream = cuda.Stream()
    
    # Test input: [B, C, T, H, W] video
    # For 480x832 resolution, 21 frames
    batch_size = 1
    height, width = 480, 832
    num_frames = 21
    
    video = torch.randn(batch_size, 3, num_frames, height, width, 
                       dtype=torch.float16, device=device)
    
    # Allocate buffers
    shape_dict = {"video": video.shape}
    engine.allocate_buffers(shape_dict, device)
    
    # Run inference
    logger.info(f"Running VAE encoder with input shape: {video.shape}")
    
    start = time.perf_counter()
    outputs = engine.infer({"video": video}, stream, use_cuda_graph=False)
    stream.synchronize()
    elapsed = time.perf_counter() - start
    
    latent = outputs["latent"]
    logger.info(f"Output shape: {latent.shape}")
    logger.info(f"Inference time: {elapsed*1000:.2f} ms")
    
    # Expected output shape: [B, 16, T//4, H//8, W//8]
    expected_shape = (batch_size, 16, num_frames // 4 + 1, height // 8, width // 8)
    if tuple(latent.shape) == expected_shape:
        logger.info("✅ VAE encoder test PASSED")
        return True
    else:
        logger.error(f"❌ VAE encoder test FAILED: expected {expected_shape}, got {tuple(latent.shape)}")
        return False


def test_vae_decoder(engine_dir: str, device: str = "cuda"):
    """Test VAE decoder engine."""
    from polygraphy import cuda
    from causvid.acceleration.tensorrt.utilities import Engine
    
    engine_path = os.path.join(engine_dir, "vae_decoder.engine")
    if not os.path.exists(engine_path):
        logger.warning(f"VAE decoder engine not found: {engine_path}")
        return False
    
    logger.info(f"Testing VAE decoder: {engine_path}")
    
    # Load engine
    engine = Engine(engine_path)
    engine.load()
    engine.activate()
    
    stream = cuda.Stream()
    
    # Test input: latent [B, 16, T_lat, H_lat, W_lat]
    batch_size = 1
    height, width = 480, 832
    num_frames = 21
    lat_t = num_frames // 4 + 1  # ~6
    lat_h, lat_w = height // 8, width // 8
    
    latent = torch.randn(batch_size, 16, lat_t, lat_h, lat_w,
                        dtype=torch.float16, device=device)
    
    shape_dict = {"latent": latent.shape}
    engine.allocate_buffers(shape_dict, device)
    
    logger.info(f"Running VAE decoder with input shape: {latent.shape}")
    
    start = time.perf_counter()
    outputs = engine.infer({"latent": latent}, stream, use_cuda_graph=False)
    stream.synchronize()
    elapsed = time.perf_counter() - start
    
    video = outputs["video"]
    logger.info(f"Output shape: {video.shape}")
    logger.info(f"Inference time: {elapsed*1000:.2f} ms")
    
    # Expected: [B, 3, T, H, W]
    expected_t = lat_t * 4
    expected_shape = (batch_size, 3, expected_t, height, width)
    
    # Shape match within temporal dimension tolerance
    if video.shape[0] == batch_size and video.shape[1] == 3:
        logger.info("✅ VAE decoder test PASSED")
        return True
    else:
        logger.error(f"❌ VAE decoder test FAILED: got {tuple(video.shape)}")
        return False


def test_dit_engine(engine_dir: str, device: str = "cuda"):
    """Test DiT engine."""
    from polygraphy import cuda
    from causvid.acceleration.tensorrt.utilities import Engine
    
    engine_path = os.path.join(engine_dir, "dit.engine")
    if not os.path.exists(engine_path):
        logger.warning(f"DiT engine not found: {engine_path}")
        return False
    
    logger.info(f"Testing DiT engine: {engine_path}")
    
    # Load engine
    engine = Engine(engine_path)
    engine.load()
    engine.activate()
    
    stream = cuda.Stream()
    
    # Test inputs (matching forward_export signature)
    batch_size = 1
    num_frames = 21
    height, width = 480, 832
    lat_h, lat_w = height // 8, width // 8
    text_len = 512
    text_dim = 4096
    
    # x: [B, F, C, H, W] noisy latents
    x = torch.randn(batch_size, num_frames, 16, lat_h, lat_w,
                   dtype=torch.float16, device=device)
    # timestep: [B, F]
    timestep = torch.randint(0, 1000, (batch_size, num_frames), device=device)
    # context: [B, text_len, text_dim]
    context = torch.randn(batch_size, text_len, text_dim,
                         dtype=torch.float16, device=device)
    # grid_sizes: [B, 3]
    grid_sizes = torch.tensor([[num_frames, lat_h // 2, lat_w // 2]] * batch_size,
                             device=device, dtype=torch.long)
    
    # Check what inputs the engine expects
    logger.info("Engine I/O tensors:")
    for idx in range(engine.engine.num_io_tensors):
        name = engine.engine.get_tensor_name(idx)
        shape = engine.engine.get_tensor_shape(name)
        mode = engine.engine.get_tensor_mode(name)
        logger.info(f"  {name}: {shape} ({mode})")
    
    shape_dict = {
        "x": x.shape,
        "timestep": timestep.shape,
        "context": context.shape,
        "grid_sizes": grid_sizes.shape,
    }
    
    try:
        engine.allocate_buffers(shape_dict, device)
        
        logger.info(f"Running DiT with x shape: {x.shape}")
        
        start = time.perf_counter()
        outputs = engine.infer({
            "x": x,
            "timestep": timestep,
            "context": context,
            "grid_sizes": grid_sizes,
        }, stream, use_cuda_graph=False)
        stream.synchronize()
        elapsed = time.perf_counter() - start
        
        output = outputs.get("output") or list(outputs.values())[0]
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Inference time: {elapsed*1000:.2f} ms")
        logger.info("✅ DiT engine test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ DiT engine test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test TensorRT engines")
    parser.add_argument("--engine_dir", type=str, default="./trt_engines",
                       help="Directory containing TensorRT engines")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "vae_encoder", "vae_decoder", "dit"],
                       help="Which engine to test")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("TensorRT Engine Test Suite")
    logger.info("=" * 60)
    logger.info(f"Engine directory: {args.engine_dir}")
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)
    
    # Check TensorRT availability
    try:
        import tensorrt as trt
        logger.info(f"TensorRT version: {trt.__version__}")
    except ImportError:
        logger.error("TensorRT not installed")
        sys.exit(1)
    
    results = {}
    
    if args.test in ("all", "vae_encoder"):
        results["vae_encoder"] = test_vae_encoder(args.engine_dir, args.device)
    
    if args.test in ("all", "vae_decoder"):
        results["vae_decoder"] = test_vae_decoder(args.engine_dir, args.device)
    
    if args.test in ("all", "dit"):
        results["dit"] = test_dit_engine(args.engine_dir, args.device)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("All tests PASSED!")
    else:
        logger.info("Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
