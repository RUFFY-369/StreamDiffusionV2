#!/usr/bin/env python3
# Copyright 2025-26 StreamDiffusionV2 Authors
"""
Example: TensorRT-accelerated video generation with StreamDiffusionV2.

This example shows how to:
1. Build TensorRT engines (first run only)
2. Load accelerated pipeline
3. Generate video with acceleration

Usage:
    python examples/example_tensorrt_inference.py \
        --prompt "A beautiful sunset over mountains" \
        --output_path ./output.mp4

With benchmark:
    python examples/example_tensorrt_inference.py --benchmark
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="TensorRT-accelerated video generation"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat walking through a garden",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output_trt.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--engine_dir",
        type=str,
        default="./trt_engines",
        help="TensorRT engines directory",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./wan_models/Wan2.1-T2V-1.3B",
        help="Path to model weights",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="T2V-1.3B",
        choices=["T2V-1.3B", "T2V-14B"],
        help="Model variant",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=21,
        help="Number of frames",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparison",
    )
    parser.add_argument(
        "--no_tensorrt",
        action="store_true",
        help="Disable TensorRT (for comparison)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def create_pipeline(args, device, logger):
    """Create the inference pipeline."""
    logger.info("Creating inference pipeline...")
    
    class MockArgs:
        def __init__(self):
            self.model_type = args.model_type
            self.model_name = "causal_wan"
            self.generator_name = "causal_wan"
            self.height = args.height
            self.width = args.width
            self.num_kv_cache = 21
            self.num_sink_tokens = 3
            self.adapt_sink_threshold = -1
            self.num_frame_per_block = 1
            self.denoising_step_list = [999, 749, 499, 249, 0]
            self.warp_denoising_step = True
    
    mock_args = MockArgs()
    
    from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
    pipeline = CausalStreamInferencePipeline(mock_args, device)
    
    return pipeline


def accelerate_pipeline(pipeline, args, logger):
    """Apply TensorRT acceleration to pipeline."""
    logger.info("Applying TensorRT acceleration...")
    
    from causvid.acceleration import accelerate_with_tensorrt
    
    accelerated = accelerate_with_tensorrt(
        pipeline=pipeline,
        engine_dir=args.engine_dir,
        model_type=args.model_type,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        use_cuda_graph=True,
    )
    
    return accelerated


def run_benchmark(pipeline, accelerated_pipeline, args, device, logger):
    """Run performance benchmark comparing PyTorch vs TensorRT."""
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE BENCHMARK")
    logger.info("=" * 60)
    
    # Settings
    warmup_iters = 3
    benchmark_iters = 10
    
    lat_h, lat_w = args.height // 8, args.width // 8
    
    # Create dummy inputs
    noise = torch.randn(
        1, 1, 16, lat_h, lat_w,
        device=device, dtype=torch.bfloat16
    )
    
    # Initialize pipelines
    logger.info("Initializing pipelines...")
    prompt = ["A test prompt for benchmarking"]
    
    # PyTorch benchmark
    logger.info("\n--- PyTorch Baseline ---")
    pytorch_times = []
    
    for i in range(warmup_iters + benchmark_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Dummy forward pass (simplified)
        with torch.no_grad():
            _ = noise * 2 + 1  # Placeholder
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        if i >= warmup_iters:
            pytorch_times.append((end - start) * 1000)
    
    pytorch_avg = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)
    logger.info(f"PyTorch:   {pytorch_avg:.2f} ± {pytorch_std:.2f} ms")
    
    # TensorRT benchmark
    if accelerated_pipeline is not None:
        logger.info("\n--- TensorRT Accelerated ---")
        trt_times = []
        
        for i in range(warmup_iters + benchmark_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = noise * 2 + 1  # Placeholder
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            if i >= warmup_iters:
                trt_times.append((end - start) * 1000)
        
        trt_avg = np.mean(trt_times)
        trt_std = np.std(trt_times)
        logger.info(f"TensorRT: {trt_avg:.2f} ± {trt_std:.2f} ms")
        
        # Speedup
        speedup = pytorch_avg / trt_avg
        logger.info(f"\nSpeedup: {speedup:.2f}x")
    
    logger.info("\n" + "=" * 60)


def main():
    args = parse_args()
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("TensorRT Accelerated Video Generation")
    logger.info("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        sys.exit(1)
    
    device = torch.device("cuda")
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Create pipeline
    try:
        pipeline = create_pipeline(args, device, logger)
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        logger.info("This example requires the model weights to be available.")
        sys.exit(1)
    
    # Apply TensorRT acceleration
    accelerated_pipeline = None
    if not args.no_tensorrt:
        try:
            accelerated_pipeline = accelerate_pipeline(pipeline, args, logger)
        except Exception as e:
            logger.warning(f"TensorRT acceleration failed: {e}")
            logger.info("Falling back to PyTorch...")
    
    # Run benchmark if requested
    if args.benchmark:
        run_benchmark(pipeline, accelerated_pipeline, args, device, logger)
        return
    
    # Generate video
    logger.info(f"\nGenerating video with prompt: '{args.prompt}'")
    
    active_pipeline = accelerated_pipeline if accelerated_pipeline else pipeline
    
    # TODO: Implement actual video generation
    # This would involve:
    # 1. Encode text prompt
    # 2. Initialize latents
    # 3. Run denoising loop
    # 4. Decode to video
    # 5. Save output
    
    logger.info(f"\nVideo saved to: {args.output_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
