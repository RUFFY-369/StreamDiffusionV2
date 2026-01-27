#!/usr/bin/env python3
# Copyright 2025-26 StreamDiffusionV2 Authors
"""
Build TensorRT engines for StreamDiffusionV2.

Usage:
    python scripts/build_tensorrt_engines.py \
        --model_path ./wan_models/Wan2.1-T2V-1.3B \
        --output_dir ./trt_engines \
        --height 480 --width 832

For 14B model:
    python scripts/build_tensorrt_engines.py \
        --model_path ./wan_models/Wan2.1-T2V-14B \
        --model_type T2V-14B \
        --output_dir ./trt_engines_14b
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engines for StreamDiffusionV2"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./wan_models/Wan2.1-T2V-1.3B",
        help="Path to Wan model directory",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="T2V-1.3B",
        choices=["T2V-1.3B", "T2V-14B"],
        help="Model variant",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trt_engines",
        help="Output directory for TensorRT engines",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width in pixels",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=21,
        help="Number of video frames",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Optimization batch size",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 precision",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuild even if engines exist",
    )
    parser.add_argument(
        "--build_dit",
        action="store_true",
        default=True,
        help="Build DiT engine",
    )
    parser.add_argument(
        "--build_vae",
        action="store_true",
        default=True,
        help="Build VAE engines",
    )
    parser.add_argument(
        "--build_t5",
        action="store_true",
        default=True,
        help="Build T5 encoder engine",
    )
    parser.add_argument(
        "--skip_t5",
        action="store_true",
        default=False,
        help="Skip T5 engine build (recommended for prod - T5 gives <1%% speedup)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--skip_onnx_optimize",
        action="store_true",
        help="Skip ONNX optimization step (saves RAM, TensorRT handles optimization)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("TensorRT Engine Builder for StreamDiffusionV2")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Resolution: {args.height}x{args.width}")
    logger.info(f"Frames: {args.num_frames}")
    logger.info("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. TensorRT requires CUDA.")
        sys.exit(1)
    
    device = torch.device("cuda")
    logger.info(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # Check TensorRT availability
    try:
        import tensorrt as trt
        logger.info(f"TensorRT version: {trt.__version__}")
    except ImportError:
        logger.error("TensorRT is not installed. Please install tensorrt package.")
        sys.exit(1)
    
    # Create mock pipeline args for initialization
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
    
    # Initialize pipeline (for model access)
    logger.info("Loading models...")
    from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
    
    try:
        pipeline = CausalStreamInferencePipeline(mock_args, device)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.info("Proceeding with standalone engine building...")
        pipeline = None
    
    # Build engines
    from causvid.acceleration.tensorrt.builder import EngineBuilder
    
    builder = EngineBuilder(
        engine_dir=args.output_dir,
        model_path=args.model_path,
        model_type=args.model_type,
        fp16=args.fp16,
        device=str(device),
        force_rebuild=args.force_rebuild,
    )
    
    if pipeline is not None:
        # Use full pipeline for building
        engines = builder.build_all(
            pipeline=pipeline,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            skip_onnx_optimize=args.skip_onnx_optimize,
            skip_t5=args.skip_t5,
        )
        
        logger.info("\nBuilt engines:")
        for name, path in engines.items():
            logger.info(f"  {name}: {path}")
    else:
        logger.warning("Pipeline not available. Manual engine building required.")
    
    logger.info("\n" + "=" * 60)
    logger.info("Engine building complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
