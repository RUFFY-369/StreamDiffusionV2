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
        "--skip_vae",
        action="store_true",
        default=True,  # Skip VAE by default - TRT VAE is slow due to 3D conv dynamic shapes
        help="Skip VAE engine build (default: True - use PyTorch VAE instead)",
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
        "--skip_dit",
        action="store_true",
        default=False,
        help="Skip DiT engine build (use when DiT already built and valid)",
    )
    parser.add_argument(
        "--vae_only",
        action="store_true",
        default=False,
        help="Build only VAE engines (encoder + decoder)",
    )
    parser.add_argument(
        "--dit_only",
        action="store_true",
        default=False,
        help="Build only DiT engine (recommended for prod)",
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
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Build streaming-optimized DiT engine with KV cache",
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
        engines = {}
        
        # Handle selective building modes
        # Handle selective building modes
        if args.dit_only:
            # Build only DiT (recommended for production)
            logger.info("DiT-only mode - skipping VAE and T5")
            logger.info(f"Building DiT engine (streaming={args.streaming})...")
            
            if args.streaming:
                builder.build_dit_streaming(
                    pipeline, args.batch_size, args.height, args.width,
                    num_frames=1, # Streaming uses 1 frame chunks
                    max_seq_len=50000, # Full capacity (OOM fixed in builder)
                    skip_onnx_optimize=args.skip_onnx_optimize
                )
                engines["dit"] = builder._get_engine_path("dit_streaming")
            else:
                builder.build_dit(
                    pipeline, args.batch_size, args.height, args.width,
                    args.num_frames, args.skip_onnx_optimize
                )
                engines["dit"] = builder._get_engine_path("dit")
            
        elif args.vae_only or args.skip_dit:
            logger.info("Selective build mode - building individual components")
            
            # Build VAE if not skipped
            if (args.build_vae or args.vae_only) and not args.skip_vae:
                logger.info("Building VAE encoder...")
                builder.build_vae_encoder(
                    pipeline.vae, args.batch_size, args.height, args.width, args.num_frames
                )
                engines["vae_encoder"] = builder._get_engine_path("vae_encoder")
                
                logger.info("Building VAE decoder...")
                builder.build_vae_decoder(
                    pipeline.vae, args.batch_size, args.height, args.width, args.num_frames
                )
                engines["vae_decoder"] = builder._get_engine_path("vae_decoder")
            elif args.skip_vae:
                logger.info("Skipping VAE (--skip_vae is True, use PyTorch VAE instead)")
            
            # Build DiT if not skipped
            if not args.skip_dit and not args.vae_only:
                logger.info("Building DiT engine...")
                builder.build_dit(
                    pipeline, args.batch_size, args.height, args.width,
                    args.num_frames, args.skip_onnx_optimize
                )
                engines["dit"] = builder._get_engine_path("dit")
        else:
            # Default build - respects skip_vae flag
            if args.skip_vae:
                logger.info("Skipping VAE build (--skip_vae default, TRT VAE has 3D conv issues)")
            
            engines = builder.build_all(
                pipeline=pipeline,
                batch_size=args.batch_size,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                skip_onnx_optimize=args.skip_onnx_optimize,
                skip_t5=args.skip_t5,
                skip_vae=args.skip_vae,
                streaming=args.streaming,
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
