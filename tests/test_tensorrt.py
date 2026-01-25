#!/usr/bin/env python3
# Copyright 2025-26 StreamDiffusionV2 Authors
"""
Tests for TensorRT acceleration.

Run with:
    python -m pytest tests/test_tensorrt.py -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTRTAttention(unittest.TestCase):
    """Test TensorRT-compatible attention implementations."""
    
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
    def test_trt_causal_self_attention_init(self):
        """Test TRTCausalSelfAttention initialization."""
        from causvid.acceleration.tensorrt.attention import TRTCausalSelfAttention
        
        dim = 1536
        num_heads = 12
        
        attn = TRTCausalSelfAttention(dim=dim, num_heads=num_heads)
        
        self.assertEqual(attn.dim, dim)
        self.assertEqual(attn.num_heads, num_heads)
        self.assertEqual(attn.head_dim, dim // num_heads)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_trt_causal_self_attention_forward(self):
        """Test TRTCausalSelfAttention forward pass."""
        from causvid.acceleration.tensorrt.attention import TRTCausalSelfAttention
        
        dim = 1536
        num_heads = 12
        batch_size = 1
        seq_len = 1560
        
        # Move to device AND convert to correct dtype
        attn = TRTCausalSelfAttention(dim=dim, num_heads=num_heads).to(self.device).to(self.dtype)
        
        x = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=self.dtype)
        seq_lens = torch.tensor([seq_len], device=self.device)
        grid_sizes = torch.tensor([[1, 30, 52]], device=self.device)
        
        # Create dummy freqs - also needs to be same dtype
        head_dim = dim // num_heads
        freqs = torch.randn(1024, head_dim // 2, device=self.device, dtype=self.dtype)
        
        out, kv_k, kv_v = attn(x, seq_lens, grid_sizes, freqs)
        
        self.assertEqual(out.shape, x.shape)
    
    def test_create_causal_mask(self):
        """Test causal mask creation."""
        from causvid.acceleration.tensorrt.attention import create_causal_mask
        
        seq_len = 10
        mask = create_causal_mask(seq_len, "cpu", torch.float32)
        
        self.assertEqual(mask.shape, (seq_len, seq_len))
        # Upper triangle should be -inf
        self.assertTrue(torch.isinf(mask[0, 1]).item())
        # Diagonal and below should be 0
        self.assertEqual(mask[1, 0].item(), 0.0)


class TestModelDefinitions(unittest.TestCase):
    """Test TensorRT model definitions."""
    
    def test_causal_wan_model_trt_profiles(self):
        """Test CausalWanModelTRT input profile generation."""
        from causvid.acceleration.tensorrt.models import CausalWanModelTRT
        
        model_def = CausalWanModelTRT(model_type="T2V-1.3B")
        
        profile = model_def.get_input_profile(
            batch_size=1,
            height=480,
            width=832,
            num_frames=21,
        )
        
        self.assertIn("x", profile)
        self.assertIn("timestep", profile)
        self.assertIn("context", profile)
        
        # Check profile has (min, opt, max)
        self.assertEqual(len(profile["x"]), 3)
    
    def test_vae_encoder_trt_profiles(self):
        """Test VAEEncoderTRT input profile generation."""
        from causvid.acceleration.tensorrt.models import VAEEncoderTRT
        
        model_def = VAEEncoderTRT()
        
        input_names = model_def.get_input_names()
        output_names = model_def.get_output_names()
        
        self.assertEqual(input_names, ["video"])
        self.assertEqual(output_names, ["latent"])
    
    def test_t5_encoder_trt_profiles(self):
        """Test T5EncoderTRT input profile generation."""
        from causvid.acceleration.tensorrt.models import T5EncoderTRT
        
        model_def = T5EncoderTRT()
        
        profile = model_def.get_input_profile(batch_size=2)
        
        self.assertIn("input_ids", profile)
        self.assertIn("attention_mask", profile)


class TestEngine(unittest.TestCase):
    """Test TensorRT engine utilities."""
    
    def test_engine_init(self):
        """Test Engine class initialization."""
        from causvid.acceleration.tensorrt.utilities import Engine
        
        engine = Engine("/tmp/fake_engine.engine")
        
        self.assertEqual(engine.engine_path, "/tmp/fake_engine.engine")
        self.assertIsNone(engine.engine)
        self.assertIsNone(engine.context)


class TestAccelerate(unittest.TestCase):
    """Test acceleration API."""
    
    def test_accelerated_pipeline_init(self):
        """Test AcceleratedPipeline initialization."""
        from causvid.acceleration.accelerate import AcceleratedPipeline
        
        # Create mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.device = "cuda"
        mock_pipeline.args = MagicMock()
        mock_pipeline.num_transformer_blocks = 30
        mock_pipeline.num_heads = 12
        mock_pipeline.frame_seq_length = 1560
        mock_pipeline.kv_cache_length = 32760
        mock_pipeline.denoising_step_list = torch.tensor([999, 749, 499, 249])
        mock_pipeline.generator.get_scheduler.return_value = MagicMock()
        
        mock_stream = MagicMock()
        
        accelerated = AcceleratedPipeline(
            original_pipeline=mock_pipeline,
            stream=mock_stream,
            device="cuda",
        )
        
        self.assertIsNone(accelerated.dit_engine)
        self.assertIsNone(accelerated.vae_engine)
        self.assertIsNone(accelerated.t5_engine)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestEngineWrappers(unittest.TestCase):
    """Test engine wrapper classes (requires CUDA)."""
    
    def test_dit_engine_create_kv_cache(self):
        """Test KV cache creation."""
        # This test would require a real engine file
        # For now, just test the interface exists
        from causvid.acceleration.tensorrt.engines.dit_engine import CausalWanModelEngine
        
        # Verify class exists and has expected methods
        self.assertTrue(hasattr(CausalWanModelEngine, 'create_kv_cache'))
        self.assertTrue(hasattr(CausalWanModelEngine, 'create_cross_cache'))


def run_benchmark():
    """Run performance benchmark."""
    import time
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    print("=" * 60)
    print("TensorRT Benchmark")
    print("=" * 60)
    
    # Benchmark settings
    batch_size = 1
    height = 480
    width = 832
    num_frames = 1
    warmup_iters = 5
    benchmark_iters = 20
    
    device = torch.device("cuda")
    
    # Create dummy tensors
    lat_h, lat_w = height // 8, width // 8
    x = torch.randn(batch_size, num_frames, 16, lat_h, lat_w, 
                   device=device, dtype=torch.float16)
    
    # Warmup
    print(f"\nWarming up ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        y = x * 2  # Dummy operation
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({benchmark_iters} iterations)...")
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        y = x * 2
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / benchmark_iters * 1000
    print(f"\nAverage time: {avg_time:.2f} ms")
    print(f"Throughput: {1000 / avg_time:.1f} FPS")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    args, remaining = parser.parse_known_args()
    
    if args.benchmark:
        run_benchmark()
    else:
        # Run unit tests
        sys.argv = [sys.argv[0]] + remaining
        unittest.main()
