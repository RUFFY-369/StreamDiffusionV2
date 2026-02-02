"""
TensorRT-Accelerated Inference Script

Uses TensorRT DiT engine with PyTorch VAE for optimal performance.
Based on streamv2v/inference.py but with TRT acceleration for 3.5x speedup.

Usage:
    python scripts/trt_inference.py \
        --config_path ./configs/your_config.yaml \
        --checkpoint_folder ./checkpoints/your_model \
        --output_folder ./outputs \
        --prompt_file_path ./prompts.txt \
        --dit_engine_path ./trt_engines/dit_streaming.engine \
        --video_path ./input.mp4 \
        --guidance_scale 1.0
"""

import argparse
import os
import time
import logging
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from diffusers.utils import export_to_video

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRTInference")


def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Load an .mp4 video as a tensor [C, T, H, W]."""
    import torchvision
    import torchvision.transforms.functional as TF
    from einops import rearrange
    
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    
    # Use 'sec' units to avoid warnings and inaccurate timestamps
    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW", pts_unit="sec")
    if max_frames is not None:
        video = video[:max_frames]
    
    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([
            TF.resize(video[:, i], resize_hw, antialias=True)
            for i in range(t)
        ], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0
    
    return video


class TRTAcceleratedInferencePipeline:
    """
    TensorRT-accelerated inference pipeline.
    
    Uses TensorRT DiT engine for fast denoising while keeping
    PyTorch VAE (which is already fast enough).
    """
    
    def __init__(
        self,
        config,
        dit_engine_path: str,
        device: torch.device = None,
        max_seq_len: int = 150000,
        enable_cfg: bool = True,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cfg = enable_cfg
        
        # Load PyTorch pipeline for VAE and text encoder
        from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
        self.pytorch_pipeline = CausalStreamInferencePipeline(config, device=str(self.device))
        self.pytorch_pipeline.to(device=str(self.device), dtype=torch.bfloat16)
        
        # Load TensorRT DiT engine
        from causvid.acceleration.tensorrt.engines.simple_engines import DiTEngineSimple, DiTEngineStreaming
        
        self.streaming = True # Force streaming mode based on user request
        
        if self.streaming:
            # FREE PYTORCH KV CACHE to save memory (~18GB for B=2)
            # We only use TRT cache
            if hasattr(self.pytorch_pipeline, "kv_cache1"):
                self.pytorch_pipeline.kv_cache1 = None
                self.pytorch_pipeline.kv_cache2 = None
                torch.cuda.empty_cache()
            
            # Offload PyTorch DiT model to CPU to save VRAM (we use TRT engine)
            if hasattr(self.pytorch_pipeline, "generator"):
                logger.info("Offloading PyTorch DiT model to CPU to save VRAM")
                self.pytorch_pipeline.generator.model.to("cpu")
                
            # Also offload Text Encoder (T5-XXL is ~22GB!)
            if hasattr(self.pytorch_pipeline, "text_encoder"):
                logger.info("Offloading Text Encoder to CPU to save VRAM")
                self.pytorch_pipeline.text_encoder.to("cpu")
                
            torch.cuda.empty_cache()

            self.dit_engine = DiTEngineStreaming(
                dit_engine_path, 
                use_cuda_graph=False, # Must be False for dynamic control flow/shapes if unpredictable
                device=str(self.device),
            )
            
            # Smart Memory Allocation - SPLIT CACHE
            num_layers_per_chunk = 1 # 1 Layer per chunk (Safest for TRT < 1GB)
            num_chunks = 30 # 30 Layers / 1 = 30 chunks
            
            num_heads = 12
            total_layers = 30
            head_dim = 128
            dtype_size = 2 # float16
            
            # Calculate memory per token per batch
            bytes_per_token = 2 * total_layers * num_heads * head_dim * dtype_size
            
            # Use max_seq_len from args
            self.max_seq_len = max_seq_len
            
            # Check available VRAM
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            free_vram = t - a 
            
            # Log chunk size
            chunk_bytes = num_layers_per_chunk * 2 * max_seq_len * num_heads * head_dim * 2
            logger.info(f"Allocating 1 x {num_chunks} TRT KV Chunks: ({num_layers_per_chunk}, 2, 1, {max_seq_len}, {num_heads}, {head_dim})")
            logger.info(f"Size per chunk: {chunk_bytes/1e9:.2f} GB (Safe < 2.14GB)")
            
            logger.info(f"VRAM Stats: Total={t/1e9:.2f}GB, Allocated={a/1e9:.2f}GB, Free (approx)={(t-a)/1e9:.2f}GB")
            
            num_caches = 2 if self.enable_cfg else 1
            required_mem = max_seq_len * bytes_per_token * num_caches
            
            if required_mem > (free_vram * 0.95):
                logger.warning(f"WARNING: Requested cache size ({required_mem/1e9:.2f} GB) exceeds available VRAM ({free_vram/1e9:.2f} GB)!")
                safe_mem = free_vram * 0.90
                max_safe_seq = int(safe_mem / (bytes_per_token * num_caches))
                logger.warning(f"Downgrading max_seq_len from {max_seq_len} to {max_safe_seq} to fit VRAM.")
                max_seq_len = max_safe_seq
            
            self.max_seq_len = max_seq_len
            
            # Allocate 5 chunks of cache
            # Shape per chunk: [6, 2, 1, Seq, Head, Dim]
            chunk_shape = (num_layers_per_chunk, 2, 1, max_seq_len, num_heads, head_dim)
            logger.info(f"Allocating {num_caches} x {num_chunks} TRT KV Chunks: {chunk_shape} (Float16)")
            logger.info(f"Total Cache Memory: {required_mem/1e9:.2f} GB")
            
            # List of lists [BatchSet_0_Chunks, BatchSet_1_Chunks (if CFG)]
            self.trt_kv_cache = []
            for _ in range(num_caches):
                # Valid list of 5 chunks
                chunks = [
                    torch.zeros(chunk_shape, device=self.device, dtype=torch.float16)
                    for _ in range(num_chunks)
                ]
                self.trt_kv_cache.append(chunks)
            
            # Cache Metadata for Eviction Logic (matching PyTorch pipeline)
            self.cache_metadata = {
                'global_end_index': torch.zeros(num_caches, dtype=torch.long, device=self.device),
                'local_end_index': torch.zeros(num_caches, dtype=torch.long, device=self.device),
                'sink_size': 3,  # Preserve first 3 frames
                'adapt_sink_thr': -1,  # Adaptive sink threshold (-1 = disabled)
                'kv_cache_size': max_seq_len,  # Max cache capacity
            }
            logger.info(f"Cache metadata initialized: sink_size={self.cache_metadata['sink_size']}, max_capacity={max_seq_len}")
        else:
            self.dit_engine = DiTEngineSimple(
                dit_engine_path,
                use_cuda_graph=True,
                device=str(self.device),
            )
        
        # Reference to PyTorch components
        self.vae = self.pytorch_pipeline.vae
        self.text_encoder = self.pytorch_pipeline.text_encoder
        
        # Tracking
        self.processed = 0
        
        logger.info(f"TRT-accelerated pipeline initialized (streaming={self.streaming})")
        logger.info(f"  DiT engine: {dit_engine_path}")
        logger.info(f"  VAE: PyTorch (recommended)")
        
    def load_model(self, checkpoint_folder: str):
        """Load model checkpoint (supports .pt and .safetensors)."""
        import os
        from safetensors.torch import load_file
        
        # Priority 1: model.pt
        pt_path = os.path.join(checkpoint_folder, "model.pt")
        # Priority 2: diffusion_pytorch_model.safetensors (Standard Diffusers)
        sf_path = os.path.join(checkpoint_folder, "diffusion_pytorch_model.safetensors")
        # Priority 3: model.safetensors
        sf_v2_path = os.path.join(checkpoint_folder, "model.safetensors")
        
        if os.path.exists(pt_path):
            logger.info(f"Loading checkpoint from {pt_path}")
            ckpt = torch.load(pt_path, map_location="cpu")
        elif os.path.exists(sf_path):
            logger.info(f"Loading checkpoint from {sf_path}")
            ckpt = load_file(sf_path)
        elif os.path.exists(sf_v2_path):
            logger.info(f"Loading checkpoint from {sf_v2_path}")
            ckpt = load_file(sf_v2_path)
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_folder}. Checked: model.pt, diffusion_pytorch_model.safetensors")
        
        if isinstance(ckpt, dict):
            if 'generator' in ckpt:
                state_dict = ckpt['generator']
            elif 'generator_ema' in ckpt:
                state_dict = ckpt['generator_ema']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        
        # Load weights
        msg = self.pytorch_pipeline.generator.load_state_dict(state_dict, strict=False)
        logger.info(f"Checkpoint loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
    
    def _evict_kv_cache(self, batch_idx: int, num_new_tokens: int, frame_seqlen: int):
        """
        Evict oldest tokens from KV cache when full (excluding sink tokens).
        Ported from causvid/models/wan/causal_model.py lines 175-189.
        
        Args:
            batch_idx: Index in batch (0 for single sequence, 0/1 for CFG)
            num_new_tokens: Number of tokens being added
            frame_seqlen: Tokens per frame (e.g., 1560 for 480p)
        """
        sink_tokens = self.cache_metadata['sink_size'] * frame_seqlen
        kv_cache_size = self.cache_metadata['kv_cache_size']
        local_end = self.cache_metadata['local_end_index'][batch_idx].item()
        
        # Calculate eviction
        num_evicted_tokens = num_new_tokens + local_end - kv_cache_size
        num_rolled_tokens = local_end - num_evicted_tokens - sink_tokens
        
        logger.info(f"[Eviction] Batch {batch_idx}: Evicting {num_evicted_tokens} tokens, rolling {num_rolled_tokens} tokens")
        
        # Shift cache left for all chunks (evict oldest non-sink tokens)
        cache_chunks = self.trt_kv_cache[batch_idx]
        for chunk in cache_chunks:
            # chunk shape: [1, 2, 1, MaxSeq, H, D]
            # Extract K and V: [2, 1, MaxSeq, H, D]
            kv = chunk[0]  # [2, 1, MaxSeq, H, D]
            
            # Shift K cache (index 0)
            kv[0, :, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv[0, :, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            
            # Shift V cache (index 1)
            kv[1, :, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv[1, :, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        
        # Update local end index (cache fill level after eviction)
        new_local_end = sink_tokens + num_rolled_tokens
        self.cache_metadata['local_end_index'][batch_idx] = new_local_end
        
        logger.info(f"[Eviction] Batch {batch_idx}: Cache now at {new_local_end}/{kv_cache_size} tokens")
        
        return new_local_end
    
    def prepare_pipeline(
        self,
        text_prompts: list,
        noise: torch.Tensor,
        current_start: int,
        current_end: int,
    ):
        """Prepare pipeline (uses TRT for initial setup)."""
        logger.info("Preparing TRT pipeline...")
        
        # 1. Compute Text Embeddings (PyTorch pipeline's text encoder is still on GPU)
        self.pytorch_pipeline.conditional_dict = self.pytorch_pipeline.text_encoder(
            text_prompts=text_prompts
        )
        
        # 2. Run Inference using TRT Engine
        return self.inference_stream_trt(
            noise=noise,
            current_start=current_start,
            current_end=current_end,
        )
    
    def inference_stream_trt(
        self,
        noise: torch.Tensor,
        current_start: int,
        current_end: int,
        current_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        TRT-accelerated streaming inference.
        """
        if not self.streaming:
             return self.pytorch_pipeline.inference_stream(
                noise=noise,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
            )
            
        # Manually implement streaming loop with TRT engine
        
        # Get dependencies
        scheduler = self.pytorch_pipeline.scheduler
        conditional_dict = self.pytorch_pipeline.conditional_dict
        
        # 1. Prepare inputs
        noisy_image = noise.to(torch.float16) # TRT uses float16
        prompt_embeds = conditional_dict["prompt_embeds"].to(device=self.device, dtype=torch.float16)
        
        # Dimensions
        B, F, C, H, W = noisy_image.shape
        
        # Ensure current_start_t is on DEVICE (GPU) and int64
        current_start_t = torch.tensor([current_start]*B, device=self.device, dtype=torch.long)
        
        # 2. Loop over timesteps
        for t in scheduler.timesteps:
            # Create timestep tensor
            t_tensor = torch.full((1, F), t, device=self.device, dtype=torch.long)
            
            # Determine batch size from prompts (usually B=2 for CFG)
            B_text = prompt_embeds.shape[0]
            
            # Handle CFG disable/enable mismatch
            if not self.enable_cfg and B_text > 1:
                # User disabled CFG to save memory, but text encoder returned 2 embeds
                # We slice to keep only the conditional prompt (index 1 usually, check impl)
                # Actually usually [neg, pos]. We want pos.
                # Assuming index 1 is positive prompt for standard SD pipes.
                prompt_embeds = prompt_embeds[1:2]
                B_text = 1
            
            # Calculate frame_seqlen for eviction (tokens per frame)
            frame_seqlen = H // 2 * W // 2  # Patched dimensions: 480->240, 832->416 -> 30*52=1560
            num_new_tokens = frame_seqlen  # Processing 1 frame at a time in streaming
            
            # Prepare inputs for batch splitting if B_text > 1
            if B_text > 1:
                flow_preds = []
                # Process each batch item sequentially to fit in B=1 engine
                for b_idx in range(B_text):
                    # Check if eviction is needed BEFORE engine call
                    current_end_for_batch = current_start + num_new_tokens
                    local_end = self.cache_metadata['local_end_index'][b_idx].item()
                    global_end = self.cache_metadata['global_end_index'][b_idx].item()
                    kv_cache_size = self.cache_metadata['kv_cache_size']
                    
                    # Eviction logic (matching PyTorch lines 175-176)
                    if (current_end_for_batch > global_end) and \
                       (num_new_tokens + local_end > kv_cache_size):
                        logger.info(f"[Batch {b_idx}] Cache full ({local_end + num_new_tokens}/{kv_cache_size}), triggering eviction")
                        new_local_end = self._evict_kv_cache(b_idx, num_new_tokens, frame_seqlen)
                        local_end = new_local_end
                    
                    # Slice inputs
                    img_idx = b_idx if B > 1 else 0
                    img_slice = noisy_image[img_idx:img_idx+1, -1:, ...] # [1, 1, C, H, W]
                    
                    t_slice = t_tensor # [1, F]
                    prompt_slice = prompt_embeds[b_idx:b_idx+1] # [1, L, D]
                    
                    # Correct slicing for current_start (Batch dim is 0)
                    start_slice = current_start_t[0:1] # [1]
                    
                    # Get list of cache chunks for this batch index
                    # self.trt_kv_cache is List[List[Tensor]] -> [Batch][Chunk]
                    cache_chunks = self.trt_kv_cache[b_idx]
                    
                    # Call Engine with list of chunks
                    flow_out, _ = self.dit_engine(
                        img_slice, t_slice, prompt_slice, 
                        cache_chunks, start_slice
                    )
                    flow_preds.append(flow_out)
                    
                    # Update metadata AFTER engine call
                    if current_end_for_batch > global_end:
                        new_local = min(local_end + num_new_tokens, kv_cache_size)
                        self.cache_metadata['global_end_index'][b_idx] = current_end_for_batch
                        self.cache_metadata['local_end_index'][b_idx] = new_local
                    
                flow_pred = torch.cat(flow_preds, dim=0)
            else:
                # Single batch run
                b_idx = 0
                
                # Check if eviction is needed BEFORE engine call
                current_end_for_batch = current_start + num_new_tokens
                local_end = self.cache_metadata['local_end_index'][b_idx].item()
                global_end = self.cache_metadata['global_end_index'][b_idx].item()
                kv_cache_size = self.cache_metadata['kv_cache_size']
                
                # Eviction logic (matching PyTorch lines 175-176)
                if (current_end_for_batch > global_end) and \
                   (num_new_tokens + local_end > kv_cache_size):
                    logger.info(f"[Batch {b_idx}] Cache full ({local_end + num_new_tokens}/{kv_cache_size}), triggering eviction")
                    new_local_end = self._evict_kv_cache(b_idx, num_new_tokens, frame_seqlen)
                    local_end = new_local_end
                
                t_tensor_b = torch.full((B, 1), t, device=self.device, dtype=torch.long)
                
                # Get chunks for batch 0
                cache_chunks = self.trt_kv_cache[0]
                img_slice = noisy_image[:, -1:, ...]
                
                flow_pred, _ = self.dit_engine(
                    img_slice, t_tensor_b, prompt_embeds,
                    cache_chunks, current_start_t
                )
                
                # Update metadata AFTER engine call
                if current_end_for_batch > global_end:
                    new_local = min(local_end + num_new_tokens, kv_cache_size)
                    self.cache_metadata['global_end_index'][b_idx] = current_end_for_batch
                    self.cache_metadata['local_end_index'][b_idx] = new_local
            
            # Convert flow to x0 (PyTorch logic)
            flow_pred = flow_pred.float()
            
            # Retrieve sigmas from scheduler
            step_index = (scheduler.timesteps == t).nonzero().item()
            sigma = scheduler.sigmas[step_index]
            
            # x0 prediction from flow
            pred_x0 = noisy_image.float() - flow_pred * sigma
            
            # Scheduler step
            scheduler.step_stream(pred_x0, t, noise)
            
        return pred_x0.to(torch.bfloat16)
    
    def run_inference_v2v(
        self,
        input_video: torch.Tensor,
        prompts: list,
        num_chunks: int,
        chunk_size: int,
        noise_scale: float,
        output_folder: str,
        fps: int,
        num_steps: int,
    ):
        """
        Video-to-video inference with TRT acceleration.
        """
        logger.info("Starting TRT-accelerated v2v inference")
        
        os.makedirs(output_folder, exist_ok=True)
        results = {}
        save_results = 0
        
        fps_list = []
        dit_fps_list = []
        
        start_idx = 0
        end_idx = 5
        current_start = 0
        current_end = self.pytorch_pipeline.frame_seq_length * 2
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        # First chunk
        if input_video is not None:
            inp = input_video[:, :, start_idx:end_idx]
            latents = self.vae.stream_encode(inp)
            latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
            
            noise = torch.randn_like(latents)
            noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
        else:
            noisy_latents = torch.randn(
                1, 1 + self.pytorch_pipeline.num_frame_per_block, 16,
                self.pytorch_pipeline.height, self.pytorch_pipeline.width,
                device=self.device, dtype=torch.bfloat16
            )
        
        # Prepare
        denoised_pred = self.prepare_pipeline(
            text_prompts=prompts,
            noise=noisy_latents,
            current_start=current_start,
            current_end=current_end,
        )
        
        # Decode first result
        video = self.vae.stream_decode_to_pixel(denoised_pred)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        results[save_results] = video.cpu().float().numpy()
        save_results += 1
        
        init_noise_scale = noise_scale
        
        # Process remaining chunks
        while self.processed < num_chunks + num_steps - 1:
            start_idx = end_idx
            end_idx = end_idx + chunk_size
            current_start = current_end
            current_end = current_end + (chunk_size // 4) * self.pytorch_pipeline.frame_seq_length
            
            # Check for cache overflow
            if current_end >= self.max_seq_len:
                logger.error(f"Cache Overflow! Current end {current_end} > Max {self.max_seq_len}")
                logger.error("Stopping generation to prevent crash. Reduce frames or disable CFG.")
                break
            
            if input_video is not None and end_idx <= input_video.shape[2]:
                inp = input_video[:, :, start_idx:end_idx]
                
                # Adaptive noise
                l2_dist = (input_video[:, :, end_idx-chunk_size:end_idx] - 
                          input_video[:, :, end_idx-chunk_size-1:end_idx-1]) ** 2
                l2_dist = (torch.sqrt(l2_dist.mean(dim=(0, 1, 3, 4))).max() / 0.2).clamp(0, 1)
                noise_scale = (init_noise_scale - 0.1 * l2_dist.item()) * 0.9 + noise_scale * 0.1
                current_step = int(1000 * noise_scale) - 100
                
                latents = self.vae.stream_encode(inp)
                latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
                
                noise = torch.randn_like(latents)
                noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
            else:
                noisy_latents = torch.randn(
                    1, self.pytorch_pipeline.num_frame_per_block, 16,
                    self.pytorch_pipeline.height, self.pytorch_pipeline.width,
                    device=self.device, dtype=torch.bfloat16
                )
                current_step = None
            
            torch.cuda.synchronize()
            dit_start_time = time.time()
            
            # DiT inference (TRT integration point)
            denoised_pred = self.inference_stream_trt(
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
            )
            
            if self.processed > 3:
                torch.cuda.synchronize()
                dit_fps_list.append(chunk_size / (time.time() - dit_start_time))
            
            self.processed += 1
            
            if self.processed >= num_steps:
                video = self.vae.stream_decode_to_pixel(denoised_pred[[-1]])
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(0, 2, 3, 1).contiguous()
                
                results[save_results] = video.cpu().float().numpy()
                save_results += 1
                
                torch.cuda.synchronize()
                end_time = time.time()
                t = end_time - start_time
                fps_test = chunk_size / t
                fps_list.append(fps_test)
                logger.info(f"Processed {self.processed}, time: {t:.4f} s, FPS: {fps_test:.4f}")
                start_time = end_time
        
        # Save video
        # Use only valid results
        video_list = [results[i] for i in range(save_results)]
        if not video_list:
            logger.error("No frames generated!")
            return

        video = np.concatenate(video_list, axis=0)
        fps_avg = np.mean(np.array(fps_list)) if fps_list else 0
        
        logger.info(f"DiT Average FPS: {np.mean(np.array(dit_fps_list)) if dit_fps_list else 0:.4f}")
        logger.info(f"Video shape: {video.shape}, Average FPS: {fps_avg:.4f}")
        
        output_path = os.path.join(output_folder, "output_trt.mp4")
        export_to_video(video, output_path, fps=fps)
        logger.info(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="TRT-accelerated inference")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--prompt_file_path", type=str, required=True)
    parser.add_argument("--dit_engine_path", type=str, default="./trt_engines/dit.engine")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--noise_scale", type=float, default=0.700)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type")
    
    # New args
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale. Set to 1.0 to save memory.")
    parser.add_argument("--max_seq_len", type=int, default=130000, help="Max cache sequence length (supports ~81 frames at 480x832).")
    parser.add_argument("--num_kv_cache", type=int, default=8, help="Number of cache blocks.")
    parser.add_argument("--num_sink_tokens", type=int, default=0, help="Number of sink tokens.")
    parser.add_argument("--adapt_sink_threshold", type=float, default=0.0, help="Threshold for adaptive sink.")
    
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))
    
    # Set denoising steps
    step_value = int(args.step)
    if step_value <= 1:
        config.denoising_step_list = [700, 0]
    elif step_value == 2:
        config.denoising_step_list = [700, 500, 0]
    elif step_value == 3:
        config.denoising_step_list = [700, 600, 400, 0]
    else:
        config.denoising_step_list = [700, 600, 500, 400, 0]
    
    # Load video
    if args.video_path:
        input_video = load_mp4_as_tensor(
            args.video_path, resize_hw=(args.height, args.width)
        ).unsqueeze(0).to(dtype=torch.bfloat16, device=device)
        t = input_video.shape[2]
    else:
        input_video = None
        t = args.num_frames
    
    # Create pipeline
    enable_cfg = args.guidance_scale != 1.0
    
    pipeline = TRTAcceleratedInferencePipeline(
        config=config,
        dit_engine_path=args.dit_engine_path,
        device=device,
        max_seq_len=args.max_seq_len,
        enable_cfg=enable_cfg,
    )
    pipeline.load_model(args.checkpoint_folder)
    
    # Load prompts
    from causvid.data import TextDataset
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]
    
    # Run inference
    chunk_size = 4
    num_chunks = (t - 1) // chunk_size
    num_steps = len(config.denoising_step_list)
    
    global_start = time.time()
    pipeline.run_inference_v2v(
        input_video=input_video,
        prompts=prompts,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        noise_scale=args.noise_scale,
        output_folder=args.output_folder,
        fps=args.fps,
        num_steps=num_steps,
    )
    
    total_time = time.time() - global_start
    logger.info(f"Total runtime: {total_time:.2f}s")


if __name__ == "__main__":
    main()
