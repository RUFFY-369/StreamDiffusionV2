
import torch
import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causvid.models.wan.wan_base.modules.vae import _video_vae
from causvid.models.wan.wan_wrapper import WanVAEWrapper

def test_vae_state_persistence():
    print("Initializing WanVAE...")
    # Use small VAE config to avoid OOM and speed up
    from causvid.models.wan.wan_base.modules.vae import WanVAE_
    
    cfg = dict(
        dim=32, # Small dim
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[False, False], # Simple
        dropout=0.0
    )
    
    model = WanVAE_(**cfg).eval().cuda()
    
    # Simulator
    # Use T=6 frames total. 
    # VAE stream_decode behavior:
    # First batch: splits into 0:1 (first frame) and 1: (rest)
    # So we need input T >= 2 to avoid empty slice error in current code.
    T = 6
    H, W = 64, 64
    z_dim = 4
    
    # 1. Run "Prepare" phase (Frame 0-3)
    print("\n--- Phase 1: Prepare (Frame 0-3) ---")
    latents_1 = torch.randn(1, z_dim, 4, H//8, W//8).cuda()
    scale = [0.0, 1.0] # No scaling
    
    # Decode Frame 0-3
    out0_pass1 = model.stream_decode(latents_1, scale)
    print(f"Pass 1 Frames 0-3 Computed. Shape: {out0_pass1.shape}")
    
    # 2. "Rewind" - We WANT to start over from Frame 0.
    # But we DO NOT call clear_cache or reset first_decode
    print("\n--- Phase 2: Rewind & Stream (Frame 0-3) ---")
    
    # Decode Frame 0-3 (Again) - Should match Pass 1 if stateless/reset
    out0_pass2 = model.stream_decode(latents_1, scale)
    
    diff_0 = (out0_pass1 - out0_pass2).abs().max().item()
    print(f"Diff (Pass 1 vs Pass 2): {diff_0}")
    
    if diff_0 > 1e-5:
        print("FAIL: VAE State Persisted! Pass 2 output differs from Pass 1.")
    else:
        print("SUCCESS: VAE State did not persist (Unexpected if bug exists).")

    # 3. "Rewind" WITH Masking Fix
    print("\n--- Phase 3: Rewind WITH Reset ---")
    
    # Manual Reset
    model.first_decode = True
    model.clear_cache_decode()
    
    out0_pass3 = model.stream_decode(latents_1, scale)
    
    diff_reset = (out0_pass1 - out0_pass3).abs().max().item()
    print(f"Diff (Pass 1 vs Pass 3 [Reset]): {diff_reset}")
    
    if diff_reset < 1e-5:
        print("SUCCESS: With Reset, output matches Pass 1.")
    else:
        print("FAIL: Even with reset, output differs?")

if __name__ == "__main__":
    try:
        test_vae_state_persistence()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
