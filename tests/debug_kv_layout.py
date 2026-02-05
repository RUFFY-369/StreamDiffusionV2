
import torch

def test_kv_layout():
    print("Testing KV Layout & Strides...")
    B = 1
    S_cache = 1560 * 5 # 5 frames
    H = 12
    D = 128
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}, Dtype: {dtype}")

    # Simulated Cache [B, S, H, D] mimicking TRT layout choice
    # TRT usually likes NCHW or similar, but here we have [B, S, H, D]
    cache = torch.randn(B, S_cache, H, D, device=device, dtype=dtype)
    
    # 1. Simulate Ring Buffer Update pattern
    # Wrap-around write indices
    start_logical = 1560 * 6 # Frame 6
    start_physical = (start_logical) % S_cache # Wrap around
    length = 1560
    
    # Create indices
    indices = torch.arange(start_physical, start_physical+length, device=device) % S_cache
    
    # Simulated Update Data
    new_data = torch.randn(B, length, H, D, device=device, dtype=dtype)
    
    # Scatter update (mimic TRT implementation of index_put_)
    cache[:, indices] = new_data
    
    # 2. Slice for attention (The "Linear Phase" check - Frame 5/6 transition)
    # Suppose we are at Frame 5 (Linear, just before ring buffer wraps usually?)
    # Wait, Ring Buffer logic usually wraps much later, but let's test striding.
    
    current_end_pos = start_physical + length
    
    # In TRT code: k_full = new_kv_cache_k[:, :end_idx]
    # Note: If we wrapped, we can't slice simple [:, :end_idx]. 
    # But currently `trt_inference.py` seems to NOT assume wrapping for the view?
    # Actually `attention.py` Line 364: `k_full = new_kv_cache_k[:, :end_idx]`
    # This implies LINEAR view of cache?
    # IF we are wrapping, `new_kv_cache_k` is the PHYSICAL cache.
    # Slicing `[:end_idx]` only makes sense if we haven't wrapped yet OR if we unrolled?
    # NO. The TRT engine logic in `attention.py` processes Ring Buffer properly ONLY IF the mask handles the wrapping?
    # Wait. `attention.py` logic:
    # "Constraint: Within the current chunk ... enforces causality."
    
    # Let's check strides after Transpose
    k_slice = cache[:, :current_end_pos]
    k_transposed = k_slice.transpose(1, 2) # [B, H, S, D]
    
    print(f"Start Physical: {start_physical}")
    print(f"Cache Shape: {cache.shape}")
    print(f"Cache Stride: {cache.stride()}")
    print(f"Slice Shape: {k_slice.shape}")
    print(f"Slice Stride: {k_slice.stride()}")
    print(f"Trans Shape: {k_transposed.shape}")
    print(f"Trans Stride: {k_transposed.stride()}")
    
    # Check alignment
    is_contiguous = k_transposed.is_contiguous()
    print(f"Transposed is Contiguous? {is_contiguous}")
    
    # Test SDPA
    q = torch.randn(B, H, length, D, device=device, dtype=dtype)
    try:
        import torch.nn.functional as F
        # Use simple SDPA
        out = F.scaled_dot_product_attention(q, k_transposed, k_transposed)
        print("SDPA run successful.")
    except Exception as e:
        print(f"SDPA Failed: {e}")

if __name__ == "__main__":
    test_kv_layout()
