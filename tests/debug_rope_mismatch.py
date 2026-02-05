
import torch
import math
import numpy as np

# ==============================================================================
# Original Implementation (from causvid/models/wan/wan_base/modules/model.py)
# ==============================================================================

def original_rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def original_rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

def original_causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        sf = start_frame[i].item() if isinstance(start_frame, torch.Tensor) else start_frame

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][sf:sf + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

# ==============================================================================
# TRT Implementation (from causvid/acceleration/tensorrt/attention.py)
# ==============================================================================

def trt_rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    # From causal_model_trt.py
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len).float(),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim))
    )
    return freqs  # Real-valued angles

def trt_causal_rope_apply(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: torch.Tensor
) -> torch.Tensor:
    b, seq_len, n, d = x.shape
    half_d = d // 2
    
    freq_splits = [half_d - 2 * (half_d // 3), half_d // 3, half_d // 3]
    freqs_split = freqs.split(freq_splits, dim=1)
    
    output = []
    for i in range(b):
        f, h, w = torch.unbind(grid_sizes[i], dim=0)
        actual_seq_len = f * h * w
        sf = start_frame[i]
        
        x_i = x[i, :actual_seq_len].float()  # [seq, n, d]
        
        # Build frequency tensor
        idx_f = torch.arange(f, device=x.device) + sf
        freq_max = int(freqs_split[0].shape[0]) - 1
        idx_f = idx_f.long().clamp(max=freq_max) 
        
        freqs_f = freqs_split[0][idx_f].view(f, 1, 1, -1).expand(f, h, w, -1)
        freqs_h = freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
        freqs_w = freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        freqs_i = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(actual_seq_len, 1, -1)  # [seq, 1, half_d]
        
        # CORRECTION: WanModel uses adjacent pairs (view_as_complex on last dim)
        # So Real = x[0::2], Imag = x[1::2]
        x_real = x_i[..., 0::2]
        x_imag = x_i[..., 1::2]
        
        # freqs_i is [seq, 1, half_d] -> expand to [seq, n, half_d]
        cos_freqs = freqs_i.cos().expand(-1, n, -1)
        sin_freqs = freqs_i.sin().expand(-1, n, -1)
        
        # Rotate
        # (a + ib) * (cos + isin) = (acos - bsin) + i(asin + bcos)
        out_real = x_real * cos_freqs - x_imag * sin_freqs
        out_imag = x_real * sin_freqs + x_imag * cos_freqs
        
        # Interleave back: stack on last dim then flatten
        # Stack: [seq, n, half_d, 2] -> Flatten: [seq, n, d]
        x_rotated = torch.stack([out_real, out_imag], dim=-1).flatten(-2)
        
        if actual_seq_len < seq_len:
            x_rotated = torch.cat([x_rotated, x[i, actual_seq_len:].float()], dim=0)
        
        output.append(x_rotated)
    
    return torch.stack(output).type_as(x)

# ==============================================================================
# Correctness Test
# ==============================================================================

def test_rope_equivalence():
    print("Testing RoPE equivalence...")
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B = 1
    N = 12
    D = 128 # Head dim
    F = 1   # Frames per block
    H = 30
    W = 52
    
    grid_sizes = torch.tensor([[F, H, W]], device=device)
    start_frame = torch.tensor([5], device=device) # Test at frame 5 (mosaic frame)
    
    # Input
    x = torch.randn(B, F*H*W, N, D, device=device)
    
    # Params
    max_seq_len = 10000
    
    # helper to construct composed freqs
    def build_orig_freqs(dim_head):
        d = dim_head # 128
        p1 = original_rope_params(max_seq_len, d - 4 * (d // 6))
        p2 = original_rope_params(max_seq_len, 2 * (d // 6))
        p3 = original_rope_params(max_seq_len, 2 * (d // 6))
        return torch.cat([p1, p2, p3], dim=1)

    def build_trt_freqs(dim_head):
        d = dim_head
        p1 = trt_rope_params(max_seq_len, d - 4 * (d // 6))
        p2 = trt_rope_params(max_seq_len, 2 * (d // 6))
        p3 = trt_rope_params(max_seq_len, 2 * (d // 6))
        return torch.cat([p1, p2, p3], dim=1)

    # 1. Original
    orig_freqs = build_orig_freqs(D).to(device)
    orig_out = original_causal_rope_apply(x, grid_sizes, orig_freqs, start_frame)
    
    # 2. TRT
    trt_freqs = build_trt_freqs(D).to(device)
    # trt_rope_params returns angles. original returns complex(cos, sin).
    # We must ensure trt_causal_rope_apply uses these angles correctly.
    
    # Run TRT
    trt_out = trt_causal_rope_apply(x, grid_sizes, trt_freqs, start_frame)
    
    # Compare
    diff = (orig_out - trt_out).abs().max().item()
    print(f"Max Difference: {diff}")
    
    if diff > 1e-3:
        print("FAILED: Mismatch detected!")
        # Dig deeper
        print(f"Original shape: {orig_out.shape}")
        print(f"TRT shape: {trt_out.shape}")
    else:
        print("PASSED: Output matches within tolerance.")

if __name__ == "__main__":
    test_rope_equivalence()
