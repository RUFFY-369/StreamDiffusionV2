
import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.getcwd())

from causvid.models.wan.wan_base.modules.model import WanModel

def check_patch_embedding():
    print("Checking WanModel patch embedding...")
    
    # Dummy args
    model = WanModel(
        in_dim=16,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=12,
        num_layers=30,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6
    )
    
    print(f"Patch Size: {model.patch_size}")
    
    # Check patch_embedding layer if it exists
    if hasattr(model, 'patch_embedding'):
        pe = model.patch_embedding
        print(f"Patch Embedding Layer: {pe}")
        if hasattr(pe, 'kernel_size'):
            print(f"Kernel Size: {pe.kernel_size}")
        if hasattr(pe, 'stride'):
            print(f"Stride: {pe.stride}")
            
    # Check if stride matches assumption (1,2,2)
    stride_t = pe.stride
    is_stride_1 = all(s == 1 for s in stride_t)
    print(f"Is Stride 1? {is_stride_1}")

if __name__ == "__main__":
    check_patch_embedding()
