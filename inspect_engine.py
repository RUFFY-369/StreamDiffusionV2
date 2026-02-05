import tensorrt as trt
import sys
import os

def inspect_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        
    if not engine:
        print(f"Failed to load engine: {engine_path}")
        return

    print(f"Inspecting Engine: {engine_path}")
    print("="*60)
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        is_shape = engine.is_shape_inference_io(name)
        
        mode_str = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
        print(f"Binding {i}: {name}")
        print(f"  Mode: {mode_str}")
        print(f"  Shape: {shape}")
        print(f"  Dtype: {dtype}")
        print(f"  Is Shape Tensor: {is_shape}")
        print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_engine.py <engine_path>")
        sys.exit(1)
    
    inspect_engine(sys.argv[1])
