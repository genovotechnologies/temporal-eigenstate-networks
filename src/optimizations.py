"""
Performance optimizations for TEN to match/exceed transformer speed
"""
import torch
import torch.nn as nn

def optimize_model_for_inference(model):
    """Apply all speed optimizations for inference"""
    # Use channels-last memory format for better cache locality
    # model = model.to(memory_format=torch.channels_last)
    
    # Fuse operations where possible
    model.eval()
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set matmul precision for speed (Ampere+ GPUs)
    torch.set_float32_matmul_precision('high')  # or 'medium'
    
    return model

def optimize_model_for_training(model):
    """Apply all speed optimizations for training"""
    model.train()
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Set matmul precision for speed
    torch.set_float32_matmul_precision('high')
    
    # Enable TF32 on Ampere+ GPUs for 2-3× speedup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    return model

# JIT-compiled scan for parallel state evolution
@torch.jit.script
def parallel_scan_step(state_real: torch.Tensor, state_imag: torch.Tensor,
                       magnitude: torch.Tensor, cos_phase: torch.Tensor, 
                       sin_phase: torch.Tensor, beta: torch.Tensor,
                       resonance: torch.Tensor):
    """Single step of eigenstate evolution (JIT compiled for speed)"""
    # Complex multiplication
    new_real = magnitude * (state_real * cos_phase - state_imag * sin_phase) + beta
    new_imag = magnitude * (state_real * sin_phase + state_imag * cos_phase)
    
    # Resonance coupling
    state_real = new_real @ resonance
    state_imag = new_imag @ resonance
    
    return state_real, state_imag
