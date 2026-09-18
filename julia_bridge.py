from init import jl
import numpy as np
import torch
from torch.autograd import Function

try:
    print("Loading Julia matmul functions...")
    jl.seval('include("julia_matmul.jl")')
    print("Julia functions loaded successfully")
    if not hasattr(jl, 'batched_matmul'):
        raise RuntimeError("batched_matmul function not found")
    print("Julia bridge ready")
except Exception as e:
    raise RuntimeError(f"Failed to load Julia matmul implementation: {str(e)}")

class JuliaMatmul(Function):
    @staticmethod
    def forward(ctx, a, b, transpose_b=False):
        device = a.device
        dtype = a.dtype
        
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()

        try:
            if a.dim() == 4:
                result_np = jl.batched_matmul(a_np, b_np, transpose_b)
            else:
                if transpose_b:
                    result_np = jl.optimized_matmul(a_np, b_np.T)
                else:
                    result_np = jl.optimized_matmul(a_np, b_np)

            result = torch.from_numpy(np.array(result_np)).to(device=device, dtype=dtype)
            ctx.save_for_backward(a, b)
            ctx.transpose_b = transpose_b
            return result
            
        except Exception as e:
            print(f"Julia matmul failed, falling back to PyTorch: {str(e)}")
            if transpose_b:
                result = torch.matmul(a, b.transpose(-2, -1))
            else:
                result = torch.matmul(a, b)
            ctx.save_for_backward(a, b)
            ctx.transpose_b = transpose_b
            return result

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        transpose_b = ctx.transpose_b

        if transpose_b:
            grad_a = torch.matmul(grad_output, b)
            grad_b = torch.matmul(a.transpose(-2, -1), grad_output)
        else:
            grad_a = torch.matmul(grad_output, b.transpose(-2, -1))
            grad_b = torch.matmul(a.transpose(-2, -1), grad_output)
        
        return grad_a, grad_b, None

def julia_matmul(a, b, transpose_b=False):
    return JuliaMatmul.apply(a, b, transpose_b)