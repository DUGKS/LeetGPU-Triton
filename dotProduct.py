import torch
import triton
import triton.language as tl

@triton.jit
def localDotProduct(out_ptr, a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data_A = tl.load(a_ptr + offset, mask = mask, other = 0.0)
    data_B = tl.load(b_ptr + offset, mask = mask, other = 0.0)
    tl.store(out_ptr + pid, tl.sum(data_A * data_B))

@triton.jit
def sumKernel(out_ptr, in_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data_in = tl.load(in_ptr + offset, mask = mask, other = 0.0)
    tl.store(out_ptr + pid, tl.sum(data_in))

# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    a = a.contiguous()
    b = b.contiguous()
    N = a.numel()
    BLOCK_SIZE = 1024
    grid = triton.cdiv(N, BLOCK_SIZE)

    partial_sum = torch.empty(triton.cdiv(N, BLOCK_SIZE), device = a.device, dtype = torch.float32)
    localDotProduct[(grid,)](partial_sum, a, b, N, BLOCK_SIZE)

    while partial_sum.numel() > 1:
        m = partial_sum.numel()
        nxt = torch.empty(triton.cdiv(m, BLOCK_SIZE), device = a.device, dtype = torch.float32)
        sumKernel[(triton.cdiv(m, BLOCK_SIZE),)](nxt, partial_sum, m, BLOCK_SIZE)
        partial_sum = nxt

    result.copy_(partial_sum.view_as(result))