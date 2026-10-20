import torch
import triton
import triton.language as tl

@triton.jit
def clamp(data, lo, hi):
    return tl.minimum(tl.maximum(data, lo), hi)

@triton.jit
def clampKernel(output, input, lo, hi, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_in = offset < N
    data_in = tl.load(input + offset, mask, other = 0.0)
    data_out = clamp(data_in, lo, hi)
    tl.store(output + offset, data_out, mask)
