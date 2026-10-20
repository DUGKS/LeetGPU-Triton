import torch
import triton
import triton.language as tl

@triton.jit
def reductionKernel(output, input, N, BLOCK_SIZE : tl.constexpr):
    pid = tl.program_id()
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data_in = tl.load(input + offset, mask, other = 0.0)
    block_sum = tl.sum(data_in)
    tl.atomic_add(output, block_sum)