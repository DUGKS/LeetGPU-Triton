import torch
import triton
import triton.language as tl

@triton.jit
def histogrammingKernel(output, input, N, num_bins, BLOCK_SIZE: tl.):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data_in = tl.load(input + offset, mask = mask, other = 0)
    tl.atomic_add(output + data_in, 1, mask = mask)


