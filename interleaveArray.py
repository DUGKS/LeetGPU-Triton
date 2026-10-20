import torch
import triton
import triton.language as tl

@triton.jit
def interleaveKernel(input_A, input_B, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data_A = tl.load(input_A + offset, mask, other = 0.0)
    data_B = tl.load(input_B + offset, mask, other = 0.0)
    result = tl.interleave(data_A, data_B)
    offset_out = pid * BLOCK_SIZE * 2 + tl.arange(0, 2 * BLOCK_SIZE)
    mask_out = offset_out < 2 * N
    tl.store(output + offset_out, result, mask_out)
