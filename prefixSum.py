import torch
import triton
import triton.language as tl

@triton.jit
def scanChunk(out_ptr, sums_ptr, in_ptr, N, CHUNK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    mask = offset < N
    data_in = tl.load(in_ptr + offset, mask = mask, other = 0.0)
    data_sum = tl.cumsum(data_in)
    tl.store(out_ptr + offset, data_sum, mask = mask)
    tl.store(sums_ptr + pid, tl.sum(data_in))

@triton.jit
def scanBlock(sums_ptr, N, BLOCK_SIZE: tl.constexpr):
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    block_in = tl.load(sums_ptr + offset, mask = mask, other = 0.0)
    block_sum = tl.cumsum(block_in)
    tl.store(sums_ptr + offset, block_sum, mask = mask)

@triton.jit
def prefixSum(output_ptr, sums_ptr, N, CHUNK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if (pid == 0):
       return
    offset = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    mask = offset < N
    data_in = tl.load(output_ptr + offset, mask = mask, other = 0.0)
    offset_v = tl.load(sums_ptr + pid - 1)
    tl.store(output_ptr + offset, data_in + offset_v, mask = mask)