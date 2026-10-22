import torch
import triton
import triton.language as tl

@triton.jit
def localBlockPrefixSum(out_ptr, in_ptr, block_sums, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data_in = tl.load(in_ptr + offset, mask = mask, other = 0)
    data_out = tl.cumsum(data_in, axis = 0)
    tl.store(out_ptr + offset, data_out, mask = mask)
    tl.store(block_sums + pid, tl.sum(data_in, axis = 0))

@triton.jit
def blockPrefixSum(block_sums, num, BLOCK_SIZE: tl.constexpr):
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < num
    data_in = tl.load(block_sums + offset, mask = mask, other = 0)
    data_out = tl.cumsum(data_in, axis = 0)
    tl.store(block_sums + offset, data_out, mask = mask)

@triton.jit
def prefixSumKernel(output, block_sums, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        return
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data_in = tl.load(output + offset, mask = mask, other = 0)
    value = tl.load(block_sums + pid - 1)
    tl.store(output + offset, data_in + value, mask = mask)

@triton.jit
def subarraySumMax(output, input, num_out, kernel_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset_out = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_out = offset_out < num_out
    offset_in_start = offset_out - 1
    offset_in_end = offset_out + kernel_size - 1

    data_start = tl.load(input + offset_in_start, mask = mask_out & (offset_in_start >= 0), other = 0)
    data_end = tl.load(input + offset_in_end, mask = mask_out, other = 0)

    window = tl.where(mask_out, data_end - data_start, -10000000)
    m = tl.max(window, axis = 0)
    tl.atomic_max(output, m)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
    BLOCK_SIZE = 2048
    num_grids = triton.cdiv(N, BLOCK_SIZE)

    prefix_sums = torch.empty(N, dtype = torch.int32, device = input.device)
    block_sums = torch.zeros((num_grids, ), device = input.device, dtype = input.dtype)
    output.fill_(-2**16)

    localBlockPrefixSum[(num_grids, )](prefix_sums, input, block_sums, N, BLOCK_SIZE)
    blockPrefixSum[(1,)](block_sums, num_grids, BLOCK_SIZE = triton.next_power_of_2(num_grids))
    prefixSumKernel[(num_grids, )](prefix_sums, block_sums, N, BLOCK_SIZE)

    num_out = N - window_size + 1
    grid = triton.cdiv(num_out, BLOCK_SIZE)
    subarraySumMax[(grid, )](output, prefix_sums, num_out, window_size, BLOCK_SIZE)