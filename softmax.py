import torch
import triton
import triton.language as tl

@triton.jit
def partialSumMaxKernel(block_max_ptr, block_sum_ptr, input, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    stride = num_programs * BLOCK_SIZE
    offset_start = pid * BLOCK_SIZE

    block_max = -float('inf')
    for off in range(offset_start, N, stride):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        data_in = tl.load(input + offset, mask, other = -float('inf'))
        local_block_max = tl.max(data_in, axis = 0)
        block_max = tl.maximum(local_block_max, block_max)

    block_sum = 0.0
    for off in range(offset_start, N, stride):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        data_in = tl.load(input + offset, mask, other = 0.0)
        block_sum += tl.sum(tl.exp(data_in - block_max), axis = 0)

    tl.store(block_max_ptr + pid, block_max)
    tl.store(block_sum_ptr + pid, block_sum)

def allSumMaxKernel(block_max_ptr, block_sum_ptr, global_max_ptr, global_sum_ptr, NUM_PARTIALS: tl.constexpr):
    offset = tl.arange(0, NUM_PARTIALS)
    partial_max = tl.load(block_max_ptr + offset)
    partial_sum = tl.load(block_sum_ptr + offset)
    global_max = tl.max(partial_max, axis = 0)
    global_sum = tl.sum(partial_sum * tl.exp(partial_max - global_max), axis = 0)
    tl.store(global_max_ptr, global_max)
    tl.store(global_sum_ptr, global_sum)

@triton.jit
def softmaxKernel(output, input, N, BLOCK_SIZE: tl.constexpr, global_max, global_sum):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offset_start = pid * BLOCK_SIZE
    inverse_global_sum = 1.0 / global_sum
    for off in range(offset_start, N, BLOCK_SIZE * num_programs):
        offset = offset_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        data_in = tl.load(input + offset, mask = mask, other = 0.0)
        data_out = tl.exp(data_in - global_max) * inverse_global_sum
        tl.store(output + offset, data_out, mask = mask)

@triton.jit
def solve_softmax(x: torch.Tensor) -> torch.Tensor:
    N = x.numel()
    ouput = torch.empty_like(x)

    NUM_PROGRAMS = 128
    BLOCK_SIZE = 1024

    partial_sum = torch.empty(NUM_PROGRAMS, device = x.device, dtype = torch.float32)
    partial_max = torch.empty(NUM_PROGRAMS, device = x.device, dtype = torch.float32)

    global_sum = torch.empty(1, device = x.device, dtype = torch.float32)
    global_max = torch.empty(1, device = x.device, dtype = torch.float32)

    partialSumMaxKernel[(NUM_PROGRAMS,)](partial_max, partial_sum, x, N, BLOCK_SIZE)
    allSumMaxKernel[(1,)](partial_max, partial_sum, global_max, global_sum, NUM_PROGRAMS)
    softmaxKernel[(NUM_PROGRAMS, )](output, x, N, BLOCK_SIZE, global_max.item(), global_sum.item())