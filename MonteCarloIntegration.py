import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl

@triton.jit
def sumBlock(partial_ptr, input_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_input = offset < N
    data_input = tl.load(input_ptr + offset, mask = mask_input, other = 0.0)
    data_out = tl.sum(data_input, axis = 0)
    tl.store(partial_ptr + pid, data_out)

@triton.jit
def sumTotal(output, input_ptr, N, alpha, BLOCK_SIZE: tl.constexpr):
    assert N <= BLOCK_SIZE, "N should be less than BLOCK_SIZE"
    offset = tl.arange(0, BLOCK_SIZE)
    mask_input = offset < N
    data_input = tl.load(input_ptr + offset, mask = mask_input, other = 0.0)
    tl.store(output, tl.sum(data_input, axis = 0) * alpha)

# y_samples, result are tensors on the GPU
def solve(y_samples: torch.Tensor, result: torch.Tensor, a: float, b: float, n_samples: int):
    N = y_samples.numel()
    BLOCK_SIZE = 1024
    curr_N = N
    curr_input = y_samples

    GRID_SIZE = triton.cdiv(N, BLOCK_SIZE)
    partial_samples = torch.zeros((GRID_SIZE, ), device = y_samples.device, dtype = y_samples.dtype)
    while (curr_N > BLOCK_SIZE):
        grid_size = triton.cdiv(curr_N, BLOCK_SIZE)
        partial_samples = torch.zeros((grid_size, ), device = y_samples.device, dtype = y_samples.dtype)
        sumBlock[(grid_size, )](partial_samples, curr_input, curr_N, BLOCK_SIZE)
        curr_N = grid_size
        curr_input = partial_samples

    sumTotal[(1,)](result, curr_input, curr_N, (b - a) / N, BLOCK_SIZE)