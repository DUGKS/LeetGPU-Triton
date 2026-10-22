import torch
import triton
import triton.language as tl

@triton.jit
def countArray(output, input, N, K, first: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data_in = tl.load(input + offset, mask = mask, other = 0)
    if first:
        v = (mask & (data_in == K)).to(tl.int32)
    else:
        v = data_in
    tl.store(output + pid, tl.sum(v, axis = 0))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    BLOCK_SIZE = 1024
    curr_input = input
    curr_num = N
    first = True
    while True:
        grid = triton.cdiv(curr_num, BLOCK_SIZE)
        if (grid == 1):
            curr_output = output
        else:
            curr_output = torch.empty((grid, ), dtype = input.dtype, device = input.device)
        countArray[(grid, )](curr_output, curr_input, curr_num, K, first, BLOCK_SIZE)
        first = False
        curr_input = curr_output
        curr_num = grid
        if (grid == 1):
            break
