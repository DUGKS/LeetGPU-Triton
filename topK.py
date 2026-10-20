import torch
import triton
import triton.language as tl

@triton.jit
def topKBlock(output_ptr, input_ptr, N, K: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data_x = tl.load(input_ptr + offset, mask = mask, other = -float('inf'))
    data_sorted = tl.sort(data_x, descending = True)
    offset_output = pid * K + tl.arange(0, BLOCK_SIZE)
    mask_out = (offset_output - pid * K) < K
    tl.store(output_ptr + offset_output, data_sorted, mask = mask_out)

@triton.jit
def topK(output_ptr, input_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    offset = tl.arange(0, BLOCK_SIZE)
    mask_in = offset < N
    data_x = tl.load(input_ptr + offset, mask = mask_in, other = -float('inf'))
    data_sorted = tl.sort(data_x, descending = True)
    mask_out = offset < K
    tl.store(output_ptr + offset, data_sorted, mask_out)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    BLOCK_SIZE = 1024
    curr_N = N
    curr_input = input
    FINAL_SIZE = 4096
    while curr_N > FINAL_SIZE:
        num_blocks = triton.cdiv(curr_N, BLOCK_SIZE)
        num = num_blocks * k
        partial_K = torch.empty((num, ), device = input.device, dtype = torch.float32)
        topKBlock[(num_blocks,)](partial_K, curr_input, curr_N, k, BLOCK_SIZE)
        curr_N = num
        curr_input = partial_K
    BS = triton.next_power_of_2(max(curr_N, 16))
    top[(1,)](output, curr_input, curr_N, k, BS)
