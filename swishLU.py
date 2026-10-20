import triton
import triton.language as tl
import torch

@triton.jit
def swishLUKernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N // 2
    data_input_A = tl.load(input + offset, mask, other = 0.0)
    data_input_B = tl.load(input + offset + N // 2, mask, other = 0.0)
    output_data = data_input_A * tl.sigmoid(data_input_A) * data_input_B
    tl.store(output + offset, output_data, mask = mask)
