import torch
import triton
import triton.language as tl
import time

@triton.jit
def conv1d_kernel(output, input, kernel, input_size, kernel_size, BLOCK_SIZE: tl.constexpr, CHUNK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_output = offset < input_size - kernel_size + 1
    offset_output = offset
    acc = tl.zeros((BLOCK_SIZE, ), dtype = tl.float32)
    for k in range(0, kernel_size, CHUNK_SIZE):
        offset_kernel = k + tl.arange(0, CHUNK_SIZE)
        offset_input = offset[:, None] + offset_kernel[None, :]
        mask_kernel = offset_kernel < kernel_size
        mask_input = (offset_input < input_size) & (offset_kernel[None, :] < kernel_size)
        kernel_data = tl.load(kernel + offset_kernel, mask = mask_kernel, other = 0.0)
        input_data = tl.load(input + offset_input, mask = mask_input, other = 0.0)
        acc += tl.sum(input_data * kernel_data[None, :], axis = 1)
    tl.store(output + offset_output, acc, mask = mask_output)
