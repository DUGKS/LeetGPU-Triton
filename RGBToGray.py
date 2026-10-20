import triton
import triton.language as tl
import torch

@triton.jit
def RGBToGrayKernel(input, output, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id()
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < width * height
    data_red = tl.load(input + offset * 3, mask = mask, other = 0.0)
    data_green = tl.load(input + offset * 3 + 1, mask = mask, other = 0.0)
    data_blue = tl.load(input + offset * 3 + 2, mask = mask, other = 0.0)
    data_gray = 0.299 * data_red + 0.587 * data_green + 0.114 * data_blue
    tl.store(output + offset, data_gray, mask = mask)
