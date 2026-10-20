import torch
import triton
import triton.language as tl

@triton.jit
def maxPoolingKernel(output, input, kernel_size, N, C, H, W, H_out, W_out, stride, padding, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * H_out * W_out
    mask = offset < total

    w_out = offset % W_out
    tmp = offset // W_out
    h_out = offset % H_out
    nc = offset // H_out

    h_in_start = h_out * stride - padding
    w_in_start = w_out *stride - padding
    base = nc * H * W

    acc = tl.full((BLOCK_SIZE, ), float('-inf'), dtype = tl.float32)
    for kh in range(0, kernel_size):
        for kw in range(0, kernel_size):
            h_idx = h_in_start + hk
            w_idx = w_in_start + kw
            valid = mask & (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
            val = tl.load(input + base + h_idx * W + w_idx, mask = valid, other = -float('inf'))
            acc = tl.maximum(acc, val)

    tl.store(output + offset, acc, mask = mask)

# input, output are tensors on the GPU
def solve(input, output, N, C, H, W, kernel_size, stride, padding):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N * C * H_out * W_out, BLOCK_SIZE), )
    maxPoolingKernel[grid](output, input, kernel_size, N, C, H, W, H_out, W_out, stride, padding, BLOCK_SIZE)
