import torch
import triton
import triton.language as tl

@triton.jit
def expectationKernel(expectation, input, N, C, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    offset_col = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offset_row = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offset_in = offset_row[:, None] * C + offset_col[None, :]
    mask_in = (offset_col[None, :] < C) & (offset_row[:, None] < N)
    data_in = tl.load(input + offset_in, mask = mask_in, other = 0.0)
    data_out = tl.sum(data_in, axis = 0) / N
    offset_out = offset_col
    mask_out = offset_col < C
    tl.atomic_add(expectation + offset_out, data_out, mask_out)

@triton.jit
def varianceKernel(variance, input, expectation, N, C, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offset_col = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offset_row = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offset_in = offset_row[:, None] * C + offset_col[None, :]
    mask_in = (offset_col[None, :] < C) & (offset_row[:, None] < N)
    mask_exp = offset_col < C

    data_in = tl.load(input + offset_in, mask = mask_in, other = 0.0)
    data_exp = tl.load(expectation + offset_col, mask = mask_exp, other = 0.0)
    data_out = data_in - data_exp[None, :]
    data_out = tl.where(mask_in, data_out, 0.0)
    data_out = tl.sum(data_out * data_out / N, axis = 0)

    offset_out = offset_col
    mask_out = offset_col < C
    tl.atomic_add(variance + offset_out, data_out, mask_out)

@triton.jit
def batchNormalization(output, input, expectation, variance, gamma, beta, eps, N, C, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offset_col = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offset_row = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offset = offset_row[:, None] * C + offset_col[None, :]
    mask = (offset_col[None, :] < C) & (offset_row[:, None] < N)
    mask_col = offset_col < C

    data_in = tl.load(input + offset, mask = mask, other = 0.0)

    data_mu = tl.load(expectation + offset_col, mask = mask_col, other = 0.0)
    data_sigma = tl.load(variance + offset_col, mask = mask_col, other = 0.0)
    data_gamma = tl.load(gamma + offset_col, mask = mask_col, other = 0.0)
    data_beta = tl.load(beta + offset_col, mask = mask_col, other = 0.0)

    data_alpha = data_gamma * tl.math.rsqrt(data_sigma + eps)
    data_norm = data_alpha[None, :] * (data_in - data_mu[None, :]) + data_beta[None, :]

    tl.store(output + offset, data_norm, mask = mask)



# input, gamma, beta, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    expectation = torch.zeros((C, ), device = input.device, dtype = torch.float32)
    variance = torch.zeros((C, ), device = input.device, dtype = torch.float32)
    BLOCK_X = 16
    BLOCK_Y = 64
    grid = (triton.cdiv(C, BLOCK_X), triton.cdiv(N, BLOCK_Y))
    expectationKernel[grid](expectation, input, N, C, BLOCK_X, BLOCK_Y)
    varianceKernel[grid](variance, input, expectation, N, C, BLOCK_X, BLOCK_Y)
    batchNormalization[grid](output, input, expectation, varia