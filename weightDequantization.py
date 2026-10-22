import torch
import triton
import triton.language as tl

@triton.jit
def weightDequantizationKernel(y_ptr, x_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    s_cols = tl.num_programs(0)
    offset_x = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_y = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offset = offset_y[:, None] * N + offset_x[None, :]
    mask = (offset_y[:, None] < M) & (offset_x[None, :] < N)

    data_x = tl.load(x_ptr + offset, mask = mask, other = 0.0)
    data_s = tl.load(s_ptr + pid_y * s_cols + pid_x)
    data_y = data_x * data_s
    tl.store(y_ptr + offset, data_y, mask = mask)

# X, S, Y are tensors on the GPU
def solve(X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, TILE_SIZE: int):
    grid_size = (triton.cdiv(N, TILE_SIZE), triton.cdiv(M, TILE_SIZE))
    weightDequantizationKernel[grid_size](Y, X, S, M, N, BLOCK_SIZE = TILE_SIZE)