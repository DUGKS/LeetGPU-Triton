import torch
import triton
import triton.language as tl

@triton.jit
def JacobiStencil(output, input, rows, cols, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    offset_row = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]
    offset_col = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[None, :]

    in_bounds = (offset_row < rows) & (offset_col < cols)
    interior = (offset_row > 0) & (offset_row < rows - 1) & (offset_col > 0) & (offset_col < cols - 1)

    pos = offset_row * cols + offset_col
    up = tl.load(input + pos - cols, mask = interior, other = 0.0)
    down = tl.load(input + pos + cols, mask = interior, other = 0.0)
    left = tl.load(input + pos - 1, mask = interior, other = 0.0)
    right = tl.load(input + pos + 1, mask = interior, other = 0.0)
    edge = tl.load(input + pos, mask = ~interior & in_bounds, other = 0.0)

    data_out = tl.where(interior, 0.25 * (up + down + left + right), edge)
    tl.store(output + pos, data_out, mask = in_bounds)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK_X = 128
    BLOCK_Y = 16
    grid = (triton.cdiv(cols, BLOCK_X), triton.cdiv(rows, BLOCK_Y))
    JacobiStencil[grid](output, input, rows, cols, BLOCK_X, BLOCK_Y)