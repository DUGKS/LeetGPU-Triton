import torch
import triton
import triton.language as tl

@triton.jit
def subarraySum(output, input, N, M, S_row, length_row, S_col, length_col, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    nprog_x = tl.num_programs(0)
    nprog_y = tl.num_programs(1)
    acc = tl.zeros((BLOCK_Y, BLOCK_X), dtype = tl.int32)

    for start_row in range(pid_y * BLOCK_Y, length_row, nprog_y * BLOCK_Y):
        offset_y = start_row + tl.arange(0, BLOCK_Y)
        for start_col in range(pid_x * BLOCK_X, length_col, nprog_x * BLOCK_X):
            offset_x = start_col + tl.arange(0, BLOCK_X)
            offset = offset_y[:, None] * M + offset_x[None, :]
            mask = (offset_y[:, None] < length_row) & (offset_x[None, :] < length_col)
            data_in = tl.load(input + S_row * M + S_col + offset, mask = mask, other = 0)
            acc += data_in
    tl.atomic_add(output, tl.sum(acc))

# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    S_ROW: int,
    E_ROW: int,
    S_COL: int,
    E_COL: int,
):
    BLOCK_X = 128
    BLOCK_Y = 16
    length_row = E_ROW - S_ROW + 1
    length_col = E_COL - S_COL + 1
    grid = (triton.cdiv(length_col, BLOCK_X), triton.cdiv(length_row, BLOCK_Y))
    subarraySum[grid](output, input, N, M, S_ROW, length_row, S_COL, length_col, BLOCK_X, BLOCK_Y)