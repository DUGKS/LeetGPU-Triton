import torch
import triton
import triton.language as tl

@triton.jit
def subarraySum(output, input, N, M, K, S_dep, length_depth, S_row, length_row, S_col, length_col, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr, BLOCK_Z: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    nprog_x = tl.num_programs(0)
    nprog_y = tl.num_programs(1)
    nprog_z = tl.num_programs(2)
    input_start = input + (S_dep * M + S_row) * K + S_col
    acc = tl.zeros((BLOCK_Z, BLOCK_Y, BLOCK_X), dtype = tl.int32)
    for start_depth in range(pid_z * BLOCK_Z, length_depth, nprog_z * BLOCK_Z):
        offset_z = start_depth + tl.arange(0, BLOCK_Z)
        for start_row in range(pid_y * BLOCK_Y, length_row, nprog_y * BLOCK_Y):
            offset_y = start_row + tl.arange(0, BLOCK_Y)
            for start_col in range(pid_x * BLOCK_X, length_col, nprog_x * BLOCK_X):
                offset_x = start_col + tl.arange(0, BLOCK_X)
                offset = (offset_z[:, None, None] * M + offset_y[None, :, None]) * K + offset_x[None, None, :]
                mask = (offset_z[:, None, None] < length_depth) & (offset_y[None, :, None] < length_row) & (offset_x[None, None, :] < length_col)
                data_in = tl.load(input_start + offset, mask = mask, other = 0)
                acc += data_in
    tl.atomic_add(output, tl.sum(acc))

# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    K: int,
    S_DEP: int,
    E_DEP: int,
    S_ROW: int,
    E_ROW: int,
    S_COL: int,
    E_COL: int,
):
    BLOCK_X = 32
    BLOCK_Y = 16
    BLOCK_Z = 8
    length_depth = E_DEP - S_DEP + 1
    length_row = E_ROW - S_ROW + 1
    length_col = E_COL - S_COL + 1
    grid = (triton.cdiv(length_col, BLOCK_X), triton.cdiv(length_row, BLOCK_Y), triton.cdiv(length_depth, BLOCK_Z))
    subarraySum[grid](output, input, N, M, K, S_DEP, length_depth, S_ROW, length_row, S_COL, length_col, BLOCK_X, BLOCK_Y, BLOCK_Z)