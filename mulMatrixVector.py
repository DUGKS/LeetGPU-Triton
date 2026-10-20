import torch
import triton
import triton.language as tl

@triton.jit
def mulMatrixVectorKernel(output, value, col_indices, row_ptr, x, M, N, nnz, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_start = tl.load(row_ptr + pid)
    if pid + 1 < M:
        row_end = tl.load(row_ptr + pid + 1)
    else:
        row_end = nnz

    acc = tl.zeros((BLOCK_SIZE, ), dtype = tl.float32)

    for start in tl.range(row_start, row_end, BLOCK_SIZE):
        offset =  start + tl.arange(0, BLOCK_SIZE)
        mask = offset < row_end
        data_A = tl.load(value + offset, mask = mask, other = 0.0)
        offset_x = tl.load(col_indices + offset, mask = mask, other = 0.0)
        mask_x = offset_x < N
        data_x = tl.load(x + offset_x, mask = mask_x, other = 0.0)
        acc += data_A * data_x

    val = tl.sum(acc, axis = 0)
    tl.store(output + pid, val)


