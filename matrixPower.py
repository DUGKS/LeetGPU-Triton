import torch
import triton
import triton.language as tl

@triton.jit
def matrixMultiplyKernel(C, A, B, N, BLOCK: tl.constexpr):
    pid_x = tl.program_id(1)
    pid_y = tl.program_id(0)

    offset_col = pid_x * BLOCK + tl.arange(0, BLOCK)
    offset_row = pid_y * BLOCK + tl.arange(0, BLOCK)
    offset_out = offset_row[:, None] * N + offset_col[None, :]

    mask_col = offset_col < N
    mask_row = offset_row < N
    mask_out = mask_row[:, None] & mask_col[None, :]

    acc = tl.zeros((BLOCK, BLOCK), dtype = tl.float32)
    for k in range(0, N, BLOCK):
        offset_k = k + tl.arange(0, BLOCK)

        offset_A = offset_row [:, None] * N + offset_k[None, :]
        mask_A = mask_row[:, None] & (offset_k[None, :] < N)
        data_A = tl.load(A + offset_A, mask = mask_A, other = 0.0)

        offset_B = offset_k[:, None] * N + offset_col[None, :]
        mask_B = (offset_k[:, None] < N) & mask_col[None, :]
        data_B = tl.load(B + offset_B, mask = mask_B, other = 0.0)

        acc += tl.dot(data_A, data_B, input_precision="ieee")
    tl.store(C + offset_out, acc, mask = mask_out)

@triton.jit
def identityKernel(C, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N * N
    row = offset // N
    col = offset % N
    val = tl.where(row == col, 1.0, 0.0)

    tl.store(C + offset, val, mask = mask)

def matmul(C, A, B, N, BLOCK = 32):
    grid = (triton.cdiv(N, BLOCK), triton.cdiv(N, BLOCK))
    matrixMultiplyKernel[grid](C, A, B, N, BLOCK = BLOCK)

def identity(C, N, BLOCK = 1024):
    grid = (triton.cdiv(N * N, BLOCK), )
    identityKernel[grid](C, N, BLOCK = BLOCK)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, P: int):
    if P == 0:
        identity(output, N)
        return
    if P == 1:
        output.copy_(input)
        return

    base = input.contiguous().clone()
    result = None
    tmp = torch.empty_like(base)

    p = P
    while p > 0:
        if (p & 1):
            if result is None:
                result = base.clone()
            else:
                matmul(tmp, result, base, N)
                result, tmp = tmp, result
        p >>= 1
        if (p > 0):
            matmul(tmp, base, base, N)
            base, tmp = tmp, base

    output.copy_(result)