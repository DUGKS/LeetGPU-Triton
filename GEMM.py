import torch
import triton
import triton.language as tl

@triton.jit
def GEMM(C, A, B, M, K, N, alpha, beta, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offset_N = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_M = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_N = offset_N < N
    mask_M = offset_M < M

    offset_C = offset_M[:, None] * N + offset_N[None, :]
    mask_C = mask_M[:, None] & mask_N[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    for start in range(0, K, BLOCK_K):
        offset_K = start + tl.arange(0, BLOCK_K)
        mask_K = offset_K < K

        offset_A = offset_M[:, None] * K + offset_K[None, :]
        mask_A = mask_M[:, None] & mask_K[None, :]
        data_A = tl.load(A + offset_A, mask = mask_A, other = 0.0)

        offset_B = offset_K[:, None] * N + offset_N[None, :]
        mask_B = mask_K[:, None] & mask_N[None, :]
        data_B = tl.load(B + offset_B, mask = mask_B, other = 0.0)
        acc += tl.dot(data_A, data_B)

    data_C = tl.load(C + offset_C, mask = mask_C, other = 0.0)
    result = alpha * acc + beta * data_C
    tl.store(C + offset_C, result.to(tl.float16), mask = mask_C)

# a, b, c are tensors on the GPU
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    GEMM[grid](c, a, b, M, K, N, alpha, beta, BLOCK_M, BLOCK_K, BLOCK_N)