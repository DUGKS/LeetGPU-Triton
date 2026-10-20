import torch
import triton
import triton.language as tl
import time

@triton.jit
def multKernel(a, b, c, M, K, N, BLOCK_SIZE_K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    if row >= M or col >= N:
        return

    offset_k = tl.arange(0, BLOCK_SIZE_K)
    result = 0.0;
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offset = k * BLOCK_SIZE_K + offset_k
        a_row = tl.load(a + row * K + offset, mask = offset < K, other = 0.0)
        b_col = tl.load(b + offset * N + col, mask = offset < K, other = 0.0)
        result += tl.sum(a_row * b_col)

    tl.store(c + row * N + col, result)

# A: M x K    B: K x N    C: M x N
@triton.jit
def multTiledKernel(a, b, c, M, K, N,
                              stride_am, stride_ak,
                              stride_bk, stride_bn,
                              stride_cm, stride_cn,
                              BLOCK_SIZE_M: tl.constexpr,
                              BLOCK_SIZE_N: tl.constexpr,
                              BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    range_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a + offset_m[:, None] * stride_am + range_k[None, :] * stride_ak
    b_ptrs = b + range_k[:, None] * stride_bk + offset_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offset_k = k * BLOCK_SIZE_K + range_k
        a_block = tl.load(a_ptrs, mask = (offset_m[:, None] < M) & (offset_k[None, :] < K), other = 0.0)
        b_block = tl.load(b_ptrs, mask = (offset_k[:, None] < K) & (offset_n[None, :] < N), other = 0.0)
        acc += tl.dot(a_block, b_block, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask = (offset_m[:, None] < M) & (offset_n[None, :] < N))

if __name__ == "__main__":
    M, K, N = 128, 256, 128
    torch.manual_seed(0)
    a = torch.randn((M, K), device='cuda', dtype = torch.float32)
    b = torch.randn((K, N), device='cuda', dtype = torch.float32)

    c_plain = torch.zeros((M, N), device = 'cuda', dtype = torch.float32)
    c_tiled = torch.zeros((M, N), device = 'cuda', dtype = torch.float32)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c_plain.stride()

    BLOCK_SIZE_K = 32
    grid_plain = (M, N)
    multKernel[grid_plain](a, b, c_plain, M, K, N, BLOCK_SIZE_K)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid_tiled = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    multTiledKernel[grid_tiled](a, b, c_tiled, M, K, N, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    c_ref = torch.matmul(a, b)

    diff_plain_ref = torch.max(torch.abs(c_plain - c_ref)).item()
    diff_tiled_ref = torch.max(torch.abs(c_tiled - c_ref)).item()

    print(f"plain kernel maximum error: {diff_plain_ref}")
    print(f"tiled kernel maximum error: {diff_tiled_ref}")
