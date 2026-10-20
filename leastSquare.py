import torch
import triton
import triton.language as tl

@triton.jit
def gramKernel(A, X, M, N, BI: tl.constexpr, BJ: tl.constexpr, BS: tl.constexpr):
    pid_i = tl.program_id(0) # y
    pid_j = tl.program_id(1) # x
    offset_i = pid_i * BI + tl.arange(0, BI)
    offset_j = pid_j * BJ + tl.arange(0, BJ)
    mask_i = offset_i < N
    mask_j = offset_j < N
    mask_A = mask_i[:, None] & mask_j[None, :]

    acc = tl.zeros((BI, BJ), dtype = tl.float32)
    for start in range(0, M, BS):
        offset_k = start + tl.arange(0, BS)
        mask_k = offset_k < M

        offset_x = offset_k[:, None] * N + offset_j[None, :]
        mask_x = mask_k[:, None] & mask_j[None, :]
        data_x = tl.load(X + offset_x, mask = mask_x, other = 0.0)

        offset_xT = offset_k[:, None] * N + offset_i[None, :]
        mask_xT = mask_k[:, None] & mask_i[None, :]
        data_xT = tl.load(X + offset_xT, mask = mask_xT, other = 0.0)

        acc += tl.dot(tl.trans(data_xT), data_x, input_precision = "ieee")

    out = A + offset_i[:, None] * (N + 1) + offset_j[None, :]
    tl.store(out, acc.to(tl.float32), mask = mask_A)

@triton.jit
def XTyKernel(A, X, y, M, N, BI: tl.constexpr, BS: tl.constexpr):
    pid = tl.program_id(0)
    offset_i = pid * BI + tl.arange(0, BI)
    mask_i = offset_i < N

    acc = tl.zeros((BI, ), dtype = tl.float32)
    for start in range(0, M, BS):
        offset_m = start + tl.arange(0, BS)
        mask_m = offset_m < M
        data_y = tl.load(y + offset_m, mask = mask_m, other = 0.0)

        offset_x = offset_m[:, None] * N + offset_i[None, :]
        mask_x = mask_m[:, None] & mask_i[None, :]
        data_x = tl.load(X + offset_x, mask = mask_x, other = 0.0)

        acc += tl.sum(data_x * data_y[:, None], axis = 0)
    offset_A = offset_i * (N + 1) + N
    tl.store(A + offset_A, acc, mask = mask_i)

@triton.jit
def GaussJordanEliminateKernel(A_out, A_in, N, BI: tl.constexpr, BJ: tl.constexpr):
    pid_i = tl.program_id(0)  # y
    pid_j = tl.program_id(1)  # x
    offset_i =