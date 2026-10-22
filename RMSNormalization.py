import torch
import triton
import triton.language as tl

@triton.jit
def sumKernel(output, input, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    pid_start = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE
    acc = tl.zeros((BLOCK_SIZE, ), dtype = tl.float32)
    for start in range(pid_start, N, stride):
        offset = start + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        data_in = tl.load(input + offset, mask = mask, other = 0.0)
        acc += data_in * data_in
    tl.atomic_add(output, tl.sum(acc, axis = 0) / N)

@triton.jit
def RMSKernel(output, input, N, rms, gamma, beta, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pid_start = pid * BLOCK_SIZE
    offset = pid_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    val = tl.sqrt(tl.load(rms) + eps)
    data_in = tl.load(input + offset, mask = mask, other = 0.0)
    data_out = data_in * gamma / val + beta
    tl.store(output + offset, data_out, mask = mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, N: int, eps: float):
    BLOCK_SIZE = 1024
    grid = triton.cdiv(N, BLOCK_SIZE)
    rms = torch.zeros(1, dtype = torch.float32, device = input.device)
    sumKernel[(grid,)](rms, input, N, BLOCK_SIZE)
    RMSKernel[(grid,)](output, input, N, rms, gamma, beta, eps, BLOCK_SIZE)
