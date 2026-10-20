import torch
import triton
import triton.language as tl

@triton.jit
def crossEntropyLossKernel(row_sum_ptr, in_ptr, label_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if (pid >= N):
        return
    acc = tl.zeros((BLOCK_SIZE,), dtype = tl.float32)
    in_col = in_ptr + pid * C
    label_idx = tl.load(label_ptr + pid)
    for col in range(0, C, BLOCK_SIZE):
        offset_col = col + tl.arange(0, BLOCK_SIZE)
        mask = offset_col < C
        data_in = tl.load(in_col + offset_col, mask = mask, other = -float('inf'))
        acc += tl.exp(data_in)
    value = tl.log(tl.sum(acc)) - tl.load(in_col + label_idx)
    tl.store(row_sum_ptr + pid, value)

@triton.jit
def averageSumKernel(out_ptr, in_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data_in = tl.load(in_ptr + offset, mask = mask, other = 0.0)
    value = tl.sum(data_in) / tl.cast(N, tl.float32)
    tl.atomic_add(out_ptr, value)

# logits, true_labels, loss are tensors on the GPU
def solve(logits: torch.Tensor, true_labels: torch.Tensor, loss: torch.Tensor, N: int, C: int):
    loss.zero_()
    partial_sum = torch.zeros((N,), device = logits.device, dtype = torch.float32)
    BLOCK_SIZE = 512
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    crossEntropyLossKernel[(N, )](partial_sum, logits, true_labels, N, C, BLOCK_SIZE)
    averageSumKernel[grid](loss, partial_sum, N, BLOCK_SIZE)

