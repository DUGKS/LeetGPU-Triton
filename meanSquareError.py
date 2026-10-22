import torch
import triton
import triton.language as tl

@triton.jit
def meanSquareError(mse_ptr, pre_ptr, targets_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    data_pre = tl.load(pre_ptr + offset, mask = mask, other = 0.0)
    data_target = tl.load(targets_ptr + offset, mask = mask, other = 0.0)

    data_mse = (data_pre - data_target) * (data_pre - data_target)
    tl.atomic_add(mse_ptr, tl.sum(data_mse, axis = 0) / N)

# predictions, targets, mse are tensors on the GPU
def solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    BLOCK_SIZE = 2048
    grid = triton.cdiv(N, BLOCK_SIZE)
    meanSquareError[(grid, )](mse, predictions, targets, N, BLOCK_SIZE = BLOCK_SIZE)