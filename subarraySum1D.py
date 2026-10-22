import torch
import triton
import triton.language as tl

@triton.jit
def subarraySum(output, input, S, length, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros((BLOCK_SIZE, ), dtype = tl.int32)

    for start in range(pid * BLOCK_SIZE, length, nprog * BLOCK_SIZE):
        offset = start + tl.arange(0, BLOCK_SIZE)
        mask = offset < length
        data_in = tl.load(input + S + offset, mask = mask, other = 0)
        acc += data_in
    tl.atomic_add(output, tl.sum(acc, axis = 0))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):
    BLOCK_SIZE = 2048
    length = E - S + 1
    output.zero_()
    grid = min(1024, triton.cdiv(length, BLOCK_SIZE))
    subarraySum[(grid, )](output, input, S, length, BLOCK_SIZE)