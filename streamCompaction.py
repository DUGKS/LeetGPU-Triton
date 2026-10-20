import torch
import triton
import triton.language as tl

@triton.jit
def countKernel(counts, A, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    x = tl.load(A + offset, mask = mask, other = 0.0)
    c = tl.sum((x > 0).to(tl.int32), axis = 0)
    tl.store(counts + pid, c)

@triton.jit
def scanKernel(counts, ex_pre_sum, num_blocks, CHUNK: tl.constexpr):
    running = tl.zeros((1,), dtype = tl.int32)
    for start in range(0, num_blocks, CHUNK):
        offset = start + tl.arange(0, CHUNK)
        mask = offset < num_blocks
        x = tl.load(counts + offset, mask = mask, other = 0)
        inc_sum = tl.cumsum(x, axis = 0)
        tl.store(ex_pre_sum + offset, running + inc_sum - x, mask = mask)
        running += tl.sum(x, axis = 0)

@triton.jit
def scatterKernel(out, A, ex_pre_sum, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask_in = offset < N
    x = tl.load(A + offset, mask = mask_in, other = 0.0)
    flag = (x > 0.0) & mask_in
    f = flag.to(tl.int32)
    ex_sum = tl.cumsum(f, axis = 0) - f
    base = tl.load(ex_pre_sum + pid)
    pos = base + ex_sum
    tl.store(out + pos, x, mask = flag)

# A, out are tensors on the GPU
def solve(A: torch.Tensor, N: int, out: torch.Tensor):
    BLOCK_SIZE = 2048
    grid = triton.cdiv(N, BLOCK_SIZE)
    counts = torch.empty((grid,), dtype = torch.int32, device = A.device)
    ex_pre_sum = torch.empty((grid,), dtype = torch.int32, device = A.device)

    countKernel[(grid, )](counts, A, N, BLOCK_SIZE)
    scanKernel[(1, )](counts, ex_pre_sum, grid, BLOCK_SIZE)
    scatterKernel[(grid, )](out, A, ex_pre_sum, N, BLOCK_SIZE)