import torch
import triton
import triton.language as tl

@triton.jit
def rowScanKernel(out, input, H, W, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * W
    run = 0.0
    for start in range(0, W, BLOCK):
        offset = start + tl.arange(0, BLOCK)
        mask = offset < W
        data_in = tl.load(input + base + offset, mask = mask, other = 0.0)
        data_out = tl.cumsum(data_in, axis = 0) + run
        run = run + tl.sum(data_in, axis = 0)
        tl.store(out + base + offset, data_out, mask = mask)

@triton.jit
def colScanKernel(out, H, W, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < W
    acc = tl.zeros((BLOCK, ), dtype = tl.float32)
    for row in range(0, H):
        data_in = tl.load(out + row * W + cols, mask = mask, other = 0.0)
        acc += data_in
        tl.store(out + row * W + cols, acc, mask = mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, H: int, W: int):
    BLOCK = 2048
    rowScanKernel[(H, )](output, input, H, W, BLOCK = BLOCK)
    COLS = 32
    colScanKernel[(triton.cdiv(W, COLS),)](output, H, W, BLOCK = COLS)