import torch
import triton
import triton.language as tl

@triton.jit
def comb(f1, v1, f2, v2):
    return f1 | f2, tl.where(f2 != 0, v2, v1 + v2)

@triton.jit
def blockAggregateKernel(agg_value, agg_flag, values_ptr, flags_ptr, N, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * TILE + tl.arange(0, TILE)
    mask = offset < N
    v = tl.load(values_ptr + offset, mask = mask, other = 0.0)
    f = tl.load(flags_ptr + offset, mask = mask, other = 0)
    ff, vv = tl.associative_scan((f, v), 0, comb)
    last = TILE - 1
    valid = (tl.arange(0, TILE) == last)
    tl.store(agg_value + pid, tl.sum(tl.where(valid, vv, 0.0), 0))
    tl.store(agg_flag + pid, tl.sum(tl.where(valid, ff, 0), 0))

@triton.jit
def carryKernel(carry, agg_value, agg_flag, num, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    tl.store(carry, 0.0)
    run = 0.0
    for start in range(0, num, BLOCK):
        offset = start + off
        mask = offset < num
        f = tl.load(agg_flag + offset, mask = mask, other = 0)
        v = tl.load(agg_value + offset, mask = mask, other = 0.0)
        ff, vv = tl.associative_scan((f, v), 0, comb)
        incl_value = tl.where(ff != 0, vv, run + vv)
        tl.store(carry + offset + 1, incl_value, mask = (offset + 1) < num)
        last = tl.minimum(BLOCK, num - start) - 1
        run = tl.sum(tl.where(off == last, incl_value, 0.0), 0)

@triton.jit
def offsetKernel(output, values, flags, carry, N, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * TILE + tl.arange(0, TILE)
    mask = offset < N
    v = tl.load(values + offset, mask = mask, other = 0.0)
    f = tl.load(flags + offset, mask = mask, other = 0)

    ff, vv = tl.associative_scan((f, v), 0, comb)
    carry_value = tl.load(carry + pid)
    incl_value = tl.where(ff == 0, carry_value + vv, vv)

    out = tl.where(f != 0, 0.0, incl_value - v)
    tl.store(output + offset, out, mask = mask)

# values, flags, output are tensors on the GPU
def solve(values: torch.Tensor, flags: torch.Tensor, output: torch.Tensor, N: int):
    TILE = 1024
    block_nums = triton.cdiv(N, TILE)

    agg_flags = torch.empty((block_nums, ), dtype = torch.int32, device = values.device)
    agg_values = torch.empty((block_nums, ), dtype = torch.float32, device = values.device)
    carry = torch.empty((block_nums, ), dtype = torch.float32, device = values.device)

    blockAggregateKernel[(block_nums, )](agg_values, agg_flags, values, flags, N, TILE)
    carryKernel[(1,)](carry, agg_values, agg_flags, block_nums, 2048)
    offsetKernel[(block_nums, )](output, values, flags, carry, N, TILE)
