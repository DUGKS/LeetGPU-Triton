import torch
import triton
import triton.language as tl

@triton.jit
def nearestNeighbor(output, input, N, BLOCK_Q: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    offset_q = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_q = offset_q < N

    qx = tl.load(input + offset_q * 3 + 0, mask = mask_q, other = 0.0)
    qy = tl.load(input + offset_q * 3 + 1, mask = mask_q, other = 0.0)
    qz = tl.load(input + offset_q * 3 + 2, mask = mask_q, other = 0.0)

    best_val = tl.full((BLOCK_Q, ), float('inf'), dtype = tl.float32)
    best_idx = tl.full((BLOCK_Q, ), -1, dtype = tl.int32)
    for base in range(0, N, BLOCK_C):
        offset_c = base + tl.arange(0, BLOCK_C)
        mask_c = offset_c < N
        cx = tl.load(input + offset_c * 3 + 0, mask = mask_c, other = 0.0)
        cy = tl.load(input + offset_c * 3 + 1, mask = mask_c, other = 0.0)
        cz = tl.load(input + offset_c * 3 + 2, mask = mask_c, other = 0.0)

        dx = qx[:, None] - cx[None, :]
        dy = qy[:, None] - cy[None, :]
        dz = qz[:, None] - cz[None, :]
        d = dx * dx + dy * dy + dz * dz
        valid = mask_c[None, :] & (offset_c[None, :] != offset_q[:, None])
        d = tl.where(valid, d, float('inf'))
        tile_min = tl.min(d, axis = 1)
        tile_arg = tl.argmin(d, axis = 1)
        better = tile_min < best_val
        best_val = tl.where(better, tile_min, best_val)
        best_idx = tl.where(better, base + tile_arg, best_idx)

    tl.store(output + offset_q, best_idx, mask_q)





# points and indices are tensors on the GPU
def solve(points: torch.Tensor, indices: torch.Tensor, N: int):
    block_q = 64
    block_c = 64
    grid = (N + block_q - 1) // block_q
    nearestNeighbor[(grid, )](indices, points, N, block_q, block_c)