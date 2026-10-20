import torch
import triton
import triton.language as tl

@triton.jit
def GaussianBlurKernel(output_ptr, input_ptr, kernel_ptr, input_rows, input_cols, kernel_rows, kernel_cols, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    row_start = pid_y * BLOCK_SIZE
    col_start = pid_x * BLOCK_SIZE
    halo_rows = kernel_rows // 2
    halo_cols = kernel_cols // 2

    offset_row = row_start + tl.arange(0, BLOCK_SIZE)
    offset_col = col_start + tl.arange(0, BLOCK_SIZE)
    offset_out = offset_row[:, None] * input_cols + offset_col[None, :]

    mask_row = (offset_row >= 0) & (offset_row < input_rows)
    mask_col = (offset_col >= 0) & (offset_col < input_cols)
    mask_out = mask_row[:, None] & mask_col[None, :]

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype = tl.float32)
    for jy in range(0, kernel_rows):
        for ix in range(0, kernel_cols):
            weight = tl.load(kernel_ptr + jy * kernel_cols + ix)
            offset_row_in = offset_row - halo_rows + jy
            offset_col_in = offset_col - halo_cols + ix
            offset = offset_row_in[:, None] * input_cols + offset_col_in[None, :]
            mask_row_in = (offset_row_in >= 0) & (offset_row_in < input_rows)
            mask_col_in = (offset_col_in >= 0) & (offset_col_in < input_cols)
            mask_in = mask_row_in[:, None] & mask_col_in[None, :]
            data_in = tl.load(input_ptr + offset, mask = mask_in, other = 0.0)
            acc += weight * data_in
    tl.store(output_ptr + offset_out, acc, mask = mask_out)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    tile_size = 32
    grid = (triton.cdiv(input_cols, tile_size), triton.cdiv(input_rows, tile_size))
    GaussianBlurKernel[grid](output, input, kernel, input_rows, input_cols, kernel_rows, kernel_cols, tile_size)