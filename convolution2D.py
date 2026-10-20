import torch
import triton
import triton.language as tl

@triton.jit
def convolution2DKernel(output, input, kernel, input_rows, input_cols, kernel_rows: tl.constexpr, kernel_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    base_row = pid_y * BLOCK_SIZE
    base_col = pid_x * BLOCK_SIZE
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1


    offset_output_col = base_col + tl.arange(0, BLOCK_SIZE)
    offset_output_row = base_row + tl.arange(0, BLOCK_SIZE)
    offset_output = offset_output_row[:, None] * output_cols + offset_output_col[None, :]
    mask_output = (offset_output_row[:, None] < output_rows) & (offset_output_col[None, :] < output_cols)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype = tl.float32)
    for row_k in range(kernel_rows):
        for col_k in range(kernel_cols):
            weight = tl.load(kernel + row_k * kernel_cols + col_k)
            offset_input_row = base_row + row_k + tl.arange(0, BLOCK_SIZE)
            offset_input_col = base_col + col_k + tl.arange(0, BLOCK_SIZE)
            offset_input = offset_input_row[:, None] * input_cols + offset_input_col[None, :]
            mask_input = (offset_input_row[:, None] < input_rows) & (offset_input_col[None, :] < input_cols)
            data_input = tl.load(input + offset_input, mask = mask_input, other = 0.0)
            acc += weight * data_input

    tl.store(output + offset_output, acc, mask = mask_output)

