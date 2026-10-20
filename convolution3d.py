import torch
import triton
import triton.language as tl

@triton.jit
def convolution3DKernel(output, input, kernel, input_depth, input_rows, input_cols,
                        kernel_depth: tl.constexpr,
                        kernel_rows: tl.constexpr,
                        kernel_cols: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    base_depth = pid_z * BLOCK_SIZE
    base_row = pid_y * BLOCK_SIZE
    base_col = pid_x * BLOCK_SIZE
    num_kernel = kernel_depth * kernel_rows * kernel_cols

    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    offset_depth = base_depth + tl.arange(0, BLOCK_SIZE)
    offset_row = base_row + tl.arange(0, BLOCK_SIZE)
    offset_col = base_col + tl.arange(0, BLOCK_SIZE)

    offset_output = offset_depth[:, None, None] * output_rows * output_cols + offset_row[None, :, None] * output_cols + offset_col[None, None, :]
    mask_output = (offset_depth[:, None, None] < output_depth) & (offset_row[None, :, None] < output_rows) & (offset_col[None, None, :] < output_cols)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), dtype = tl.float32)

    for k in range(0, kernel_depth):
        for j in range(0, kernel_rows):
            for i in range(0, kernel_cols):
                kernel_idx = k * kernel_rows * kernel_cols + j * kernel_cols + i
                mask_kernel = kernel_idx < num_kernel
                weight = tl.load(kernel + kernel_idx, mask = mask_kernel, other = 0.0)

                offset_z = offset_depth + k
                offset_y = offset_row + j
                offset_x = offset_col + i

                offset_input = offset_z[:, None, None] * input_rows * input_cols + offset_y[None, :, None] * input_cols + offset_x[None, None, :]
                mask_input = (offset_z[:, None, None] < input_depth) & (offset_y[None, :, None] < input_rows) & (offset_x[None, None, :] < input_cols)
                data_in = tl.load(input + offset_input, mask = mask_input, other = 0.0)
                acc += weight * data_in

    tl.store(output + offset_output, acc, mask = mask_output)