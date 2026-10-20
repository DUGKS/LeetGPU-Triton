import torch
import triton
import triton.language as tl
import time

@triton.jit
def matrixTranspose(output, input, num_row, num_col, stride_in_row, stride_in_col, stride_out_row, stride_out_col, BLOCK_SIZE: tl.constexpr):

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    offset_row = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_col = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    input_ptrs = input + offset_row[:, None] * stride_in_row + offset_col[None, :] * stride_in_col
    mask = (offset_row[:, None] < num_row) & (offset_col[None, :] < num_col)
    in_block = tl.load(input_ptrs, mask = mask, other = 0.0)

    out_block = tl.trans(in_block)
    output_ptrs = output + offset_col[:, None] * stride_out_row + offset_row[None, :] * stride_out_col
    mask = (offset_col[:, None] < num_col) & (offset_row[None, :] < num_row)
    tl.store(output_ptrs, out_block, mask)

if __name__ == "__main__":
    num_row = 1024
    num_col = 512
    BLOCK_SIZE = 32
    torch.manual_seed(42)

    input = torch.randn((num_row, num_col), dtype = torch.float32, device = 'cuda')
    output = torch.empty((num_col, num_row), dtype = torch.float32, device = 'cuda')

    grid = ((num_row + BLOCK_SIZE - 1) // BLOCK_SIZE, (num_col + BLOCK_SIZE - 1) // BLOCK_SIZE)
    matrixTranspose[grid](output, input, num_row, num_col, input.stride(0), input.stride(1), output.stride(0), output.stride(1), BLOCK_SIZE)

    output_ref = input.T.contiguous()
    print("Pass: ", torch.allclose(output, output_ref))