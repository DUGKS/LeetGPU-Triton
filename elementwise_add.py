import torch
import triton
import triton.language as tl
import time

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y;
    tl.store(output_ptr + offsets, output, mask=mask)
    # if pid == 0:
    #     tl.device_print("add_kernel: x = ", x)
    #     tl.device_print("add_kernel: y = ", y)
    #     tl.device_print("add_kernel: output = ", output)

def elementwise_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 256):
    assert x.shape == y.shape
    output = torch.empty_like(x)
    num_elements = x.numel()
    grid = (triton.cdiv(num_elements, block_size),)
    add_kernel[grid](x, y, output, num_elements, block_size)
    return output


N = 4096
idx = torch.arange(N, dtype = torch.float16, device='cuda')
h_a = idx * 0.001
h_b = (idx * 2 + 1.0) * 0.001
h_c = elementwise_add(h_a, h_b)
h_c_expected = h_a + h_b

if torch.allclose(h_c, h_c_expected, rtol = 1E-3, atol = 1E-3):
    print("results right!")
else:
    print("results wrong!")


# for i in range(N):
        # if i < 1000:
            # print(f"i = {i}, {h_a[i]} + {h_b[i]} = {h_c[i]}")
