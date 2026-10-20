import triton
import triton.language as tl

@triton.jit
def invert_kernel(image, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < width * height
    image32 = tl.cast(image, tl.pointer_type(tl.int32))
    data = tl.load(image32 + offsets, mask = mask, other = 0.0)
    b = data & 0xFF
    g = (data >> 8) & 0xFF
    r = (data >> 16) & 0xFF
    a = (data >> 24) & 0xFF

    b = 255 - b
    g = 255 - g
    r = 255 - r
    result = b | g << 8 | r << 16 | a << 24
    tl.store(image32 + offsets, result, mask = mask)