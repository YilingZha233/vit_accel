"""Arithmetic and byte images for matmul-accel commit 966e3459.

These model the source's intended arithmetic and RAM layout, not bus timing.
All offsets are accelerator-local BYTES, never HPS physical addresses.
"""

ARRAY = 16
W_DEPTH, X_DEPTH, Y_DEPTH = 1024, 256, 128
W_BASE, X_BASE, Y_BASE = 0x1000, 0x5000, 0x7000


def saturate(value):
    return max(-128, min(127, value))


def validate(w, x, q16):
    if not isinstance(q16, int) or not 0 <= q16 <= 65535:
        raise ValueError("Q0.16 scale must be an integer in [0,65535]")
    if not w or not w[0] or not x or not x[0]:
        raise ValueError("Matrices must be nonempty")
    k, n, m = len(w), len(w[0]), len(x[0])
    if len(x) != n or any(len(row) != n for row in w) or any(len(row) != m for row in x):
        raise ValueError("Expected W[K,N] and X[N,M]")
    if any(not isinstance(v, int) or not -128 <= v <= 127 for a in (w, x) for row in a for v in row):
        raise ValueError("Operands must be signed INT8 integers")
    # Conservative bound for signed int8 products, excluding bias.
    if n * 16384 > 2147483647:
        raise ValueError("Contraction may overflow INT32")
    return k, n, m


def hardware_shape(k, n, m):
    if min(k, n, m) <= 0 or n % ARRAY:
        raise ValueError("Positive dimensions required; pad N to a multiple of 16")
    if ((k + 15) // 16) * n > W_DEPTH:
        raise ValueError("Weight bank capacity exceeded")
    if (n // ARRAY) * m > X_DEPTH:
        raise ValueError("Input bank capacity exceeded")
    if ((k + 15) // 16) * m > Y_DEPTH:
        raise ValueError("Output bank capacity exceeded")


def full_accumulation(w, x, q16):
    """Full integer sum, then one arithmetic shift and INT8 saturation.

    Python uses unbounded integers as the independent golden accumulator.
    This is the proposed full-precision contract, not the current RTL contract.
    """
    k, n, m = validate(w, x, q16)
    return [[saturate((sum(w[r][p] * x[p][c] for p in range(n)) * q16) >> 16)
             for c in range(m)] for r in range(k)]


def rtl_compatibility(w, x, q16):
    """Each 16-term partial sum is shifted/clipped; each merge is clipped."""
    k, n, m = validate(w, x, q16)
    hardware_shape(k, n, m)
    y = [[0] * m for _ in range(k)]
    for start in range(0, n, ARRAY):
        for r in range(k):
            for c in range(m):
                partial = sum(w[r][p] * x[p][c] for p in range(start, start + ARRAY))
                term = saturate((partial * q16) >> 16)
                y[r][c] = saturate(y[r][c] + term)
    return y


def weight_offset(row, col, n):
    return (row % ARRAY) * W_DEPTH + (row // ARRAY) * n + col


def input_offset(row, col, m):
    return (row % ARRAY) * X_DEPTH + (row // ARRAY) * m + col


def output_offset(row, col, m):
    return (row % ARRAY) * Y_DEPTH + (row // ARRAY) * m + col


def pack_inputs(w, x):
    k, n, m = validate(w, x, 0)
    hardware_shape(k, n, m)
    weights, inputs = bytearray(ARRAY * W_DEPTH), bytearray(ARRAY * X_DEPTH)
    for r in range(k):
        for p in range(n):
            weights[weight_offset(r, p, n)] = w[r][p] & 255
    for p in range(n):
        for c in range(m):
            inputs[input_offset(p, c, m)] = x[p][c] & 255
    return bytes(weights), bytes(inputs)


def row_major_bytes(y):
    return bytes(v & 255 for row in y for v in row)


def pack_output(y):
    k, m = len(y), len(y[0])
    if any(len(row) != m for row in y) or ((k + 15) // 16) * m > Y_DEPTH:
        raise ValueError("Invalid output dimensions")
    result = bytearray(ARRAY * Y_DEPTH)
    for r in range(k):
        for c in range(m):
            result[output_offset(r, c, m)] = y[r][c] & 255
    return bytes(result)


def unpack_output(image, k, m):
    if len(image) != ARRAY * Y_DEPTH or min(k, m) <= 0 or ((k + 15) // 16) * m > Y_DEPTH:
        raise ValueError("Expected the full 2048-byte output-bank image and valid dimensions")
    def signed(v):
        return v - 256 if v >= 128 else v
    return [[signed(image[output_offset(r, c, m)]) for c in range(m)] for r in range(k)]
