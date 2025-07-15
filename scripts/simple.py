"""
The most basic example

https://cubed-dev.github.io/cubed/examples/basic-array-ops.html

This required patching backend_array_api to not convert to numpy.
"""

import zarr

import cubed.array_api as xp

zarr.config.enable_gpu()


if __name__ == "__main__":
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
    )
    b = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
    )
    c = xp.add(a, b)
    res = c.compute()
    print(res)
