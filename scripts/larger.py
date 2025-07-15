"""
https://cubed-dev.github.io/cubed/examples/basic-array-ops.html#adding-two-larger-arrays
"""

import cubed
import zarr

import cubed.array_api as xp

import cubed.random
from cubed.diagnostics import ProgressBar
from cubed.diagnostics.history import HistoryCallback
from cubed.diagnostics.timeline import TimelineVisualizationCallback


zarr.config.enable_gpu()


if __name__ == "__main__":
    # My GPU failed to allocate 200,000,000 byte chunk, so lets go smaller (more chunks).
    # 200MB chunks
    a = cubed.random.random((50000, 50000), chunks=(1000, 1000))
    b = cubed.random.random((50000, 50000), chunks=(1000, 1000))
    c = xp.add(a, b)

    # use store=None to write to temporary zarr
    with ProgressBar(), HistoryCallback(), TimelineVisualizationCallback():
        cubed.to_zarr(c, store=None)
