# https://cubed-dev.github.io/cubed/examples/basic-array-ops.html#matmul
import cupy

# the array API uses bool, not bool_
# TODO: Check on status in cupy

cupy.bool = cupy.bool_

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.diagnostics.history import HistoryCallback
from cubed.diagnostics.rich import RichProgressBar
from cubed.diagnostics.timeline import TimelineVisualizationCallback
import zarr

zarr.config.enable_gpu()

if __name__ == "__main__":
    a = cubed.random.random((25000, 25000), chunks=(1000, 1000))
    b = cubed.random.random((25000, 25000), chunks=(1000, 1000))
    c = xp.matmul(a, b)

    progress = RichProgressBar()
    hist = HistoryCallback()
    timeline_viz = TimelineVisualizationCallback()
    # use store=None to write to temporary zarr
    cubed.to_zarr(
        c,
        store=None,
        callbacks=[progress, hist, timeline_viz],
    )
