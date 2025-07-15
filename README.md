# cubed + cupy

Some experiments at running cubed with a GPU (via cupy).

## Issues

- cupy's main namespace isn't compatible with the array-api spec? (`cupy.bool`). Need to check on that.
- cubed casts to a numpy array, assuming it's cheap. We ~never want to do that with GPU data.

Hence the changes at https://github.com/TomAugspurger/cubed/tree/twa/cupy-compat

## Overview

1. set `CUBED_BACKEND_ARRAY_API_MODULE`
2. call `zarr.enable_gpu`

and stuff seems to work?