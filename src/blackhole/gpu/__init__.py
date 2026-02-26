"""GPU-accelerated accretion disk physics via CuPy/NumPy array dispatch.

This subpackage provides pure array-op reimplementations of all physics
functions from the parent ``blackhole`` package.  Every function works
transparently with both NumPy and CuPy arrays — no numba dependency.

Key utilities
-------------
get_xp(*arrays)
    Return ``cupy`` if any input is a CuPy array, else ``numpy``.
to_device(arr)
    Move a NumPy array to the GPU (no-op without CuPy).
to_host(arr)
    Move an array to the CPU.
is_gpu_available()
    Check whether CuPy and a CUDA device are usable.
HAS_CUPY : bool
    ``True`` if CuPy was importable at startup.
"""

import numpy as np

try:
    import cupy as cp  # noqa: F401

    HAS_CUPY = True

    # On Windows, NVIDIA pip packages store CUDA DLLs in separate
    # site-packages/nvidia/*/bin/ directories.  Register them so that
    # transitive dependencies (cusparse, cusolver, etc.) resolve correctly.
    import os
    import sys

    if sys.platform == "win32":
        import importlib.util

        _spec = importlib.util.find_spec("nvidia")
        if _spec and _spec.submodule_search_locations:
            for _nvidia_root in _spec.submodule_search_locations:
                for _child in os.listdir(_nvidia_root):
                    _bin = os.path.join(_nvidia_root, _child, "bin")
                    if os.path.isdir(_bin) and any(
                        f.endswith(".dll") for f in os.listdir(_bin)
                    ):
                        os.add_dll_directory(_bin)
        del _spec
    del os, sys

except ImportError:
    HAS_CUPY = False


def get_xp(*arrays):
    """Return the array module (``cupy`` or ``numpy``) for *arrays*.

    If *any* array is a CuPy array the function returns ``cupy`` so that
    all downstream operations stay on the GPU.  Falls back to ``numpy``.
    """
    if HAS_CUPY:
        import cupy  # local import keeps startup fast when unused

        for a in arrays:
            if isinstance(a, cupy.ndarray):
                return cupy
    return np


def to_device(arr):
    """Move *arr* to the GPU.  No-op if CuPy is unavailable."""
    if HAS_CUPY:
        import cupy

        return cupy.asarray(arr)
    return np.asarray(arr)


def to_host(arr):
    """Move *arr* to the CPU (handles both CuPy and NumPy arrays)."""
    if HAS_CUPY:
        import cupy

        if isinstance(arr, cupy.ndarray):
            return arr.get()
    return np.asarray(arr)


def is_gpu_available():
    """Return ``True`` if CuPy is installed and a CUDA device is usable."""
    if not HAS_CUPY:
        return False
    try:
        import cupy

        cupy.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False
