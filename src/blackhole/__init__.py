"""Black Hole X-ray Outburst modelling package."""

import functools

import numpy


def get_xp(*args):
    """Return cupy if any arg is a CuPy array, else numpy."""
    try:
        import cupy
        for a in args:
            if isinstance(a, cupy.ndarray):
                return cupy
    except ImportError:
        pass
    return numpy


def gpu_jit(func=None, **kwargs):
    """Decorator that selects CUDA JIT, CPU JIT, or passthrough.

    Tries numba CUDA first, then numba CPU JIT, then falls back to
    a plain function (so tests run without numba installed).
    """
    def decorator(fn):
        try:
            from numba import jit as _jit

            try:
                from numba import cuda as _cuda
                if _cuda.is_available():
                    return _jit(target_backend="cuda", nopython=True, **kwargs)(fn)
            except Exception:
                pass
            return _jit(nopython=True, **kwargs)(fn)
        except ImportError:
            @functools.wraps(fn)
            def wrapper(*args, **kw):
                return fn(*args, **kw)
            return wrapper

    if func is not None:
        return decorator(func)
    return decorator
