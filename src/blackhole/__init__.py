"""Black Hole X-ray Outburst modelling package."""

import functools


def cpu_jit(func=None, **kwargs):
    """CPU-only numba JIT. Falls back to passthrough without numba."""
    def decorator(fn):
        try:
            from numba import jit as _jit
            return _jit(nopython=True, **kwargs)(fn)
        except ImportError:
            @functools.wraps(fn)
            def wrapper(*args, **kw):
                return fn(*args, **kw)
            return wrapper
    if func is not None:
        return decorator(func)
    return decorator


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
