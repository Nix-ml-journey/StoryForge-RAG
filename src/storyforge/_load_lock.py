"""One lock for the "load a model once and cache it" helpers, so two threads never load the same model twice."""
import functools
import threading

_LOCK = threading.RLock()


def serialized(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _LOCK:
            return fn(*args, **kwargs)

    return wrapper
