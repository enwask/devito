import sys
import threading

from functools import cache

__all__ = ['sympy_mutex', 'safe_dict_copy', 'is_free_threading']


sympy_mutex = threading.RLock()


def safe_dict_copy(v):
    """
    Thread-safe copy of a dict.

    Being implemented as a retry loop around the copy(), this function is
    indicated for situations in which concurrent dict updates are unlikely,
    otherwise it might eventually cause performance degradations (in which
    case, lock-based solutions would be preferable).

    Notes
    -----
    See https://bugs.python.org/issue40327
    """
    while True:
        try:
            return v.copy()
        except RuntimeError:
            pass


def _is_gil_enabled() -> bool:
    """
    Checks if the GIL is enabled.

    Python 3.13+ has a build flag to disable the GIL for proper
    multithreading; we check if the version is high enough and
    that flag has been set.
    """
    if sys.version_info < (3, 13):
        return True

    return sys._is_gil_enabled()


@cache
def is_free_threading() -> bool:
    """
    Checks if the current Python interpreter is free-threading; i.e.,
    the Global Interpreter Lock (GIL) is disabled.
    """
    return not _is_gil_enabled()
