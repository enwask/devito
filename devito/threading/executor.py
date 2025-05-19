import sys
from concurrent.futures import Executor
from functools import cache

__all__ = ['PoolExecutor']


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
def _get_executor_type() -> type[Executor]:
    """
    Returns the appropriate executor type based on the GIL status.

    If the GIL is enabled, we use `ProcessPoolExecutor` for parallelism.
    Otherwise, we can use `ThreadPoolExecutor` since there's no lock on
    the interpreter.
    """
    if _is_gil_enabled():
        from concurrent.futures import ProcessPoolExecutor
        return ProcessPoolExecutor

    from concurrent.futures import ThreadPoolExecutor
    return ThreadPoolExecutor


class PoolExecutor(_get_executor_type()):
    """
    A wrapper around the appropriate executor type based on GIL status.
    """

    def __init__(self, max_workers=None, initializer=None, initargs=()):
        super().__init__(max_workers=max_workers,
                         initializer=initializer, initargs=initargs)
