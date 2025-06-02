from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Callable, Iterable, Optional, TypeVar
import sys
import threading

from functools import cache

__all__ = ['sympy_mutex', 'safe_dict_copy', 'is_free_threading',
           'get_executor', 'GenericExecutor', 'SerialExecutor']


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


ResultType = TypeVar('ResultType', covariant=True)


class GenericExecutor(Executor):
    """
    A generic executor that can be used to submit tasks and retrieve results. May be a
    `ThreadPoolExecutor` or a `SerialExecutor`, depending on whether free threading is
    available (i.e., the GIL is disabled).

    Exists to provide typing for `Executor.map`.
    """

    def submit(self, fn: Callable[..., ResultType],
               /, *args, **kwargs) -> Future[ResultType]:
        return super().submit(fn, *args, **kwargs)

    def map(self, fn: Callable[..., ResultType], *iterables,
            timeout: float = None, chunksize: int = 1) -> Iterable[ResultType]:
        return super().map(fn, *iterables, timeout=timeout, chunksize=chunksize)


class SerialExecutor(GenericExecutor):
    """
    A serial executor that runs tasks in the current thread, with the same interface as
    `ThreadPoolExecutor`. This is used when free threading is not available.
    """

    def submit(self, fn, /, *args, **kwargs):
        """
        Runs the function immediately in the current thread and returns a Future.
        """
        future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

        return future


_global_executor: Optional[GenericExecutor] = None


def get_executor() -> GenericExecutor:
    """
    Returns the global thread pool executor instance. With free threading, this is an
    instance of `ThreadPoolExecutor`; otherwise, it's an executor that runs tasks in
    serial (which defers to e.g. the built-in `map` function).
    """
    global _global_executor

    if _global_executor is None:
        if is_free_threading():
            _global_executor = ThreadPoolExecutor()
        else:
            _global_executor = SerialExecutor()

    return _global_executor
