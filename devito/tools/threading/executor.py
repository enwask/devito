from concurrent.futures import Executor, Future, ThreadPoolExecutor
from functools import cache
from typing import Callable, Iterable, TypeVar
import sys

__all__ = ['is_free_threading', 'GenericExecutor', 'SerialExecutor', 'get_executor']


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
    available (i.e., whether the GIL is disabled).

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

    Serial execution ignores all arguments that would be relevant for
    `ThreadPoolExecutor` or `ProcessPoolExecutor`, including timeouts etc.
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

    def map(self, fn, *iterables, **_):
        return (fn(*args) for args in zip(*iterables))


def get_executor(threading: bool | None = None) -> GenericExecutor:
    """
    Returns a new executor for mapping tasks. If a value is provided for `threading`,
    it will be used to decide between a `ThreadPoolExecutor` and a `SerialExecutor`.

    If no value is provided, a `ThreadPoolExecutor` is returned iff free threading is
    available (i.e., the GIL is disabled).

    The resulting executor should be used as a context manager.
    """
    if threading or is_free_threading():
        return ThreadPoolExecutor()

    return SerialExecutor()
