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

    @property
    def max_workers(self) -> int:
        """
        Returns the maximum number of workers that can be used by this executor.
        """
        ...

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


class ThreadedExecutor(GenericExecutor, ThreadPoolExecutor):
    """
    A threaded executor that runs tasks in a thread pool, with the same interface as
    `ThreadPoolExecutor`. This is used when free threading is available.

    Exposes the value for `max_workers`.
    """

    def __init__(self, max_workers: int = None):
        super().__init__(max_workers=max_workers)
        self.max_workers = self._max_workers  # Expose the computed max_workers value


def get_executor(max_workers: int = None,
                 force_threaded: bool = False) -> GenericExecutor:
    """
    Returns a new executor for mapping tasks. If `force_threaded` is `True`, a
    `ThreadedExecutor` is returned regardless of whether free threading is available.

    Otherwise, if `max_workers` is equal to 0, a `SerialExecutor` is returned.

    If neither of the above conditions is met, a `ThreadedExecutor` is returned iff
    free threading is available (i.e., the GIL is disabled), using the specified value
    for `max_workers`.

    The resulting executor should be used as a context manager.
    """
    if force_threaded or (max_workers and is_free_threading()):
        return ThreadedExecutor(max_workers=max_workers)

    return SerialExecutor()
