from collections import defaultdict
from collections.abc import Hashable, Iterator
from functools import partial, wraps
from itertools import tee
from threading import RLock
from typing import Any, Callable, TypeVar

__all__ = ['has_memoized_methods', 'memoized_meth', 'memoized_generator']


ReturnType = TypeVar('ReturnType')
MethodType = Callable[..., ReturnType]


def has_memoized_methods(cls: type) -> type:
    """
    Class decorator to provide instance-level caches for methods decorated with
    `@memoized_meth`. If this initialization were done in the decorated methods,
    we might introduce race conditions when multiple threads concurrently call a
    method for the first time.
    """
    # If the class has an __init_finalize__ method, we need to modify it instead in case
    # it calls memoized methods during initialization (e.g. SubDimension)
    if hasattr(cls, '__init_finalize__'):
        # Don't modify the class if we've already applied this decorator
        init_finalize = cls.__init_finalize__
        if hasattr(init_finalize, '__has_memoized_methods'):
            return cls

        @wraps(init_finalize)
        def _init_finalize(obj, *args, **kwargs) -> None:
            # Apply our caches first
            obj.__method_cache = {}
            obj.__method_locks = defaultdict(RLock)

            # Call the original __init_finalize__ method
            init_finalize(obj, *args, **kwargs)

        # Set our flag to avoid re-initialization
        _init_finalize.__has_memoized_methods = True

        # Apply the replacement
        cls.__init_finalize__ = _init_finalize
        return cls

    # Make sure we're not modifying a class that already has this decorator
    new = cls.__new__
    if hasattr(new, '__has_memoized_methods'):
        return new

    @wraps(new)
    def _new(_cls, *args, **kwargs):
        # Don't pass arguments to object constructor if there are no parent classes
        if new is object.__new__:
            obj = new(_cls)
        else:
            obj = new(_cls, *args, **kwargs)

        # Initialize the method cache and locks
        obj.__method_cache = {}
        obj.__method_locks = defaultdict(RLock)

        return obj

    # Set our flag to avoid re-initialization
    _new.__has_memoized_methods = True

    # Apply the replacement
    cls.__new__ = staticmethod(_new)
    return cls


def memoized_meth(meth: MethodType[ReturnType]) -> MethodType[ReturnType]:
    """
    Decorator for a thread-safe (concurrent read + write) cache of instance methods.

    The class whose methods this decorator is applied to must itself be decorated with
    `@has_memoized_methods`. If the class decorator is not present, a RuntimeError will
    be raised upon invocation of the wrapped method.
    """
    # A global (to the method) lock used to ensure per-instance locks are created safely
    _global_lock = RLock()

    @wraps(meth)
    def _meth(obj, *args: Hashable, **kwargs: Hashable) -> ReturnType:
        try:
            cache: dict[int, ReturnType] = obj.__method_cache
            locks: defaultdict[MethodType[Any], RLock] = obj.__method_locks

            # If arguments are not hashable, just evaluate the method directly
            if not isinstance(args, Hashable):
                return meth(obj, *args, **kwargs)

            # Key for the method call with all arguments
            key = hash((meth, args, frozenset(kwargs.items())))

            # Briefly use the global lock to safely access the per-instance lock if it
            # hasn't been initialized yet
            with _global_lock:
                lock = locks[meth]

            # Lock on the method
            # TODO: Should we lock on arguments for more granularity?
            with lock:
                if key not in cache:
                    # If the result is not cached, call the method and cache the result
                    result = meth(obj, *args, **kwargs)
                    cache[key] = result
                else:
                    # Otherwise retrieve the cached result
                    result = cache[key]
            return result

        except AttributeError as e:
            # If the cache is missing, the class doesn't have the required decorator
            cls = type(obj)
            raise RuntimeError("Class '%s' must be decorated with "
                               "@has_memoized_methods to use @memoized_meth"
                               % cls.__name__) from e

    return _meth


ElementType = TypeVar('ElementType', covariant=True)
GeneratorType = Callable[..., Iterator[ElementType]]


class SafeTee(Iterator[ElementType]):
    """
    A thread-safe version of `itertools.tee` that allows multiple iterators to safely
    share the same buffer.
    
    This comes at a cost to performance of iterating elements that haven't yet been
    generated, as `itertools.tee` is implemented in C (i.e. is fast) but we need to
    buffer (and lock against that buffer) in Python instead.
    
    However, the lock is not needed for elements that have already been buffered,
    allowing for concurrent iteration after the generator is initially consumed.
    """
    def __init__(self, source_iter: Iterator[ElementType],
                 buffer: list[ElementType] = None, lock: RLock = None) -> None:
        self._source_iter = source_iter
        self._buffer = buffer if buffer is not None else []
        self._lock = lock if lock is not None else RLock()
        self._next = 0

    def __iter__(self) -> Iterator[ElementType]:
        return self

    def __next__(self) -> ElementType:
        """
        Safely retrieves the buffer if available, or generates the next element
        from the source iterator if not.
        """
        while True:
            if self._next < len(self._buffer):
                # If we have another buffered element, return it
                result = self._buffer[self._next]
                self._next += 1

                return result

            # Otherwise, we may need to generate a new element
            with self._lock:
                if self._next < len(self._buffer):
                    # Another thread has already generated the next element; retry
                    continue

                # Generate the next element from the source iterator
                try:
                    # Try to get the next element from the source iterator
                    result = next(self._source_iter)
                    self._buffer.append(result)
                    self._next += 1
                    return result
                except StopIteration as e:
                    # The source iterator has been exhausted
                    raise StopIteration from e
    
    def __copy__(self) -> 'SafeTee':
        """
        Creates a copy of this iterator that shares the same buffer and lock.
        """
        return SafeTee(self._source_iter, self._buffer, self._lock)
    
    def tee(self) -> Iterator[ElementType]:
        """
        Creates a new iterator that shares the same buffer and lock.
        """
        return SafeTee(self._source_iter, self._buffer, self._lock)


def memoized_generator(meth: GeneratorType[ElementType]) -> GeneratorType[ElementType]:
    """
    Decorator for a thread-safe cache of instance generator methods.

    The class whose methods this decorator is applied to must itself be decorated with
    `@has_memoized_methods`. If the class decorator is not present, a RuntimeError will
    be raised upon invocation of the wrapped method.

    The decorated generator method should be evaluated only once per unique set of
    arguments, and the results will be cached for subsequent calls via a thread-safe
    version of `itertools.tee`.
    """

    # A global (to the method) lock used to ensure per-instance locks are created safely
    _global_lock = RLock()

    @wraps(meth)
    def _meth(obj, *args: Hashable, **kwargs: Hashable) -> Iterator[ElementType]:
        try:
            cache: dict[int, SafeTee[ElementType]] = obj.__method_cache
            locks: defaultdict[GeneratorType[Any], RLock] = obj.__method_locks

            # If arguments are not hashable, just evaluate the method directly
            if not isinstance(args, Hashable):
                return meth(obj, *args, **kwargs)

            # Key for the method call with all arguments
            key = hash((meth, args, frozenset(kwargs.items())))

            # Briefly use the global lock to safely access the per-instance lock if it
            # hasn't been initialized yet
            with _global_lock:
                lock = locks[meth]

            # Lock on the method
            # TODO: Should we lock on arguments for more granularity?
            with lock:
                if key not in cache:
                    # If the result is not cached, call the method and cache the result
                    source_tee = SafeTee(meth(obj, *args, **kwargs))
                    cache[key] = source_tee
                else:
                    # Otherwise retrieve the cached result
                    source_tee = cache[key]

            # Safely tee the cached generator
            return source_tee.tee()

        except AttributeError as e:
            # If the cache is missing, the class doesn't have the required decorator
            cls = type(obj)
            raise RuntimeError("Class '%s' must be decorated with "
                               "@has_memoized_methods to use @memoized_generator"
                               % cls.__name__) from e

    return _meth
