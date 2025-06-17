from collections import defaultdict
from collections.abc import Hashable, Iterator
from functools import partial, update_wrapper, wraps
from threading import RLock
from typing import Callable, Generic, Protocol, TypeVar


__all__ = ['has_memoized_methods', 'memoized_meth', 'memoized_generator']


InstanceType = TypeVar('InstanceType', bound=object)
ReturnType = TypeVar('ReturnType')
CachedType = TypeVar('CachedType')


class Method(Generic[InstanceType, ReturnType], Protocol):
    """
    Protocol for an instance method
    """
    def __call__(self, obj: InstanceType,
                 *args: Hashable, **kwargs: Hashable) -> ReturnType:
        ...


def has_memoized_methods(cls: type[InstanceType]) -> type[InstanceType]:
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
            obj._memoized_method_cache = {}
            obj._memoized_method_locks = defaultdict(RLock)

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
        obj._memoized_method_cache = {}
        obj._memoized_method_locks = defaultdict(RLock)

        return obj

    # Set our flag to avoid re-initialization
    _new.__has_memoized_methods = True

    # Apply the replacement
    cls.__new__ = staticmethod(_new)
    return cls


class memoized_meth(Generic[InstanceType, ReturnType, CachedType]):
    """
    Decorator for a thread-safe (concurrent read + write) cache of instance methods.

    The class whose methods this decorator is applied to must itself be decorated with
    `@has_memoized_methods`. If the class decorator is not present, a RuntimeError will
    be raised upon invocation of the wrapped method.
    """

    def __init__(self, meth: Method[InstanceType, ReturnType]) -> None:
        self._meth = meth
        self._lock = RLock()  # Global lock for safely initializing per-instance locks
        update_wrapper(self, meth)

    def _acquire_method_lock(self, obj: InstanceType,
                             *args: Hashable, **kwargs: Hashable) -> RLock:
        """
        Acquires a lock for the method call on the given object and arguments.
        """
        # Briefly use the global lock to safely access the per-instance lock in
        # case it hasn't been initialized yet
        with self._lock:
            # TODO: Should we lock on the full method call for more granularity?
            return obj._memoized_method_locks[self._meth]

    def _to_cached(self, value: ReturnType) -> CachedType:
        """
        Converts the return value of the method to a cached type.
        This can be overridden in subclasses to customize caching behavior.
        """
        return value

    def _from_cached(self, value: CachedType) -> ReturnType:
        """
        Converts the cached value back to the original return type.
        This can be overridden in subclasses to customize caching behavior.
        """
        return value

    def __get__(self, obj: InstanceType,
                cls: type[InstanceType] | None = None) -> Callable[..., ReturnType]:
        """
        Binds the memoized method to an instance.
        """
        if obj is None:
            return self

        return partial(self, obj)

    def __call__(self, obj: InstanceType,
                 *args: Hashable, **kwargs: Hashable) -> ReturnType:
        """
        Invokes the memoized method, caching the result if it hasn't been evaluated yet.
        """
        try:
            cache: dict[int, CachedType] = obj._memoized_method_cache

            # If arguments are not hashable, just evaluate the method directly
            if not isinstance(args, Hashable):
                return self._meth(obj, *args, **kwargs)

            # Key for the method call with all arguments
            key = hash((self._meth, args, frozenset(kwargs.items())))

            with self._acquire_method_lock(obj, *args, **kwargs):
                if key not in cache:
                    # If the result is not cached, call the method and cache the result
                    result = self._to_cached(self._meth(obj, *args, **kwargs))
                    cache[key] = result
                else:
                    # Otherwise retrieve the cached result
                    result = self._from_cached(cache[key])
            return result

        except AttributeError as e:
            # If the cache is missing, the class doesn't have the required decorator
            cls = type(obj)
            raise RuntimeError("Class '%s' must be decorated with "
                               "@has_memoized_methods to use @memoized_meth"
                               % cls.__name__) from e


ElementType = TypeVar('ElementType', covariant=True)
GeneratorMethod = Method[InstanceType, Iterator[ElementType]]


class SafeTee(Iterator[ElementType]):
    """
    A thread-safe version of `itertools.tee` that allows multiple iterators to safely
    share the same buffer.

    In theory, this comes at a cost to performance of iterating elements that haven't
    yet been generated, as `itertools.tee` is implemented in C (i.e. is fast) but we
    need to buffer (and lock) in Python instead.

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


class memoized_generator(memoized_meth[InstanceType, Iterator[ElementType],
                                       SafeTee[ElementType]]):
    """
    Decorator for a thread-safe cache of instance generator methods.

    The class whose methods this decorator is applied to must itself be decorated with
    `@has_memoized_methods`. If the class decorator is not present, a RuntimeError will
    be raised upon invocation of the wrapped method.

    The decorated generator method will be evaluated only once per unique argument set,
    and the result will be cached for subsequent calls via a thread-safe version of
    `itertools.tee`.
    """

    def __init__(self, meth: GeneratorMethod[InstanceType, ElementType]) -> None:
        super().__init__(meth)

    def _to_cached(self, value: Iterator[ElementType]) -> SafeTee[ElementType]:
        """
        Caches the returned generator wrapped in a SafeTee for buffer sharing.
        """
        return SafeTee(value)

    def _from_cached(self, value: SafeTee[ElementType]) -> Iterator[ElementType]:
        """
        Safely tees the cached generator, sharing the original buffer and lock.
        """
        return value.tee()
