from collections import defaultdict
from collections.abc import Hashable
from functools import partial, wraps
from itertools import tee
from threading import RLock
from typing import Any, Callable, TypeVar

__all__ = ['has_memoized_methods', 'memoized_meth', 'memoized_generator']


ReturnType = TypeVar('ReturnType', contravariant=True)
MethodType = TypeVar('MethodType', bound=Callable[..., ReturnType])


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
        init = cls.__init_finalize__
        if hasattr(init, '__has_memoized_methods'):
            return cls

        @wraps(init)
        def _init_finalize(obj, *args, **kwargs) -> None:
            # Apply our caches first
            obj.__method_cache = {}
            obj.__method_locks = defaultdict(RLock)

            # Call the original __init_finalize__ method
            init(obj, *args, **kwargs)

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


def memoized_meth(meth: MethodType) -> MethodType:
    """
    Decorator for a thread-safe (concurrent read + write) cache of instance methods.

    The class whose methods this decorator is applied to must itself be decorated with
    `@has_memoized_methods`. If the class decorator is not present, a RuntimeError will
    be raised upon invocation of the wrapped method.
    """
    # A global lock used to ensure per-method locks are created safely
    _global_lock = RLock()

    @wraps(meth)
    def _wrapped_meth(obj, *args: Hashable, **kwargs: Hashable) -> ReturnType:
        try:
            cache: dict[int, ReturnType] = obj.__method_cache
            locks: defaultdict[MethodType[Any], RLock] = obj.__method_locks

            # Key for the method call with all arguments
            _key = hash((meth, args, frozenset(kwargs.items())))

            # Briefly use the global lock to safely access the per-method lock if it
            # hasn't been initialized yet
            with _global_lock:
                lock = locks[meth]

            # Lock on the method
            # TODO: Should we lock on arguments for more granularity?
            with lock:
                if _key not in cache:
                    # If the result is not cached, call the method and cache the result
                    result = meth(obj, *args, **kwargs)
                    cache[_key] = result
                else:
                    # Otherwise retrieve the cached result
                    result = cache[_key]
            return result

        except AttributeError as e:
            # If the cache is missing, the class doesn't have the required decorator
            cls = type(obj)
            raise RuntimeError("Class '%s' must be decorated with "
                               "@has_memoized_methods to use @memoized_meth"
                               % cls.__name__) from e

    return _wrapped_meth


class memoized_generator:

    """
    Decorator. Cache the return value of an instance generator method.
    """

    def __init__(self, func):
        self.func = func

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kwargs):
        if not isinstance(args, Hashable):
            # Uncacheable, a list, for instance.
            # Better to not cache than blow up.
            return self.func(*args)
        obj = args[0]
        try:
            cache = obj.__cache_gen
        except AttributeError:
            cache = obj.__cache_gen = {}
        key = (self.func, args[1:], frozenset(kwargs.items()))
        it = cache[key] if key in cache else self.func(*args, **kwargs)
        cache[key], result = tee(it)
        return result
