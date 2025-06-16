from collections.abc import Hashable
from functools import partial
from itertools import tee

__all__ = ['memoized_meth', 'memoized_generator']


class memoized_meth:
    """
    Decorator. Cache the return value of a class method.

    Unlike ``memoized_func``, the return value of a given method invocation
    will be cached on the instance whose method was invoked. All arguments
    passed to a method decorated with memoize must be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method: ::

        class Obj:
            @memoize
            def add_to(self, arg):
                return self + arg
        Obj.add_to(1) # not enough arguments
        Obj.add_to(1, 2) # returns 3, result is not cached

    Adapted from: ::

        code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kw):
        if not isinstance(args, Hashable):
            # Uncacheable, a list, for instance.
            # Better to not cache than blow up.
            return self.func(*args)
        obj = args[0]
        try:
            cache = obj.__cache_meth
        except AttributeError:
            cache = obj.__cache_meth = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


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
