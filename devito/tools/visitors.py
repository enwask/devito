import inspect
from functools import update_wrapper
from threading import local
from typing import Callable, Generic, Hashable, TypeVar

__all__ = ['GenericVisitor', 'DagVisitor', 'dag_visitor']


class GenericVisitor:

    """
    A generic visitor.

    To define handlers, subclasses should define :data:`visit_Foo`
    methods for each class :data:`Foo` they want to handle.
    If a specific method for a class :data:`Foo` is not found, the MRO
    of the class is walked in order until a matching method is found.

    The method signature is:

        .. code-block::
           def visit_Foo(self, o, [*args, **kwargs]):
               pass

    The handler is responsible for visiting the children (if any) of
    the node :data:`o`.  :data:`*args` and :data:`**kwargs` may be
    used to pass information up and down the call stack.  You can also
    pass named keyword arguments, e.g.:

        .. code-block::
           def visit_Foo(self, o, parent=None, *args, **kwargs):
               pass
    """

    def __init__(self):
        handlers = {}
        # visit methods are spelt visit_Foo.
        prefix = "visit_"
        # Inspect the methods on this instance to find out which
        # handlers are defined.
        for (name, meth) in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith(prefix):
                continue
            # Check the argument specification
            # Valid options are:
            #    visit_Foo(self, o, [*args, **kwargs])
            argspec = inspect.getfullargspec(meth)
            if len(argspec.args) < 2:
                raise RuntimeError("Visit method signature must be "
                                   "visit_Foo(self, o, [*args, **kwargs])")
            handlers[name[len(prefix):]] = meth
        self._handlers = handlers

    """
    :attr:`default_args`. A dict of default keyword arguments for the visitor.
    These are not used by default in :meth:`visit`, however, a caller may pass
    them explicitly to :meth:`visit` by accessing :attr:`default_args`.
    For example::

        .. code-block::
           v = FooVisitor()
           v.visit(node, **v.default_args)
    """
    default_args = {}

    @classmethod
    def default_retval(cls):
        """
        A method that returns an object to use to populate return values.

        If your visitor combines values in a tree-walk, it may be useful to
        provide a object to combine the results into. :meth:`default_retval`
        may be defined by the visitor to be called to provide an empty object
        of appropriate type.
        """
        return None

    def lookup_method(self, instance):
        """
        Look up a handler method for a visitee.

        Parameters
        ----------
        instance : object
            The instance to look up a method for.
        """
        cls = instance.__class__
        try:
            # Do we have a method handler defined for this type name
            return self._handlers[cls.__name__]
        except KeyError:
            # No, walk the MRO.
            for klass in cls.mro()[1:]:
                entry = self._handlers.get(klass.__name__)
                if entry:
                    # Save it on this type name for faster lookup next time
                    self._handlers[cls.__name__] = entry
                    return entry
        raise RuntimeError("No handler found for class %s", cls.__name__)

    def visit(self, o, *args, **kwargs):
        """
        Apply this Visitor to an object.

        Parameters
        ----------
        o : object
            The object to be visited.
        *args
            Optional arguments to pass to the visit methods.
        **kwargs
            Optional keyword arguments to pass to the visit methods.
        """
        ret = self._visit(o, *args, **kwargs)
        ret = self._post_visit(ret)
        return ret

    def _visit(self, o, *args, **kwargs):
        """Visit ``o``."""
        meth = self.lookup_method(o)
        return meth(o, *args, **kwargs)

    def _post_visit(self, ret):
        """Postprocess the visitor output before returning it to the caller."""
        return ret

    def visit_object(self, o, **kwargs):
        return self.default_retval()


# Generic return type for a DAG visitor
ReturnType = TypeVar('ReturnType', covariant=True)


class DagVisitor(Generic[ReturnType]):
    """
    A generic wrapper for a DAG visitor which provides a visit method that memoizes
    calls to the wrapped function, allowing the original function to use recursive
    calls as normal.

    The memo table is in thread-local storage, so concurrency within nested calls is
    unsafe but the visitor can be parallelized at the root level.
    """

    def __init__(self, func: Callable[..., ReturnType],
                 key: Callable[..., Hashable] | None = None) -> None:
        self._func = func
        self._key = key or (lambda *a, **kw: (a, frozenset(kw.items())))

        self._local = local()
        self._local.memo = None

        update_wrapper(self, func)

    def visit(self, *args, **kwargs) -> ReturnType:
        """
        Applies the visitor with memoization.
        """
        if self._local.memo is not None:
            raise RuntimeError("DagVisitor.visit() called while already in "
                               "a traversal context.")

        # Initialize memo and call the wrapped function
        self._local.memo = {}
        res = self(*args, **kwargs)

        # Clear the memo table after the call
        self._local.memo = None
        return res

    def __call__(self, *args, **kwargs) -> ReturnType:
        """
        Invokes the wrapped function. If in a traversal context, uses the memo table.
        """
        if self._local.memo is None:
            return self._func(*args, **kwargs)

        key = self._key(*args, **kwargs)
        memo = self._local.memo

        if key in memo:
            return memo[key]

        result = self._func(*args, **kwargs)
        memo[key] = result
        return result


def dag_visitor(key: Callable[..., Hashable] | None = None) \
        -> Callable[[Callable[..., ReturnType]], DagVisitor[ReturnType]]:
    """
    A decorator to create a DagVisitor for a function.

    May be passed a key function to customize how memoization keys are generated;
    by default, positional and keyword arguments are hashed together.
    """
    def decorator(fun: Callable[..., ReturnType]) -> DagVisitor[ReturnType]:
        """
        Decorates a function to create a DagVisitor.
        """
        return DagVisitor(fun, key)

    return decorator
