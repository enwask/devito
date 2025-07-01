from typing import Iterator
import numpy as np
import pytest
from sympy.abc import a, b, c, d, e

import time

from devito.tools import (UnboundedMultiTuple, ctypes_to_cstr, toposort,
                          filter_ordered, transitive_closure, UnboundTuple,
                          has_memoized_methods, memoized_meth,
                          memoized_generator)
from devito.types.basic import Symbol


@pytest.mark.parametrize('elements, expected', [
    ([[a, b, c], [c, d, e]], [a, b, c, d, e]),
    ([[e, d, c], [c, b, a]], [e, d, c, b, a]),
    ([[a, b, c], [b, d, e]], [a, b, d, c, e]),
    ([[a, b, c], [d, b, c]], [a, d, b, c]),
    ([[a, b, c], [c, d, b]], None),
])
def test_toposort(elements, expected):
    try:
        ordering = toposort(elements)
        assert ordering == expected
    except ValueError:
        assert expected is None


def test_sorting():
    key = lambda x: x

    # Need predictable random sequence or test will
    # have inconsistent behaviour results between tests.
    np.random.seed(0)
    array = np.random.randint(-1000, 1000, 10000)

    t0 = time.time()
    for _ in range(100):
        sort_key = filter_ordered(array, key=key)
    t1 = time.time()
    for _ in range(100):
        sort_nokey = filter_ordered(array)
    t2 = time.time()

    assert t2 - t1 < 0.8 * (t1 - t0)
    assert sort_key == sort_nokey


def test_transitive_closure():
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    e = Symbol('e')
    f = Symbol('f')

    mapper = {a: b, b: c, c: d, f: e}
    mapper = transitive_closure(mapper)
    assert mapper == {a: d, b: d, c: d, f: e}


def test_loops_in_transitive_closure():
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    e = Symbol('e')

    mapper = {a: b, b: c, c: d, d: e, e: b}
    mapper = transitive_closure(mapper)
    assert mapper == {a: b, b: c, c: d, d: e, e: b}


@pytest.mark.parametrize('mapper, expected', [
    ([{a: b, b: a, c: d, d: e, e: c}, [a, a, c, c, c]]),
    ([{a: b, b: c, c: b, d: e, e: d}, [b, b, b, d, d]]),
    ([{a: c, b: a, c: a, d: e, e: d}, [a, a, a, d, d]]),
    ([{c: a, b: a, a: c, e: c, d: e}, [a, a, a, c, c]]),
    ([{a: b, b: c, c: d, d: e, e: b}, [b, b, b, b, b]]),
])
def test_sympy_subs_symmetric(mapper, expected):
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    e = Symbol('e')

    input = [a, b, c, d, e]
    input = [i.subs(mapper) for i in input]
    assert input == expected


@pytest.mark.parametrize('dtype, expected', [
    (np.float32, 'float'),
    (np.float64, 'double'),
    (np.int32, 'int'),
    (np.int64, 'long'),
    (np.uint64, 'unsigned long'),
    (np.int8, 'char'),
    (np.uint8, 'unsigned char'),
])
def test_ctypes_to_cstr(dtype, expected):
    a = Symbol(name='a', dtype=dtype)
    assert ctypes_to_cstr(a._C_ctype) == expected


def test_unbounded_multi_tuple():
    ub = UnboundedMultiTuple([1, 2], [3, 4])
    with pytest.raises(StopIteration):
        ub.next()

    with pytest.raises(StopIteration):
        assert ub.curitem()

    ub.iter()
    assert ub.curitem() == (1, 2)
    assert ub.next() == 1
    assert ub.next() == 2

    with pytest.raises(StopIteration):
        ub.next()

    ub.iter()
    assert ub.next() == 3
    assert ub.next() == 4

    with pytest.raises(StopIteration):
        ub.next()

    ub.iter()
    assert ub.next() == 3

    assert ub.nextitem() == (3, 4)


def test_unbound_tuple():
    # Make sure we don't drop needed None for 2.5d
    ub = UnboundTuple(None, None)
    assert len(ub) == 2
    assert ub[10] is None

    ub = UnboundTuple(1, 2, 3)
    assert len(ub) == 3
    assert ub[10] == 3
    assert ub[1:4] == (2, 3, 3)
    assert ub.next() == 1
    assert ub.next() == 2
    ub.iter()
    assert ub.next() == 1


class TestMemoizedMethods:
    """
    Tests tools for memoization of instance methods and generators, including
    concurrent invocations and iteration.
    """

    @has_memoized_methods
    class Base:
        """
        Base class for testing memoized instance methods.
        """

        with_init_finalize = False

        def __new__(cls, *args, **kwargs):
            obj = super().__new__(cls)
            return obj

        def __init__(self, *args, **kwargs):
            self.did_init = True

    class BaseWithFinalize(Base):
        """
        Base class for testing memoized instance methods on a class with Devito's
        __init_finalize__ pattern.
        """

        with_init_finalize = True

        def __new__(cls, *args, **kwargs):
            obj = super().__new__(cls, *args, **kwargs)
            obj.__init_finalize__(*args, **kwargs)
            return obj

        def __init_finalize__(self, *args, **kwargs):
            self.did_init_finalize = True

    @pytest.fixture(params=[Base, BaseWithFinalize])
    def base(self, request) -> type[Base]:
        """
        Parametrizes all tests to run against a base class both with and without
        Devito's `__init_finalize__` pattern.
        """
        return request.param

    @pytest.fixture(autouse=True)
    def track_cache_hits_misses(self, monkeypatch: pytest.MonkeyPatch):
        """
        Modifies `memoized_meth` and `memoized_generator` to track cache hits
        and misses for testing.
        """
        def new(cls, *a, **kw):
            dec = object.__new__(cls)
            dec.hits = dec.misses = 0
            return dec

        def postprocess_meth(dec, value, cache_hit):
            dec.hits += int(cache_hit)
            dec.misses += int(not cache_hit)
            return value

        def postprocess_generator(dec, value, cache_hit):
            dec.hits += int(cache_hit)
            dec.misses += int(not cache_hit)
            return value.tee()

        monkeypatch.setattr(memoized_meth, '__new__', new)
        monkeypatch.setattr(memoized_meth, '_postprocess', postprocess_meth)
        monkeypatch.setattr(memoized_generator, '_postprocess', postprocess_generator)

    def test_has_memoized_methods_idempotency(self, base: type[Base]):
        """
        Tests that applying the `@has_memoized_methods` decorator multiple times
        in an inheritance chain does not lead to multiple initializations
        """
        @has_memoized_methods
        @has_memoized_methods
        class Test(base):
            def __init__(self, *args, **kwargs) -> None:
                assert not hasattr(self, 'did_init')
                super().__init__(*args, **kwargs)

            def __init_finalize__(self, *args, **kwargs) -> None:
                # Only called when the base has __init_finalize__
                assert not hasattr(self, 'did_init_finalize')
                super().__init_finalize__(*args, **kwargs)

        # Create an instance to trigger initialization
        instance = Test()

        # Check that the instance has been initialized correctly
        assert instance.did_init
        if base.with_init_finalize:
            assert instance.did_init_finalize

    def test_memoized_meth(self, base: type[Base]):
        """
        Tests basic functionality of memoized methods.
        """

        class Test(base):
            def __init__(self) -> None:
                super().__init__()
                self.seen: set[tuple[int, int]] = set()


            @memoized_meth
            def add(self, x: int, y: int) -> int:
                assert (x, y) not in self.seen
                self.seen.add((x, y))

                return x + y

        test = Test()
        assert test.add(1, 2) == test.add(2, 1) == 3
        assert test.add(1, 2) == test.add(2, 1) == 3

        # Should have hit the cache once for (1,2 ) and once for (2, 1)
        assert Test.add.hits == 2

        # Calls on a new instance should not hit the cache
        test2 = Test()
        assert test2.add(1, 2) == test2.add(2, 1) == 3
        assert Test.add.hits == 2

    def test_unhashable_args(self, base: type[Base]):
        """
        Tests that memoized methods do not cache results for unhashable arguments.
        """

        class Test(base):
            def __init__(self) -> None:
                super().__init__()

            @memoized_meth
            def add(self, x: int, y: int, _: dict) -> int:
                return x + y

            @memoized_generator
            def range(self, n: int, _: dict) -> Iterator[int]:
                yield from range(n)


        test = Test()
        assert test.add(1, 2, {}) == test.add(2, 1, {}) == 3
        assert test.add(1, 2, {}) == test.add(2, 1, {}) == 3

        assert list(test.range(5, {})) == list(test.range(5, {})) == [0, 1, 2, 3, 4]

        # Should not have interacted with the cache
        assert Test.add.hits == Test.add.misses == 0
        assert Test.range.hits == Test.range.misses == 0


    def test_memoized_generator(self, base: type[Base]):
        """
        Tests basic functionality of memoized generators.
        """

        class Test(base):
            def __init__(self) -> None:
                super().__init__()
                self.seen: set[tuple[int, int]] = set()

            @memoized_generator
            def range(self, n: int) -> Iterator[int]:
                for i in range(n):
                    assert (n, i) not in self.seen
                    self.seen.add((n, i))
                    yield i

        test = Test()
        gen1 = test.range(5)
        gen2 = test.range(5)

        # Both generators should yield the same values
        assert list(gen1) == list(gen2) == [0, 1, 2, 3, 4]

        # Should have hit the cache once for the generator
        assert Test.range.hits == 1

        # Calls on a new instance should not hit the cache
        test2 = Test()
        gen3 = test2.range(5)
        assert list(gen3) == [0, 1, 2, 3, 4]
        assert Test.range.hits == 1
