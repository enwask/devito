from concurrent.futures import ThreadPoolExecutor
from devito import Operator, TimeFunction, Grid, Eq
from devito.logger import info
import numpy as np
from threading import Barrier, current_thread

from devito.tools import has_memoized_methods, memoized_meth


def test_concurrent_executing_operators():
    rng = np.random.default_rng()

    # build a simple operator and force it to compile
    grid = Grid(shape=(50, 50, 50))
    u = TimeFunction(name='u', grid=grid)
    op = Operator(Eq(u.forward, u + 1))

    # this forces the compile
    op.cfunction

    def do_run(op):
        # choose a new size
        shape = (rng.integers(20, 22), 30, rng.integers(20, 22))

        # make concurrent executions put a different value in the array
        # so we can be sure they aren't sharing an object even though the
        # name is reused
        val = current_thread().ident % 100000

        grid_private = Grid(shape=shape)
        u_private = TimeFunction(name='u', grid=grid_private)
        u_private.data[:] = val

        op(u=u_private, time_m=1, time_M=100)
        assert np.all(u_private.data[1, :, :, :] == val + 100)

    info("First running serially to demonstrate it works")
    do_run(op)

    info("Now creating thread pool")
    tpe = ThreadPoolExecutor(max_workers=16)

    info("Running operator in threadpool")
    futures = []
    for i in range(1000):
        futures.append(tpe.submit(do_run, op))

    # Get results - exceptions will be raised here if there are any
    for f in futures:
        f.result()


def test_memoized_meth_safety():
    """
    Tests that the `memoized_meth` decorator is thread-safe for concurrent invocations.
    """
    num_threads = 100
    num_tests = 100

    @has_memoized_methods
    class TestClass:
        def __init__(self):
            self.value = 0

        @memoized_meth
        def increment(self):
            # self.value should only be incremented once
            self.value += 1
            return self.value

    def do_work(test_instance: TestClass):
        # Wait for all threads to be ready
        barrier.wait()

        # Call the memoized method
        result = test_instance.increment()
        return result

    for i in range(num_tests):
        # Create a barrier to synchronize a bunch of calls to the memoized method
        barrier = Barrier(num_threads)
        test_instance = TestClass()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(do_work, test_instance) for _ in range(num_threads)]
        
        # Wait for all threads to complete and collect results
        results = set((f.result() for f in futures))
        assert len(results) == 1, "increment() should return the same result for all threads"
