from concurrent.futures import ThreadPoolExecutor
import random

import pytest
from devito import Operator, TimeFunction, Grid, Eq
from devito.logger import info
import numpy as np
from threading import Barrier, current_thread

from devito.tools import has_memoized_methods, memoized_meth
from devito.tools.threading.executor import get_executor
from devito.tools.threading.queue import RecursionQueue, RecursionRoutine, parallel_recursive


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
    num_threads = 100 # Number of threads per test case
    num_tests = 100 # Number of times to try for a clash
    seed = 42 # Seed for reproducibility

    @has_memoized_methods
    class TestClass:
        def __init__(self) -> None:
            self.value = 0

        @memoized_meth
        def increment(self, _: int) -> int:
            # self.value should only be incremented once per unique argument
            self.value += 1
            return self.value

    def do_work(test_instance: TestClass, arg: int) -> int:
        # Wait for all threads to be ready
        barrier.wait()

        # Call the memoized method
        result = test_instance.increment(arg)
        return result

    random.seed(seed)
    for i in range(num_tests):
        # Create a barrier to synchronize a bunch of calls to the memoized method
        barrier = Barrier(num_threads)
        instance = TestClass()

        # Choose a random number of unique arguments to pass to the method
        num_unique_args = random.randint(1, num_threads)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(do_work, instance, i % num_unique_args)
                       for i in range(num_threads)]

        # Wait for all threads to complete and collect results
        results = set((f.result() for f in futures))
        assert len(results) == num_unique_args, ("increment() should return a number of "
                                                 "unique values equal to unique args")


class Node:
    """
    A simple node class for testing the `RecursionQueue`.
    """
    def __init__(self, value, *children: 'Node'):
        self.value = value
        self.children = list(children)

    def add_child(self, child):
        self.children.append(child)


def make_random_tree(seed: int | None = None,
                     depth: int = 8, branching_factor: int = 4) -> tuple[Node, int]:
    """
    Creates a random tree structure for testing, returning the root and sum of values.
    """
    if seed is not None:
        random.seed(seed)

    if depth == 0:
        value = random.randint(1, 100)
        return Node(value), value

    root = Node(random.randint(1, 100))
    total = root.value
    for _ in range(random.randint(1, branching_factor)):
        child, value = make_random_tree(None, depth - 1, branching_factor)
        root.add_child(child)
        total += value

    return root, total


class TestRecursionQueue:
    """
    Tests functionality of the `RecursionQueue` class and `parallel_recursive` decorator.
    """

    def test_multiple_application(self):
        """
        Tests reapplication of the same `RecursionQueue` in one context.
        """

        @parallel_recursive
        def sum_values(queue: RecursionQueue[Node, int], node: Node) -> RecursionRoutine[Node, int]:
            """
            Recursively sums the values of nodes in a tree using `RecursionQueue`.
            """
            child_sums = yield queue.request(node.children)
            return node.value + sum(child_sums)

        # Create a random tree
        root, total = make_random_tree(seed=42)
        with sum_values(get_executor(max_workers=16, force_threaded=True)) as queue:
            result1 = queue.apply(root)
            result2 = queue.apply(root)

        # Check that both results are correct
        assert result1 == total, f"Expected {total}, got {result1}"
        assert result2 == total, f"Expected {total}, got {result2}"

        # Check that the executor was shut down
        with pytest.raises(RuntimeError):
            queue._executor.submit(lambda: None)
