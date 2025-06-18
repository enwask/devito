from collections import namedtuple
from collections.abc import Generator, Iterable
from queue import Queue
from threading import Event, RLock
from typing import Callable, Generic, Protocol, TypeVar
from uuid import UUID, uuid4

from devito.tools import as_tuple, GenericExecutor

__all__ = ['NodeType', 'ResultType', 'RecursionRoutine', 'RecursionQueue',
           'parallel_recursive']


ResultType = TypeVar('ResultType', covariant=True)
NodeType = TypeVar('NodeType', covariant=True)


class Request(Generic[NodeType]):
    """
    Describes a request for nodes to be processed in the recursion queue.
    """
    def __init__(self, nodes: Iterable[NodeType], args: tuple = (), kwargs: dict = None):
        """
        Initializes the request with nodes, arguments, and keyword arguments.
        """
        self.nodes = as_tuple(nodes)
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}


# Describes a Process coroutine that yields tasks for children
RecursionRoutine = Generator[Request[NodeType], list[ResultType], ResultType]


class Task(Generic[NodeType, ResultType]):
    """
    Describes a task in the actual task queue or awaiting results from children.
    """
    def __init__(self, parent_id: UUID | None, index: int,
                 coroutine: RecursionRoutine[NodeType, ResultType]) -> None:
        self.parent_id = parent_id  # The ID of the parent task, if any
        self.index: int = index  # The index of the task in the parent's results
        self.coroutine = coroutine  # The process coroutine
        self.map_result: list[ResultType] | None = None  # The pending child results
        self.num_waiting: int = 0  # Number of child results we're still waiting for

    def set_waiting(self, num_children: int) -> None:
        """
        Signals that we're waiting for a number of child results; the task will then
        be put in the pending results map until all results are ready.
        """
        self.num_waiting = num_children
        self.map_result = [None] * num_children

    def set_results(self, results: list[ResultType]) -> None:
        """
        Sets results for the task immediately, bypassing the waiting mechanism.
        """
        self.map_result = results
        self.num_waiting = 0

    def clear_results(self) -> None:
        """
        Clears the pending results.
        """
        self.num_waiting = 0
        self.map_result = None

    def put_result(self, index: int, result: ResultType) -> bool:
        """
        Puts a result in the pending results and returns True if all results are ready.
        """
        self.map_result[index] = result
        self.num_waiting -= 1
        return self.num_waiting == 0

    def spin(self) -> Request[NodeType]:
        """
        Spins the task, sending the current request result if there is one and waiting for
        the process function to yield a new request.
        """
        return self.coroutine.send(self.map_result)


class Process(Protocol[NodeType, ResultType]):
    """
    A protocol for processing a node and returning a result.
    """
    def __call__(self, queue: 'RecursionQueue[NodeType, ResultType]', node: NodeType,
                 *args, **kwargs) -> RecursionRoutine[NodeType, ResultType]:
        ...


class RecursionQueue(Generic[NodeType, ResultType]):
    """
    A queue for processing recursive tasks that may or may not be parallelized.
    """
    def __init__(self, process: Process[NodeType, ResultType],
                 executor: GenericExecutor | None = None) -> None:
        """
        Initializes the queue with a processing function and an executor.
        If the provided executor is None, we skip the overhead of managing tasks and
        directly execute the recursive task.
        """
        # If we got a SerialExecutor, avoid the overhead of managing tasks
        if executor is not None and executor.max_workers == 0:
            executor = None

        self._process = process
        self._executor = executor

        if self._executor is not None:
            self._task_queue: Queue[Task[NodeType, ResultType]] = Queue()

            self._pending_results: dict[UUID, Task[NodeType, ResultType]] = {}
            self._pending_result_lock = RLock()

            self._root_result: ResultType | None = None
            self._root_result_event = Event()

    def request(self, nodes: NodeType | Iterable[NodeType], *args, **kwargs) \
            -> Request[NodeType]:
        """
        When yielded from a process function, requests the recursion queue to compute
        results for the given nodes with the provided arguments. When the results are
        available, they are sent back to the process function as a list.
        """
        return Request(as_tuple(nodes), args, kwargs)

    def apply(self, root: NodeType | Iterable[NodeType], *args, **kwargs) \
            -> ResultType | list[ResultType]:
        """
        Starts the processing of the root node(s) and returns the result.
        If the executor is None, operates in serial and bypasses the queue machinery.
        """
        if self._executor is None:
            # If no executor is provided, run the process in serial
            if isinstance(root, Iterable):
                return [self._apply_serial(node, *args, **kwargs)
                        for node in as_tuple(root)]

            return self._apply_serial(root, *args, **kwargs)

        # If an executor is provided, run the process in parallel
        # self._task_queue.shu
        self._pending_results.clear()
        self._root_result = None
        self._root_result_event.clear()

        if isinstance(root, Iterable):
            # For multiple roots, we need a meta-task
            def _process_multiple_roots(queue) -> RecursionRoutine[NodeType, ResultType]:
                results = yield queue.request(as_tuple(root), *args, **kwargs)
                return results
            root_coro = _process_multiple_roots(self)
        else:
            root_coro = self._process(self, root, *args, **kwargs)

        # Create the root task and schedule it
        root_task = Task(parent_id=None, index=-1, coroutine=root_coro)
        self._schedule(root_task)

        # Wait for the final result and return it
        self._root_result_event.wait()
        return self._root_result

    def __enter__(self) -> 'RecursionQueue[NodeType, ResultType]':
        """
        Prepares the recursion queue for use in a context manager.
        """
        if self._executor is not None:
            # Start the worker threads if an executor is provided
            for _ in range(self._executor.max_workers):
                self._executor.submit(self._worker)

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Cleans up the recursion queue after use in a context manager.
        """
        if self._executor is not None:
            # Stop the worker threads by sending None tasks
            for _ in range(self._executor.max_workers):
                self._task_queue.put(None)

            # Clean up the executor
            self._executor.shutdown(wait=True)

    def _apply_serial(self, node: NodeType, *args, **kwargs) -> ResultType:
        """
        Recursively processes a node in serial, bypassing the queue machinery.
        """
        # Call the process function directly
        coro = self._process(self, node, *args, **kwargs)
        last_result: list[ResultType] | None = None
        while True:
            try:
                # Directly process each recursion request
                request = coro.send(last_result)
                last_result = [self._apply_serial(child, *request.args, **request.kwargs)
                               for child in request.nodes]

            except StopIteration as e:
                # The coroutine is done and has a result ready
                return e.value

    def _worker(self) -> None:
        """
        Worker method that processes tasks from the queue until it is empty.
        This method is intended to be run in a separate thread.
        """
        while True:
            task = self._task_queue.get()
            if task is None:
                # None task signals completion
                break

            # Mark the task as done
            self._schedule(task)

    def _schedule(self, task: Task[NodeType, ResultType]) -> None:
        try:
            # Send the result of child tasks if available
            request = task.spin()
            assert request is not None

            # If the request is empty, immediately resume (fast path)
            if len(request.nodes) == 0:
                task.set_results([])
                self._schedule(task)
                return

            # Otherwise, process the request
            task.set_waiting(len(request.nodes))
            task_id = uuid4()
            with self._pending_result_lock:
                self._pending_results[task_id] = task

            # Schedule children for execution
            for i, node in enumerate(request.nodes):
                child_coro = self._process(self, node, *request.args, **request.kwargs)
                child_task = Task(parent_id=task_id, index=i, coroutine=child_coro)

                # Enqueue all but the last child task
                if i < len(request.nodes) - 1:
                    self._task_queue.put(child_task)
                    continue

                # Fast-path the last child task
                self._schedule(child_task)

        except StopIteration as e:
            # The coroutine is done and has a result ready
            result: ResultType = e.value
            self._process_result(task.parent_id, task.index, result)

    def _process_result(self, parent_id: UUID | None, index: int, result: ResultType) -> None:
        """
        Processes the result from a completed task, updating pending results for the
        parent if necessary or finalizing execution if it's the root task.
        """
        if parent_id is None:
            # If the parent is None, this is the root task
            assert self._root_result is None, "Root result already set"
            self._root_result = result
            self._root_result_event.set()
            return

        with self._pending_result_lock:
            parent_task = self._pending_results[parent_id]
            if parent_task.put_result(index, result):
                # All results for the parent task are ready; resume it
                self._schedule(parent_task)
                self._pending_results.pop(parent_id, None)


def parallel_recursive(process: Process[NodeType, ResultType]) \
        -> Callable[[GenericExecutor | None], RecursionQueue[NodeType, ResultType]]:
    """
    A decorator for constructing a `RecursionQueue` that runs the provided process
    function in parallel using a supplied executor.

    The decorated function should be a coroutine that takes a `RecursionQueue` as its
    first argument, followed by the node to process and any additional arguments. It
    should yield `queue.request(...)` to request results for child nodes.

    The returned function should be used as a context manager, which will manage the
    lifecycle of the recursion queue and its executor.
    """
    def setup(*args, **kwargs) -> RecursionQueue[NodeType, ResultType]:
        return RecursionQueue(process, *args, **kwargs)

    return setup
