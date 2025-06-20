from collections import defaultdict
from collections.abc import Generator, Iterable
from enum import Enum
from itertools import groupby
from threading import Event, RLock
from types import TracebackType
from queue import Queue as TaskQueue
from typing import Protocol, override
from uuid import UUID, uuid4

from sympy import N

from devito.ir.clusters.cluster import Cluster
from devito.ir.support import IterationSpace, Scope, null_ispace
from devito.tools import as_tuple, flatten, GenericExecutor, timed_pass

__all__ = ['Queue', 'QueueStateful', 'cluster_pass']


class Prefix(IterationSpace):

    def __init__(self, ispace, guards, properties, syncs):
        super().__init__(ispace.intervals, ispace.sub_iterators, ispace.directions)

        self.guards = guards
        self.properties = properties
        self.syncs = syncs

    def __eq__(self, other):
        return (isinstance(other, Prefix) and
                super().__eq__(other) and
                self.guards == other.guards and
                self.properties == other.properties and
                self.syncs == other.syncs)

    def __hash__(self):
        return hash((self.intervals, self.sub_iterators, self.directions,
                     self.guards, self.properties, self.syncs))


class Queue:

    """
    A special queue to process Clusters based on a divide-and-conquer algorithm.

    Notes
    -----
    Subclasses must override :meth:`callback`, which may get executed either
    before (fdta -- first divide then apply) or after (fatd -- first apply
    then divide) the divide phase of the algorithm.
    """

    # Handlers for the construction of the key used in the visit
    # Some visitors may need a relaxed key to process together certain groups
    # of Clusters
    _q_ispace_in_key = True
    _q_guards_in_key = False
    _q_properties_in_key = False
    _q_syncs_in_key = False

    def callback(self, clusters: list[Cluster], prefix: IterationSpace | None,
                 **kwargs) -> list[Cluster]:
        raise NotImplementedError

    def process(self, clusters):
        return self._process_fdta(clusters, 1)

    def _make_key(self, cluster, level):
        assert self._q_ispace_in_key
        ispace = cluster.ispace[:level]

        if self._q_guards_in_key:
            try:
                guards = tuple(cluster.guards.get(i.dim) for i in ispace)
            except AttributeError:
                # `cluster` is actually a ClusterGroup
                assert len(cluster.guards) == 1
                guards = tuple(cluster.guards[0].get(i.dim) for i in ispace)
        else:
            guards = None

        if self._q_properties_in_key:
            properties = cluster.properties.drop(cluster.ispace[level:].itdims)
        else:
            properties = None

        if self._q_syncs_in_key:
            try:
                syncs = tuple(cluster.syncs.get(i.dim) for i in ispace)
            except AttributeError:
                # `cluster` is actually a ClusterGroup
                assert len(cluster.syncs) == 1
                syncs = tuple(cluster.syncs[0].get(i.dim) for i in ispace)
        else:
            syncs = None

        prefix = Prefix(ispace, guards, properties, syncs)

        subkey = self._make_key_hook(cluster, level)

        return (prefix,) + subkey

    def _make_key_hook(self, cluster, level):
        return ()

    def _process_fdta(self, clusters, level, prefix=null_ispace, **kwargs):
        """
        fdta -> First Divide Then Apply
        """
        # Divide part
        processed = []
        for k, g in groupby(clusters, key=lambda i: self._make_key(i, level)):
            pfx = k[0]
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Recursion
                processed.extend(self._process_fdta(list(g), level + 1, pfx, **kwargs))

        # Apply callback
        processed = self.callback(processed, prefix, **kwargs)

        return processed

    def _process_fatd(self, clusters, level, prefix=None, **kwargs):
        """
        fatd -> First Apply Then Divide
        """
        # Divide part
        processed = []
        for k, g in groupby(clusters, key=lambda i: self._make_key(i, level)):
            pfx = k[0]
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Apply callback
                _clusters = self.callback(list(g), pfx, **kwargs)
                # Recursion
                processed.extend(self._process_fatd(_clusters, level + 1, pfx, **kwargs))

        return processed


class QueueStateful(Queue):

    """
    A Queue carrying along some state. This is useful when one wants to avoid
    expensive re-computations of information.
    """

    class State:

        def __init__(self):
            self.properties = {}
            self.scopes = {}

    def __init__(self, state=None):
        super().__init__()
        self.state = state or QueueStateful.State()

    def _fetch_scope(self, clusters):
        exprs = flatten(c.exprs for c in as_tuple(clusters))
        key = tuple(exprs)
        if key not in self.state.scopes:
            self.state.scopes[key] = Scope(exprs)
        return self.state.scopes[key]

    def _fetch_properties(self, clusters, prefix):
        # If the situation is:
        #
        # t
        #   x0
        #     <some clusters>
        #   x1
        #     <some other clusters>
        #
        # then retain only the "common" properties, that is those along `t`
        properties = defaultdict(set)
        for c in clusters:
            v = self.state.properties.get(c, {})
            for i in prefix:
                properties[i.dim].update(v.get(i.dim, set()))
        return properties


# The type used for task identifiers in `ParallelQueue`
TaskId = UUID


def new_task_id() -> TaskId:
    """
    Returns a new unique identifier for a task.
    """
    return uuid4()


class Task:
    """
    Describes a task queued for execution in a `ParallelQueue`.
    """

    __slots__ = ('parent_id', 'index', 'clusters', 'level', 'prefix', 'kwargs',
                 'is_continuation')

    def __init__(self, parent_id: TaskId | None, index: int, clusters: list[Cluster],
                 level: int, prefix: IterationSpace | None, **kwargs) -> None:
        # Parent task identifier, or None for the root task
        self.parent_id = parent_id
        # Index in the parent's requested task group
        self.index = index

        # Arguments for process + callback
        self.clusters = clusters
        self.level = level
        self.prefix = prefix
        self.kwargs = kwargs

        # Whether this is a continuation (i.e. has received results from children)
        self.is_continuation = False

    def args(self) -> tuple[list[Cluster], int, IterationSpace | None]:
        """
        Returns the arguments to be passed to the `ParallelCallback`.
        """
        return self.clusters, self.level, self.prefix

    def put_results(self, results: list[Cluster]) -> None:
        """
        Receives the results of child tasks and marks this task as a continuation.
        """
        if self.is_continuation:
            raise RuntimeError(f"Task {self} is a continuation")

        self.clusters = results
        self.is_continuation = True


class TaskResults:
    """
    Describes a group of task results that a parent `Task` is waiting for.
    """

    __slots__ = ('parent_task', 'results', 'num_waiting')

    def __init__(self, parent_task: Task, num_tasks: int = 0):
        self.parent_task = parent_task
        self.results: list[list[Cluster] | None] = [None] * num_tasks
        self.num_waiting = num_tasks

    def append(self, result: list[Cluster] | None) -> None:
        """
        Appends a result (or None value for a task we're waiting for) to
        the results list.
        """
        self.results.append(result)
        if result is None:
            self.num_waiting += 1

    def ready(self) -> bool:
        """
        Returns True if all results have been received.
        """
        return self.num_waiting == 0

    def put_result(self, index: int, result: list[Cluster]) -> bool:
        """
        Places a result in the task group and returns True if all results
        have been received.
        """
        if self.results[index] is not None:
            raise RuntimeError(f"Result {index} already exists for {self.parent_task}")
        self.results[index] = result
        self.num_waiting -= 1

        return self.ready()

    def __len__(self) -> int:
        """
        Returns the number of results in this group.
        """
        return len(self.results)

    def flatten(self) -> list[Cluster]:
        """
        Flattens the results into a single list of Clusters.
        """
        if not self.ready():
            raise RuntimeError("Not all results are ready yet")

        return flatten(self.results)


class ParallelQueue(Queue):

    """
    A `Queue` that can be used in parallel.

    Exposes the same interface as `Queue`, but can be spun up with a `GenericExecutor`
    for parallel execution. The queue should then be used in a context manager and will
    manage the lifecycle of its attached executor.

    .. code-block:: python

        queue = SomeQueue(...)
        with queue.start_threaded(executor):
            queue.process(clusters)

        # or...
        with SomeQueue(...).start_threaded(executor) as queue:
            queue.process(clusters)
    """

    class Mode(str, Enum):
        """
        Processing modes for the `ParallelQueue`.
        """
        APPLY_THEN_DIVIDE = 'fatd'  # First Apply Then Divide
        DIVIDE_THEN_APPLY = 'fdta'  # First Divide Then Apply
        NONE = 'unset'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Executor for worker threads
        self._executor: GenericExecutor | None = None

        # Queue of tasks awaiting an available worker thread
        self._task_queue: TaskQueue[Task | None] | None = TaskQueue()

        # Map of parent task IDs to the results they're waiting for
        self._tasks_waiting: dict[TaskId, TaskResults] = {}
        self._tasks_waiting_lock = RLock()

        # Root result + event for marking completion of the root task
        self._root_task_result: list[Cluster] | None = None
        self._root_task_event = Event()

        # Current processing mode, triggered by a call to one of the process methods
        self._mode = ParallelQueue.Mode.NONE

    def start_threaded(self, executor: GenericExecutor) -> 'ParallelQueue':
        """
        Starts the queue in a threaded context, using the provided `executor`.
        """
        if self._executor is not None:
            raise RuntimeError("ParallelQueue already has an executor attached")

        self._executor = executor

        return self

    def __enter__(self) -> 'ParallelQueue':
        """
        Enters the threaded context, spinning up a number of worker threads equal to
        the executor's `max_workers`.
        """
        if self._executor is None:
            raise ValueError("ParallelQueue must be initialized with an executor")
        if self._executor.max_workers < 1:
            raise ValueError("Executor must have at least one worker thread")

        # Spin up workers equal to the number of available threads
        for _ in range(self._executor.max_workers):
            self._executor.submit(self._worker)

        return self

    def __exit__(self, exc_type: type[Exception], exc_value: Exception,
                 traceback: TracebackType) -> None:
        """
        Exits the threaded context, cleaning up the executor and waiting for all
        tasks to finish.
        """
        if self._executor is None:
            raise RuntimeError("ParallelQueue executor was cleared or not initialized")

        # Signal worker threads and wait for them to shut down
        for _ in range(self._executor.max_workers):
            self._task_queue.put(None)

        self._executor.shutdown(wait=True)

    @override
    def _process_fatd(self, clusters: list[Cluster], level: int,
                      prefix: IterationSpace = null_ispace, **kwargs) -> list[Cluster]:
        """
        Processes a list of clusters in parallel with an apply-then-divide strategy.
        """
        self._mode = ParallelQueue.Mode.APPLY_THEN_DIVIDE
        return self._process(clusters, level, prefix, **kwargs)

    @override
    def _process_fdta(self, clusters: list[Cluster], level: int,
                      prefix: IterationSpace = null_ispace, **kwargs) -> list[Cluster]:
        """
        Processes a list of clusters in parallel with a divide-then-apply strategy.
        """
        self._mode = ParallelQueue.Mode.DIVIDE_THEN_APPLY
        return self._process(clusters, level, prefix, **kwargs)

    def _process(self, clusters: list[Cluster], level: int,
                      prefix: IterationSpace = null_ispace, **kwargs) -> list[Cluster]:
        """
        Processes a list of clusters in parallel with the currently set processing mode.
        """
        if self._executor is None:
            raise RuntimeError("ParallelQueue must be started with an executor")
        if self._task_queue.qsize() > 0:
            raise RuntimeError("Task queue is nonempty; something's gone terribly wrong")

        # Clean up from a potential previous run
        self._tasks_waiting.clear()
        self._root_task_result = None
        self._root_task_event.clear()

        # Create the root task and enqueue it
        root_task = Task(parent_id=None, index=-1, clusters=clusters, level=level,
                         prefix=prefix, **kwargs)
        self._task_queue.put(root_task)

        # Wait for the root task to finish processing
        self._root_task_event.wait()

        # Unset processing mode and return the root result
        self._mode = ParallelQueue.Mode.NONE
        return self._root_task_result

    def _divide(self, task_id: TaskId | None, task: Task) -> tuple[list[Task], TaskResults]:
        """
        Divide step. Returns a list of child tasks, if any, and a `TaskResults`
        object to await the results of the child tasks.
        """
        tasks: list[Task] = []
        results = TaskResults(parent_task=task)

        clusters, level, _ = task.args()
        for k, g in groupby(clusters, key=lambda i: self._make_key(i, level)):
            pfx = k[0]
            if level > len(pfx):
                # Base case; add the clusters directly to the results
                results.append(list(g))
                continue

            # Recursion; create a new task for the sub-clusters
            child_task = Task(parent_id=task_id, index=len(results),
                              clusters=list(g), level=level + 1, prefix=pfx,
                              **task.kwargs)

            # Add the child task to the task list and a placeholder to the results
            tasks.append(child_task)
            results.append(None)

        # Return the results object and the list of tasks to schedule
        return tasks, results

    def _process_task(self, task: Task) -> None:
        """
        Processes a single task depending on the current processing mode.
        """
        if task.is_continuation:
            # The task is a continuation (i.e. we already did the divide step)
            if self._mode == ParallelQueue.Mode.DIVIDE_THEN_APPLY:
                # In FDTA, apply the callback after the divide step
                task.clusters = self.callback(*task.args(), **task.kwargs)

            # At this point the task is ready to be resolved
            # If this is the root task, store its result
            if task.parent_id is None:
                self._root_task_result = task.clusters
                self._root_task_event.set()
                return

            # Otherwise, send results back to the parent task
            with self._tasks_waiting_lock:
                results = self._tasks_waiting.get(task.parent_id)
                if results is None:
                    raise RuntimeError(f"Parent task {task.parent_id} not found")

                if results.put_result(task.index, task.clusters):
                    # If all results are ready, process them
                    self._tasks_waiting.pop(task.parent_id, None)
                    self._resolve(results)

            # Done processing continuation task
            return

        # The task is not a continuation
        if self._mode == ParallelQueue.Mode.APPLY_THEN_DIVIDE:
            # In FATD, apply the callback before the divide step
            task.clusters = self.callback(*task.args(), **task.kwargs)

        # Assign an identifier and divide into sub-tasks
        task_id = new_task_id()
        child_tasks, child_results = self._divide(task_id, task)

        # If the task is ready, immediately resolve the continuation (fast path)
        if child_results.ready():
            task.put_results(child_results.flatten())
            self._process_task(task)
            return

        # If not ready, place results in the waiting map and enqueue child tasks
        with self._tasks_waiting_lock:
            self._tasks_waiting[task_id] = child_results
        map(self._task_queue.put, child_tasks)


    def _resolve(self, results: TaskResults) -> None:
        """
        Processes the results of a group of tasks that a parent task was waiting for,
        enqueueing a continuation of the parent task.
        """
        parent_task = results.parent_task
        parent_task.put_results(results.flatten())

        # Enqueue the continuation of the parent task
        self._task_queue.put(parent_task)


    def _worker(self) -> None:
        """
        Worker method to be run in each thread.
        """
        while True:
            task = self._task_queue.get()
            if task is None:
                # Stop signal
                break

            # Process the task based on the current mode
            self._process_task(task)


class cluster_pass:

    def __new__(cls, *args, mode='dense'):
        if args:
            if len(args) == 1:
                func, = args
            elif len(args) == 2:
                func, mode = args
            else:
                assert False
            obj = object.__new__(cls)
            obj.__init__(func, mode)
            return obj
        else:
            def wrapper(func):
                return cluster_pass(func, mode)
            return wrapper

    def __init__(self, func, mode='dense'):
        self.func = func

        if mode == 'dense':
            self.cond = lambda c: (c.is_dense or not c.is_sparse) and not c.is_wild
        elif mode == 'sparse':
            self.cond = lambda c: c.is_sparse and not c.is_wild
        else:
            self.cond = lambda c: True

    def __call__(self, *args, **kwargs):
        if timed_pass.is_enabled():
            maybe_timed = lambda *_args: \
                timed_pass(self.func, self.func.__name__)(*_args, **kwargs)
        else:
            maybe_timed = lambda *_args: self.func(*_args, **kwargs)
        args = list(args)
        maybe_clusters = args.pop(0)
        if isinstance(maybe_clusters, Iterable):
            # Instance method
            processed = [maybe_timed(c, *args) if self.cond(c) else c
                         for c in maybe_clusters]
        else:
            # Pure function
            self = maybe_clusters
            clusters = args.pop(0)
            processed = [maybe_timed(self, c, *args) if self.cond(c) else c
                         for c in clusters]
        return flatten(processed)
