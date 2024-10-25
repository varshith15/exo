### multipool.py

import asyncio
from typing import (
    Sequence,
    Any,
    Optional,
    Callable,
    Awaitable,
    Dict,
    List,
    Tuple
)
from .types import R, Queue
from .pool import Pool
from .scheduler import SubPoolRoundRobin, RoundRobin
from .core import get_context


async def _initializer(queues: List[Tuple[Queue, Queue]], pool_worker_type, initializer, initargs):
    from .shared import shared

    scheduler = RoundRobin()
    pool = Pool(start_init=False, scheduler=scheduler, worker_type=pool_worker_type)
    for tx, rx in queues:
        qid = scheduler.register_queue(tx)
        scheduler.register_process(qid)
        pool.queues[qid] = (tx, rx)
    pool.running = True
    pool.init_loop_finished_tasks()
    shared.set("runner_pool", pool)

    if initializer:
        if asyncio.iscoroutinefunction(initializer):
            await initializer(*initargs)
        else:
            initializer(*initargs)


def _prepare_initargs(initializer, initargs, tx_s, rx_s, slice_initargs: bool, num_processes: int, pool_worker_type: str) -> List:
    tx_s = list(map(list, zip(*tx_s)))
    rx_s = list(map(list, zip(*rx_s)))
    final_initargs = []

    for i in range(num_processes):
        tx, rx = tx_s[i], rx_s[i]
        queues = []
        for _tx, _rx in zip(tx, rx):
            queues.append((_tx, _rx))
        _initargs = initargs[i] if slice_initargs else initargs
        final_initargs.append([queues, pool_worker_type, initializer, _initargs])
    return final_initargs


class MultiPool:
    """
    The main idea here is to create (pool_1_num_processes * pool_2_num_processes) queues to synchronize and share data
    between the processes of the two pools
    pool_1: Will act as an executor pool.
    pool_2: Will act as a runner pool.
    """
    def __init__(self, pool_1_args: dict, pool_2_args: dict):
        pool_1_num_processes = pool_1_args["processes"]
        pool_2_num_processes = pool_2_args["processes"]
        pool_2_worker_type = pool_2_args["worker_type"]
        scheduler = SubPoolRoundRobin()
        pool_2_args["start_init"] = False
        pool_2_args["scheduler"] = scheduler
        self.pool_2 = Pool(**pool_2_args)

        tx_s, rx_s = [], []
        for _ in range(pool_2_num_processes):
            tx, rx = [], []
            for _ in range(pool_1_num_processes):
                tx.append(get_context().Queue())
                rx.append(get_context().Queue())
            qid = scheduler.register_queue(tx)
            self.pool_2.queues[qid] = (tx, rx)
            tx_s.append(tx)
            rx_s.append(rx)

        pool_1_args["start_init"] = False
        pool_1_initializer = pool_1_args.get("initializer")
        pool_1_initargs = pool_1_args.get("initargs", ())
        pool_1_slice_initargs = pool_1_args.get("slice_initargs", False)
        pool_1_args["initializer"] = _initializer
        pool_1_args["initargs"] = _prepare_initargs(pool_1_initializer, pool_1_initargs, tx_s, rx_s, pool_1_slice_initargs, pool_1_num_processes, pool_2_worker_type)
        pool_1_args["slice_initargs"] = True

        self.pool_1 = Pool(**pool_1_args)
        self.start_pool()

    async def apply(
        self,
        func: Optional[Callable[..., Awaitable[R]]] = None,
        timeout: Optional[float] = None,
        args: Sequence[Any] = None,
        kwds: Dict[str, Any] = None,
    ) -> R:
        return await self.pool_1.apply(func, timeout, args, kwds)

    def start_pool(self):
        self.pool_2.init()
        self.pool_2._loop_maintain_pool = asyncio.ensure_future(self.pool_2.loop_maintain_pool())

        self.pool_1.init()
        self.pool_1._loop_maintain_pool = asyncio.ensure_future(self.pool_1.loop_maintain_pool())
        self.pool_1._loop_finished_tasks = asyncio.ensure_future(self.pool_1.loop_collect_finished_tasks())

    def close(self) -> None:
        self.pool_1.running = False
        for qid in self.pool_1.processes.values():
            tx, _ = self.pool_1.queues[qid]
            if isinstance(tx, list):
                for _tx in tx:
                    _tx.put_nowait(None)
            else:
                tx.put_nowait(None)

    def terminate(self) -> None:
        if self.pool_1.running:
            self.close()

        for process in self.pool_1.processes:
            process.terminate()

    async def join(self) -> None:
        if self.pool_1.running:
            raise RuntimeError("pool is still open")

        await self.pool_1._loop_finished_tasks
        await self.pool_1._loop_maintain_pool


### model_decorate.py

import functools
import threading
from .pool import Pool
from .shared import shared


class ModelRegistry:
    def __init__(self):
        self.models = {}


model_registry = {}


def get_model_registry() -> ModelRegistry:
    t_id = threading.get_ident()
    if not model_registry.__contains__(t_id):
        model_registry[t_id] = ModelRegistry()
    return model_registry[t_id]


def remove_decorators(obj, target):
    cls = obj.__class__
    for attr_name in cls.__dict__:
        attr = getattr(cls, attr_name)
        if callable(attr):
            while hasattr(attr, "__wrapped__") and attr.__wrapped__.__name__ != target:
                attr = attr.__wrapped__

            if hasattr(attr, "__wrapped__") and attr.__wrapped__.__name__ == target:
                setattr(cls, attr_name, attr.__wrapped__)


def load(model_id, self, *args, **kwargs):
    reg = get_model_registry()
    if isinstance(self, list):
        self = self[0]
    if not reg.models.__contains__(model_id):
        remove_decorators(self, "load_new")
        remove_decorators(self, "predict_new")
        reg.models[model_id] = self
        reg.models[model_id].load_new(*args, **kwargs)


def predict(model_id, *args, **kwargs):
    reg = get_model_registry()
    return reg.models[model_id].predict_new(*args, **kwargs)


async def predict_async(model_id, *args, **kwargs):
    reg = get_model_registry()
    return await reg.models[model_id].predict_new(*args, **kwargs)


def load_in_worker_pool(load_fn):
    @functools.wraps(load_fn)
    def wrapped(self, *args, **kwargs):
        model_id = self.model_id
        worker_pool: Pool = shared.get("runner_pool")
        if worker_pool:
            return worker_pool.apply_in_all_processes(
                func=load, args=(model_id, self, *args), kwds=kwargs
            )
        else:
            return load_fn(self, *args, **kwargs)

    return wrapped


def async_predict_in_worker_pool(predict_fn):
    @functools.wraps(predict_fn)
    async def wrapped(self, *args, **kwargs):
        model_id = self.model_id
        worker_pool: Pool = shared.get("runner_pool")
        if worker_pool:
            return await worker_pool.apply(
                func=predict_async, args=(model_id, *args), kwds=kwargs
            )
        else:
            return await predict_fn(self, *args, **kwargs)

    return wrapped


def predict_in_worker_pool(predict_fn):
    @functools.wraps(predict_fn)
    def wrapped(self, *args, **kwargs):
        model_id = self.model_id
        worker_pool: Pool = shared.get("runner_pool")
        if worker_pool:
            return worker_pool.apply_sync(
                func=predict, args=(model_id, *args), kwds=kwargs
            )
        else:
            return predict_fn(self, *args, **kwargs)

    return wrapped


### worker.py


import asyncio
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor
from asyncio import iscoroutinefunction
from collections import defaultdict
import queue
import time
import traceback
from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    Optional,
    Sequence,
    List,
)

from .core import Process
from .types import LoopInitializer, PoolTask, TaskID, Queue


class MultiPoolWorker(Process):
    def __init__(
        self,
        tx: List[Queue],
        rx: List[Queue],
        ttl: int,
        concurrency: int,
        *,
        initializer: Optional[Callable] = None,
        initargs: Sequence[Any] = (),
        loop_initializer: Optional[LoopInitializer] = None,
        func: Optional[Callable] = None,
        **kwargs
    ) -> None:
        super().__init__(
            target=self.run,
            args=(tx, rx, func),
            initializer=initializer,
            initargs=initargs,
            loop_initializer=loop_initializer,
        )
        self.concurrency = max(1, concurrency)
        self.current_batch = []
        self.last_update = time.time()
        self.max_batch_size = 3
        self.max_batching_wait_time = 0.5

    def combine_tasks(self):
        def combine(batch):
            tasks, _rxs = list(zip(*batch))
            tids, timeouts, funcs, args, kwargs = list(zip(*tasks))

            timeout = min((to for to in timeouts if to is not None), default=None)
            func = funcs[0]  # figure out a clean segregation way
            merged_args = (args[0][0],) + tuple(
                [list(a[i] for a in args) for i in range(1, len(args[0]))]
            )
            merged_kwargs = defaultdict(list)

            for kwargs in merged_kwargs:
                for key, value in kwargs.items():
                    merged_kwargs[key].append(value)

            return tids, timeout, func, merged_args, merged_kwargs, _rxs

        def type_batched():
            batched_tasks = defaultdict(list)
            for _task in self.current_batch:
                batched_tasks[_task[0][2]].append(_task)
            self.current_batch = []
            return dict(batched_tasks)

        if not self.current_batch:
            return [(None, None, None)]

        batches = type_batched()
        _futures = []
        for _, batch in batches.items():
            tid, timeout, func, args, kwargs, _rx = combine(batch)

            func = func if func else function
            if iscoroutinefunction(func):
                future = asyncio.ensure_future(
                    asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                )
            else:
                future = asyncio.Future()
                future.set_result(func(*args, **kwargs))

            _futures.append((future, tid, _rx))

        return _futures

    async def run(
        self, tx: List[Queue], rx: List[Queue], function: Optional[Callable] = None
    ) -> None:
        pending: Dict[asyncio.Future, Tuple[TaskID, Queue]] = {}
        running = True
        queue_g = cycle(zip(tx, rx))
        break_check = 4 * len(tx)  # tune it better
        while running or pending:
            break_count = 0
            while running and len(pending) < self.concurrency:
                try:
                    _tx, _rx = next(queue_g)
                    task: PoolTask = _tx.get_nowait()
                    break_count = 0
                except queue.Empty:
                    break_count += 1
                    if break_count >= break_check:
                        break
                    continue

                if task is None:
                    running = False
                    break

                self.current_batch.append((task, _rx))
                if (len(self.current_batch) >= self.max_batch_size) or (
                    time.time() - self.last_update >= self.max_batching_wait_time
                ):
                    self.last_update = time.time()
                    break

            for future, tid, _rx in self.combine_tasks():
                if future:
                    pending[future] = tid, _rx

            if not pending:
                await asyncio.sleep(0.001)
                continue

            done, _ = await asyncio.wait(
                pending.keys(), timeout=0.05, return_when=asyncio.FIRST_COMPLETED
            )

            for future in done:
                tid, _rx = pending.pop(future)

                result = None
                tb = None
                child_exception = None
                try:
                    result = future.result()
                except BaseException as e:
                    child_exception = e
                    tb = traceback.format_exc()

                if isinstance(_rx, tuple):
                    if not result:
                        for t, x in zip(tid, _rx):
                            x.put_nowait((t, result, tb, child_exception))
                    else:
                        for r, t, x in zip(result, tid, _rx):
                            x.put_nowait((t, r, tb, child_exception))
                else:
                    _rx.put_nowait((tid, result, tb, child_exception))


class PoolWorker(Process):
    """Individual worker process for the async pool."""

    def __init__(
        self,
        tx: Queue,
        rx: Queue,
        ttl: int,
        concurrency: int,
        *,
        initializer: Optional[Callable] = None,
        initargs: Sequence[Any] = (),
        loop_initializer: Optional[LoopInitializer] = None,
        func: Optional[Callable] = None,
        **kwargs
    ) -> None:
        super().__init__(
            target=self.run,
            args=(tx, rx, func),
            initializer=initializer,
            initargs=initargs,
            loop_initializer=loop_initializer,
        )
        self.concurrency = max(1, concurrency)
        self.ttl = max(0, ttl)
        self.completed = 0

    async def run(
        self, tx: Queue, rx: Queue, function: Optional[Callable] = None
    ) -> None:
        """Pick up work, execute work, return results, rinse, repeat."""
        pending: Dict[asyncio.Future, TaskID] = {}
        running = True
        while running or pending:
            # TTL, Tasks To Live, determines how many tasks to execute before dying
            if self.ttl and self.completed >= self.ttl:
                running = False

            # pick up new work as long as we're "running" and we have open slots
            while running and len(pending) < self.concurrency:
                try:
                    task: PoolTask = tx.get_nowait()
                except queue.Empty:
                    break

                if task is None:
                    running = False
                    break

                tid, timeout, func, args, kwargs = task
                func = func if func else function
                future = asyncio.ensure_future(
                    asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                )
                pending[future] = tid

            if not pending:
                await asyncio.sleep(0.005)
                continue

            # return results and/or exceptions when completed
            done, _ = await asyncio.wait(
                pending.keys(), timeout=0.05, return_when=asyncio.FIRST_COMPLETED
            )
            for future in done:
                tid = pending.pop(future)

                result = None
                tb = None
                child_exception = None
                try:
                    result = future.result()
                except BaseException as e:
                    child_exception = e
                    tb = traceback.format_exc()

                rx.put_nowait((tid, result, tb, child_exception))
                if self.ttl:
                    self.completed += 1


class MultiThreadWorker(Process):
    def __init__(
        self,
        tx: List[Queue],
        rx: List[Queue],
        ttl: int,
        concurrency: int,
        *,
        initializer: Optional[Callable] = None,
        initargs: Sequence[Any] = (),
        loop_initializer: Optional[LoopInitializer] = None,
        func: Optional[Callable] = None,
        threads: int = 4
    ) -> None:
        super().__init__(
            target=self.run,
            args=(tx, rx, func),
            initializer=initializer,
            initargs=initargs,
            loop_initializer=loop_initializer,
        )
        self.concurrency = max(1, concurrency)
        self.threads = threads

    async def run(
        self, tx: List[Queue], rx: List[Queue], function: Optional[Callable] = None
    ) -> None:
        thread_pool = ThreadPoolExecutor(max_workers=self.threads)
        pending: Dict[asyncio.Future, Tuple[TaskID, Queue]] = {}
        running = True
        queue_g = cycle(zip(tx, rx))
        break_check = 10 * len(tx)
        while running or pending:
            break_count = 0
            while running and len(pending) < self.concurrency:
                try:
                    _tx, _rx = next(queue_g)
                    task: PoolTask = _tx.get_nowait()
                    break_count = 0
                except queue.Empty:
                    break_count += 1
                    if break_count >= break_check:
                        break
                    continue

                if task is None:
                    running = False
                    break

                tid, timeout, func, args, kwargs = task
                func = func if func else function
                future = asyncio.ensure_future(
                    asyncio.get_event_loop().run_in_executor(thread_pool, func, *args)
                )
                pending[future] = tid, _rx

            if not pending:
                await asyncio.sleep(0.001)
                continue

            done, _ = await asyncio.wait(
                pending.keys(), timeout=0.05, return_when=asyncio.FIRST_COMPLETED
            )
            for future in done:
                tid, _rx = pending.pop(future)
                result = None
                tb = None
                child_exception = None
                try:
                    result = future.result()
                except BaseException as e:
                    child_exception = e
                    tb = traceback.format_exc()

                _rx.put_nowait((tid, result, tb, child_exception))


### pool.py

import asyncio
import logging
import os
import queue
import traceback
import time
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    List
)

from .core import Process, get_manager, get_context
from .scheduler import RoundRobin, Scheduler
from .types import (
    LoopInitializer,
    ProxyException,
    QueueID,
    R,
    TaskID,
    TracebackStr,
)

from multiprocessing import Queue

MAX_TASKS_PER_CHILD = 0  # number of tasks to execute before recycling a child process
CHILD_CONCURRENCY = 16  # number of tasks to execute simultaneously per child process
_T = TypeVar("_T")

log = logging.getLogger(__name__)


class PoolResult(Awaitable[Sequence[_T]], AsyncIterable[_T]):
    """
    Asynchronous proxy for map/starmap results. Can be awaited or used with `async for`.
    """

    def __init__(self, pool: "Pool", task_ids: Sequence[TaskID]):
        self.pool = pool
        self.task_ids = task_ids

    def __await__(self) -> Generator[Any, None, Sequence[_T]]:
        """Wait for all results and return them as a sequence"""
        return self.results().__await__()

    async def results(self) -> Sequence[_T]:
        """Wait for all results and return them as a sequence"""
        return await self.pool.results(self.task_ids)

    def __aiter__(self) -> AsyncIterator[_T]:
        """Return results one-by-one as they are ready"""
        return self.results_generator()

    async def results_generator(self) -> AsyncIterator[_T]:
        """Return results one-by-one as they are ready"""
        for task_id in self.task_ids:
            yield (await self.pool.results([task_id]))[0]


async def _initializer(initializer, initargs, submitted_tasks):
    if initializer:
        if asyncio.iscoroutinefunction(initializer):
            await initializer(*initargs)
        else:
            initializer(*initargs)
    for _func, _args, _kwds in submitted_tasks:
        await _func(*_args, **_kwds)


class Pool:
    """Execute coroutines on a pool of child processes."""

    def __init__(
        self,
        processes: int = None,
        initializer: Callable = None,
        initargs: Sequence[Any] = (),
        maxtasksperchild: int = MAX_TASKS_PER_CHILD,
        childconcurrency: int = CHILD_CONCURRENCY,
        scheduler: Scheduler = None,
        loop_initializer: Optional[LoopInitializer] = None,
        func: Optional[Callable] = None,
        start_init: bool = True,
        worker_type: str = "default",
        slice_initargs: bool = False,
        threads: int = 4,
        **kwargs
    ) -> None:
        self.context = get_context()
        self.scheduler = scheduler or RoundRobin()
        self.process_count = max(1, processes or os.cpu_count() or 2)

        self.slice_initargs = slice_initargs
        self.initializer = initializer
        self.initargs = initargs
        if self.initargs and self.slice_initargs and len(self.initargs) != self.process_count:
            raise ValueError("slice_initargs is set to True but len(initargs) differs from process count.")

        self.loop_initializer = loop_initializer
        self.maxtasksperchild = max(0, maxtasksperchild)
        self.childconcurrency = max(1, childconcurrency)

        self.processes: Dict[Process, QueueID] = {}
        self.queues: Dict[QueueID, Tuple[Union[Queue, List[Queue]], Union[Queue, List[Queue]]]] = {}

        self.running = False
        self.last_id = 0
        self._results: Dict[TaskID, Tuple[Any, Optional[TracebackStr], Optional[BaseException]]] = {}

        self.submitted_tasks = []
        self.func = func
        self.threads = threads

        self.last_restart_time = 0
        self.restart_interval = kwargs.get("restart_interval", 5 * 60)
        self.cancelled = set()
        self.worker_type = worker_type

        if start_init:
            self.init()
            self._loop_maintain_pool = asyncio.ensure_future(self.loop_maintain_pool())
            self._loop_finished_tasks = asyncio.ensure_future(self.loop_collect_finished_tasks())

    async def __aenter__(self) -> "Pool":
        """Enable `async with Pool() as pool` usage."""
        return self

    async def __aexit__(self, *args) -> None:
        """Automatically terminate the pool when falling out of scope."""
        self.terminate()
        await self.join()

    def init(self) -> None:
        """
        Create the initial mapping of processes and queues.

        :meta private:
        """
        if not self.queues:
            for _ in range(self.process_count):
                tx = self.context.Queue()
                rx = self.context.Queue()
                qid = self.scheduler.register_queue(tx)
                self.queues[qid] = (tx, rx)

        qids = list(self.queues.keys())
        for i in range(self.process_count):
            qid = qids[i]
            self.processes[self.create_worker(qid)] = qid
            self.scheduler.register_process(qid)

        for process in self.processes:
            while not process.is_started:
                time.sleep(0.1)
        self.running = True

    def init_loop_finished_tasks(self):
        self._loop_finished_tasks = asyncio.ensure_future(self.loop_collect_finished_tasks())

    async def loop_maintain_pool(self) -> None:
        """
        Maintain the pool of workers while open.

        :meta private:
        """
        while self.processes or self.running:
            # clean up workers that reached TTL
            for process in list(self.processes):
                if not process.is_alive():
                    qid = self.processes.pop(process)
                    if self.running:
                        process = self.create_worker(qid)
                        self.processes[process] = qid

            # let someone else do some work for once
            await asyncio.sleep(0.005)

    async def loop_collect_finished_tasks(self) -> None:
        while self.running:
            # pull results into a shared dictionary for later retrieval
            for _, rx in self.queues.values():
                while True:
                    try:
                        task_id, value, tb, ce = rx.get_nowait()
                        self.finish_work(task_id, value, tb, ce)

                    except queue.Empty:
                        break

            # let someone else do some work for once
            await asyncio.sleep(0.005)

    def create_worker(self, qid: QueueID) -> Process:
        """
        Create a worker process attached to the given transmit and receive queues.

        :meta private:
        """
        if self.worker_type == "multipool":
            from .worker import MultiPoolWorker as PoolWorker
        elif self.worker_type == "multithread":
            from .worker import MultiThreadWorker as PoolWorker
        else:
            from .worker import PoolWorker
        tx, rx = self.queues[qid]

        initargs = self.initargs
        if self.initargs and self.slice_initargs:
            initargs = self.initargs[qid]

        process = PoolWorker(
            tx,
            rx,
            self.maxtasksperchild,
            self.childconcurrency,
            initializer=_initializer,
            initargs=[self.initializer, initargs, self.submitted_tasks],
            loop_initializer=self.loop_initializer,
            func=self.func,
            threads=self.threads
        )
        process.start()
        return process

    def queue_work(
        self,
        func: Optional[Callable[..., Awaitable[R]]],
        timeout: Optional[float],
        args: Sequence[Any],
        kwargs: Dict[str, Any],
    ) -> TaskID:
        """
        Add a new work item to the outgoing queue.

        :meta private:
        """
        self.last_id += 1
        task_id = TaskID(self.last_id)
        qid = self.scheduler.schedule_task(task_id, func, args, kwargs)
        tx, _ = self.queues[qid]
        tx.put_nowait((task_id, timeout, func, args, kwargs))
        return task_id

    def finish_work(
        self, task_id: TaskID, value: Any, tb: Optional[TracebackStr], ce: Optional[BaseException]
    ) -> None:
        """
        Mark work items as completed.

        :meta private:
        """
        if not (task_id in self.cancelled):
            self._results[task_id] = value, tb, ce
            self.scheduler.complete_task(task_id)
        else:
            self.cancelled.remove(task_id)

    async def results(self, tids: Sequence[TaskID]) -> Sequence[R]:
        """
        Wait for all tasks to complete, and return results, preserving order.

        :meta private:
        """
        pending = set(tids)
        ready: Dict[TaskID, R] = {}

        while pending:
            for tid in pending.copy():
                if tid in self._results:
                    result, tb, ce = self._results.pop(tid)
                    if tb is not None:
                        raise ProxyException(ce, tb)
                    ready[tid] = result
                    pending.remove(tid)

            try:
                await asyncio.sleep(0.005)
            except asyncio.CancelledError:
                self.cancelled.update(pending)
                raise

        return [ready[tid] for tid in tids]

    async def apply(
        self,
        func: Optional[Callable[..., Awaitable[R]]] = None,
        timeout: Optional[float] = None,
        args: Sequence[Any] = None,
        kwds: Dict[str, Any] = None,
    ) -> R:
        """Run a single coroutine on the pool."""
        if not self.running:
            raise RuntimeError("pool is closed")

        args = args or ()
        kwds = kwds or {}

        tid = self.queue_work(func, timeout, args, kwds)
        results: Sequence[R] = await self.results([tid])
        return results[0]

    def apply_sync(
        self,
        func: Optional[Callable[..., Awaitable[R]]] = None,
        timeout: Optional[float] = None,
        args: Sequence[Any] = None,
        kwds: Dict[str, Any] = None,
    ) -> R:
        """Run a single coroutine on the pool."""
        if not self.running:
            raise RuntimeError("pool is closed")

        args = args or ()
        kwds = kwds or {}

        self.last_id += 1
        task_id = TaskID(self.last_id)
        qid = self.scheduler.schedule_task(task_id, func, args, kwds)
        tx, rx = self.queues[qid]
        tx.put_nowait((task_id, timeout, func, args, kwds))
        task_id, value, tb, ce = rx.get()
        if tb is not None:
            raise ProxyException(ce, tb)
        return value

    def apply_in_all_processes(
        self,
        func: Optional[Callable[..., Awaitable[R]]] = None,
        timeout: Optional[float] = None,
        args: Sequence[Any] = None,
        kwds: Dict[str, Any] = None,
    ) -> R:
        if not self.running:
            raise RuntimeError("pool is closed")

        args = args or ()
        kwds = kwds or {}

        iteration = 1
        if self.worker_type == "multithread":
            iteration = 20  # Need a better guaranteed solution

        for qid in list(self.queues.keys()):
            tx, rx = self.queues[qid]
            tasks = set()
            for _ in range(iteration):
                self.last_id += 1
                task_id = TaskID(self.last_id)
                tx.put_nowait((task_id, timeout, func, args, kwds))
                tasks.add(task_id)

            while tasks:
                task_id, value, tb, ce = rx.get()
                tasks.remove(task_id)
                if tb is not None:
                    raise ProxyException(ce, tb)
        return None

    def close(self) -> None:
        """Close the pool to new visitors."""
        self.running = False
        for qid in self.processes.values():
            tx, _ = self.queues[qid]
            tx.put_nowait(None)

    def terminate(self) -> None:
        """No running by the pool!"""
        if self.running:
            self.close()

        for process in self.processes:
            process.terminate()

    async def join(self) -> None:
        """Wait for the pool to finish gracefully."""
        if self.running:
            raise RuntimeError("pool is still open")

        await self._loop_finished_tasks
        await self._loop_maintain_pool

    def restart(self):
        if (time.time() - self.last_restart_time) >= self.restart_interval:
            self.last_restart_time = time.time()
            for process in list(self.processes):
                process.terminate()
                process.join()