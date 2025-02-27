#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import functools
import math
import pathlib
import statistics
import tempfile
import time

import cupy as cp
import kvikio.defaults
import nvtx
import rich.align
import rich.progress
import rich.table
import zarr.api.asynchronous
import zarr.storage

from kvikio_zarr_v3 import GDSStore

ROOT = pathlib.Path(tempfile.gettempdir()) / "data.zarr"
SHAPE = (10, 16, 640, 1280)  # ...
CHUNKS = (1, 1, 640, 1280)  # ...
KERNEL = (1, 1, 16, 32)
DTYPE = "i4"
NBYTES = 8 * math.prod(SHAPE)


def benchmark(func=None, *, n_runs: int = 3) -> tuple[float, float]:
    if func is None:
        return functools.partial(benchmark, n_runs=n_runs)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            await func(*args, **kwargs)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        if len(times) == 1:
            return times[0], 0.0
        else:
            return statistics.median(times), statistics.stdev(times)

    return wrapper


@benchmark
async def write(compress: bool, use_kvikio: bool):
    if use_kvikio:
        store = GDSStore(ROOT)
    else:
        store = zarr.storage.LocalStore(ROOT)
    if store.root.exists():
        await store.clear()

    if compress:
        kwargs = {}
    else:
        kwargs = {"compressors": None, "filters": None}

    # TODO: various strategies for generating data, some of which are
    # hopefully friendlier to zstd compression.
    base = cp.random.randint(0, 256, size=KERNEL, dtype=DTYPE)
    n_reps = tuple(
        (c // k) * (s // c) for s, c, k in zip(SHAPE, CHUNKS, KERNEL, strict=False)
    )
    data = cp.tile(base, n_reps)
    assert data.shape == SHAPE  # noqa: S101

    data = cp.random.uniform(size=SHAPE)
    with zarr.config.enable_gpu(), zarr.config.set({"codec_pipeline.batch_size": 64}):
        a = await zarr.api.asynchronous.create_array(
            store, name="a", shape=SHAPE, chunks=CHUNKS, dtype="f8", **kwargs
        )
        nvtx.mark(message="Benchmark start")
        await a.setitem(slice(None), data)


@benchmark
async def read(compress: bool, use_kvikio: bool):
    if use_kvikio:
        store = GDSStore(ROOT)
    else:
        store = zarr.storage.LocalStore(ROOT)

    with zarr.config.enable_gpu(), zarr.config.set({"codec_pipeline.batch_size": 32}):
        g = await zarr.api.asynchronous.open_group(store=store)
        a = await g.get("a")
        if compress:
            assert len(a.metadata.codecs) == 2  # noqa: S101

        nvtx.mark(message="Benchmark start")
        await a.getitem(slice(None))


def search_num_threads():
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["read", "write", "all"])
    parser.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        help="Whether to compress the data",
    )
    parser.add_argument(
        "--kvikio",
        action=argparse.BooleanOptionalAction,
        help="Whether to compress the data",
    )
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument(
        "--profiling",
        action=argparse.BooleanOptionalAction,
        help="Whether you're profiling.",
    )
    args = parser.parse_args(None)

    if args.n_threads > 0:
        num_threads = kvikio.defaults.set_num_threads(args.n_threads)
    else:
        num_threads = contextlib.nullcontext()

    match args.action:
        case "read" | "write":
            f = write if args.action == "write" else read
            with num_threads:
                if args.profiling:
                    duration, stdev = asyncio.run(
                        benchmark(f.__wrapped__, n_runs=1)(args.compress, args.kvikio)
                    )
                else:
                    duration, stdev = asyncio.run(f(args.compress, args.kvikio))

            throughput = NBYTES / duration / 1e6
            store = "kvikio" if args.kvikio else "local "
            compression = "zstd" if args.compress else "none"
            task = args.action

            print(
                f"Task={task} Store={store} Compression={compression} "
                f"Throughput={throughput:0.2f} MB/s"
            )
        case "all":
            records = []
            combinations = [
                (compress, use_kvikio)
                for compress in [True, False]
                for use_kvikio in [True, False]
            ]
            for compress, use_kvikio in rich.progress.track(combinations):
                cname = "zstd" if compress else "none"
                sname = "kvikio" if use_kvikio else "local"

                with num_threads:
                    w_duration, w_stdev = asyncio.run(write(compress, use_kvikio))
                    r_duration, r_stdev = asyncio.run(read(compress, use_kvikio))

                records.append(
                    ("write", sname, cname, w_duration, NBYTES / w_duration / 1e6)
                )

                records.append(
                    ("read", sname, cname, r_duration, NBYTES / r_duration / 1e6)
                )

            records = sorted(records, key=lambda x: x[:3])
            t = rich.table.Table(
                "Task",
                "Store",
                "Compression",
                "Duration",
                "Effective Throughput (MB/s)",
            )
            for record in records:
                task, cname, sname, duration, throughput = record
                t.add_row(
                    cname,
                    sname,
                    task,
                    f"{duration:0.2f}",
                    rich.align.Align(f"{throughput:0.2f}", "right"),
                )
            rich.print(t)
