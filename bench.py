#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import math
import pathlib
import tempfile
import time

import cupy as cp
import kvikio.defaults
import nvtx
import zarr.api.asynchronous
import zarr.storage

from kvikio_zarr_v3 import GDSStore

ROOT = pathlib.Path(tempfile.gettempdir()) / "data.zarr"
SHAPE = (24_576, 24_576)  # 4.5 GiB in memory
CHUNKS = (4096, 4096)  # 128 MiB in memory
KERNEL = (16, 16)
xs = [slice(i * CHUNKS[0], (i + 1) * CHUNKS[0]) for i in range(SHAPE[0] // CHUNKS[0])]
ys = [slice(i * CHUNKS[1], (i + 1) * CHUNKS[1]) for i in range(SHAPE[1] // CHUNKS[1])]
NBYTES = 8 * math.prod(SHAPE)


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
    base = cp.random.randint(0, 256, size=KERNEL)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["read", "write"])
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
    args = parser.parse_args(None)

    if args.n_threads > 0:
        num_threads = kvikio.defaults.set_num_threads(args.n_threads)
    else:
        num_threads = contextlib.nullcontext()

    t0 = time.perf_counter()
    if args.action == "write":
        asyncio.run(write(args.compress, args.kvikio))
    elif args.action == "read":
        asyncio.run(read(args.compress, args.kvikio))

    t1 = time.perf_counter()

    duration = t1 - t0
    throughput = NBYTES / duration / 1e6
    store = "kvikio" if args.kvikio else "local "
    compression = "zstd" if args.compress else "none"
    task = args.action

    print(
        f"Task={task} Store={store} Compression={compression} "
        f"Throughput={throughput:0.2f} MB/s"
    )
