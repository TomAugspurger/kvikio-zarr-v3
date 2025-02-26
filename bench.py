#!/usr/bin/env python3
import math
import argparse
import time
import asyncio
import zarr.storage
import zarr.api.asynchronous
import cupy as cp
import nvtx
from kvikio_zarr_v3 import GDSStore


ROOT = "/tmp/data.zarr"
# SHAPE = (16_384, 16_384)
SHAPE = (24_576, 24_576)  # 4.5 GiB in memory
CHUNKS = (4096, 4096)  # 128 MiB in memory
KERNEL = (16, 16)
xs = [slice(i * CHUNKS[0], (i + 1) * CHUNKS[0]) for i in range(SHAPE[0] // CHUNKS[0])]
ys = [slice(i * CHUNKS[1], (i + 1) * CHUNKS[1]) for i in range(SHAPE[1] // CHUNKS[1])]
NBYTES = 8 * math.prod(SHAPE)


# async def write(concurrent: bool=True):
#     store = kvikio.zarr_v3.GDSStore(ROOT)
#     if store.root.exists():
#         await store.clear()

#     with zarr.config.enable_gpu():
#         tasks = []
#         a = await zarr.api.asynchronous.create_array(store, name="a", shape=SHAPE, chunks=CHUNKS, dtype="f8")
#         nvtx.mark(message="Benchmark start")
#         for x in xs:
#             for y in ys:
#                 coro = a.setitem((x, y), cp.random.uniform(size=CHUNKS))
#                 if concurrent:
#                     tasks.append(coro)
#                 else:
#                     await coro
#         if concurrent:
#             await asyncio.gather(*tasks)


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
    n_reps = tuple((c // k) * (s // c) for s, c, k in zip(SHAPE, CHUNKS, KERNEL))
    data = cp.tile(base, n_reps)
    assert data.shape == SHAPE

    data = cp.random.uniform(size=SHAPE)
    with zarr.config.enable_gpu():
        a = await zarr.api.asynchronous.create_array(store, name="a", shape=SHAPE, chunks=CHUNKS, dtype="f8", **kwargs)
        nvtx.mark(message="Benchmark start")
        await a.setitem(slice(None), data)


# async def read(concurrent: bool = True):
#     store = kvikio.zarr_v3.GDSStore(ROOT)

#     with zarr.config.enable_gpu():
#         tasks = []
#         g = await zarr.api.asynchronous.open_group(store=store)
#         a = await g.get("a")
#         nvtx.mark(message="Benchmark start")
#         for x in xs:
#             for y in ys:
#                 coro = a.getitem((x, y))
#                 if concurrent:
#                     tasks.append(coro)
#                 else:
#                     await coro
#         if concurrent:
#             await asyncio.gather(*tasks)


async def read(compress: bool, use_kvikio: bool):
    if use_kvikio:
        store = GDSStore(ROOT)
    else:
        store = zarr.storage.LocalStore(ROOT)

    with zarr.config.enable_gpu():
        g = await zarr.api.asynchronous.open_group(store=store)
        a = await g.get("a")
        if compress:
            assert len(a.metadata.codecs) == 2

        nvtx.mark(message="Benchmark start")
        await a.getitem(slice(None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["read", "write"])
    parser.add_argument("--compress", action=argparse.BooleanOptionalAction, help="Whether to compress the data")
    parser.add_argument("--kvikio", action=argparse.BooleanOptionalAction, help="Whether to compress the data")
    args = parser.parse_args(None)

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

    print(f"Task={task} Store={store} Compression={compression} Throughput={throughput:0.2f} MB/s")

    # print(f"Throughput={dask.utils.format_bytes(throughput)}s nbytes={dask.utils.format_bytes(NBYTES)} Duration={t1-t0:0.2f}")

