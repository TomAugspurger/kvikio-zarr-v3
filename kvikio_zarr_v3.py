# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import asyncio
import contextlib
import os
from pathlib import Path

import kvikio
import zarr.storage
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.buffer.core import default_buffer_prototype

try:
    import nvtx
except ImportError:
    HAS_NVTX = False
else:
    HAS_NVTX = True


if HAS_NVTX:
    annotate = nvtx.annotate
else:
    annotate = contextlib.nullcontext


def _get(
    path: Path, prototype: BufferPrototype, byte_range: ByteRequest | None
) -> Buffer:
    file_size = os.path.getsize(path)
    file_offset: int

    match byte_range:
        case None:
            nbytes = file_size
            file_offset = 0
        case OffsetByteRequest():  # type: ignore[misc]
            nbytes = max(0, file_size - byte_range.offset)
            file_offset = byte_range.offset
        case RangeByteRequest():  # type: ignore[misc]
            nbytes = byte_range.end - byte_range.start
            file_offset = byte_range.start
        case SuffixByteRequest():  # type: ignore[misc]
            nbytes = byte_range.suffix
            file_offset = max(0, file_size - byte_range.suffix)
        case _:
            # This isn't allowed by mypy, but the tests assert we raise
            # something here.
            raise TypeError(f"Unexpected byte_range, got {byte_range}")

    # kvikio doesn't support reading past the end of a file. Some zarr tests
    # rely on this behavior: to "read" 3 bytes out of a 0 byte file, or to
    # "seek" past the end of a file with file_offset. The semantics seem to
    # be roughly the same as slicing an empty bytestring.

    nbytes = min(nbytes, file_size)
    file_offset = min(file_offset, file_size)

    raw = prototype.nd_buffer.create(shape=(nbytes,), dtype="b").as_ndarray_like()
    buf = prototype.buffer.from_array_like(raw)

    with kvikio.CuFile(path) as f:
        # Note: this currently creates an IOFuture and then blocks
        # on reading it. The blocking read means this is in a regular
        # sync function, and so this must be run in a threadpool.
        future = f.pread(raw, size=nbytes, file_offset=file_offset)
        future.get()  # blocks

    return buf


def _put(
    path: Path,
    value: Buffer,
    start: int | None = None,
    exclusive: bool = False,
) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if start is not None:
        with kvikio.CuFile(path, "r+b") as f:
            # TODO: seems like this isn't tested
            # https://github.com/zarr-developers/zarr-python/issues/2859
            f.write(value.as_array_like(), file_offset=start)
        return None
    else:
        buf = value.as_array_like()
        if exclusive:
            if path.exists():
                raise FileExistsError(f"File exists: {path}")
        mode = "wb"
        with kvikio.CuFile(path, flags=mode) as f:
            return f.write(buf)


class GDSStore(zarr.storage.LocalStore):
    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self}')"

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        if prototype is None:
            prototype = default_buffer_prototype()
        if not self._is_open:
            await self._open()
        path = self.root / key

        if HAS_NVTX:
            kwargs = {"message": "kvikio.zarr.get", "domain": "Zarr"}
        else:
            kwargs = {}
        try:
            with annotate(**kwargs):
                return await asyncio.to_thread(_get, path, prototype, byte_range)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        return await self._set(key, value)

    async def _set(self, key: str, value: Buffer, exclusive: bool = False) -> None:
        if not self._is_open:
            await self._open()
        self._check_writable()
        if not isinstance(value, Buffer):
            raise TypeError(
                f"LocalStore.set(): `value` must be a Buffer instance. Got an "
                f"instance of {type(value)} instead."
            )
        path = self.root / key

        if HAS_NVTX:
            kwargs = {"message": "kvikio.zarr.get", "domain": "Zarr"}
        else:
            kwargs = {}

        with annotate(**kwargs):
            await asyncio.to_thread(_put, path, value, start=None, exclusive=exclusive)
