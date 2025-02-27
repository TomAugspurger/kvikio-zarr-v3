# kvikio-zarr-v3

Just a prototype

## Install uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install this

```
git clone http://github.com/TomAugspurger/kvikio-zarr-v3
uv run bench.py write
```

## Results

```
# with compression
KVIKIO_NTHREADS=64 ./bench.py write --compress --kvikio
KVIKIO_NTHREADS=64 ./bench.py write --compress --no-kvikio
KVIKIO_NTHREADS=64 ./bench.py read --compress --kvikio
KVIKIO_NTHREADS=64 ./bench.py read --compress --no-kvikio

# no compression
KVIKIO_NTHREADS=64 ./bench.py write --no-compress --kvikio
KVIKIO_NTHREADS=64 ./bench.py write --no-compress --no-kvikio
KVIKIO_NTHREADS=64 ./bench.py read --no-compress --kvikio
KVIKIO_NTHREADS=64 ./bench.py read --no-compress --no-kvikio
```

results (on a DGX of some sort):

```
# with compression
Task=write Store=kvikio Compression=zstd Throughput=82.67 MB/s
Task=write Store=local  Compression=zstd Throughput=85.99 MB/s
Task=read Store=kvikio Compression=zstd Throughput=169.32 MB/s
Task=read Store=local  Compression=zstd Throughput=182.90 MB/s

# no compression
Task=write Store=kvikio Compression=none Throughput=104.56 MB/s
Task=write Store=local  Compression=none Throughput=130.94 MB/s
Task=read Store=kvikio Compression=none Throughput=1318.49 MB/s
Task=read Store=local  Compression=none Throughput=2744.23 MB/s
```

## Profiles

```
nsys profile -t nvtx,cuda --python-sampling=true --force-overwrite=true --output=write-compressed-kvikio ./bench.py write --compress --kvikio
nsys profile -t nvtx,cuda --python-sampling=true --force-overwrite=true --output=read-compressed-kvikio ./bench.py read --compress --kvikio
```

```python
>>> import cupy as cp
>>> from nvidia import nvcomp
>>> codec = nvcomp.Codec(algorithm="Zstd")
>>> data = cp.random.randint(0, 256, size=(4096, 4096))
```