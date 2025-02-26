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