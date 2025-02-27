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
./bench.py all

┏━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Task   ┃ Store ┃ Compression ┃ Duration ┃ Effective Throughput (MB/s) ┃
┡━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ kvikio │ none  │ read        │ 0.75     │                     2809.74 │
│ kvikio │ zstd  │ read        │ 2.10     │                      999.20 │
│ local  │ none  │ read        │ 0.45     │                     4695.28 │
│ local  │ zstd  │ read        │ 1.69     │                     1240.97 │
│ kvikio │ none  │ write       │ 11.53    │                      181.92 │
│ kvikio │ zstd  │ write       │ 14.68    │                      142.88 │
│ local  │ none  │ write       │ 13.47    │                      155.70 │
│ local  │ zstd  │ write       │ 15.54    │                      134.96 │
└────────┴───────┴─────────────┴──────────┴─────────────────────────────┘
```

## Profiles

```
nsys profile -t nvtx,cuda --python-sampling=true --force-overwrite=true --output=write-compressed-kvikio ./bench.py write --compress --kvikio --profiling
nsys profile -t nvtx,cuda --python-sampling=true --force-overwrite=true --output=read-compressed-kvikio ./bench.py read --compress --kvikio --profiling
```

```python
>>> import cupy as cp
>>> from nvidia import nvcomp
>>> codec = nvcomp.Codec(algorithm="Zstd")
>>> data = cp.random.randint(0, 256, size=(4096, 4096))
```