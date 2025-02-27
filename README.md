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


On curiousity (NVIDIA A100-SXM4-80GB). Unclear if GDS is actually enabled.

```
$ docker run --gpus all --rm -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 toaugspurger/kvikio-zarr-v3 /bin/bash
$ . .venv/bin/activate
$ KVIKIO_NTHREADS=32 ./bench.py all
┏━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Task   ┃ Store ┃ Compression ┃ Duration ┃ Effective Throughput (MB/s) ┃
┡━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ kvikio │ none  │ read        │ 0.11     │                     9480.99 │
│ kvikio │ zstd  │ read        │ 0.31     │                     3344.54 │
│ local  │ none  │ read        │ 0.17     │                     6211.48 │
│ local  │ zstd  │ read        │ 0.44     │                     2371.81 │
│ kvikio │ none  │ write       │ 0.24     │                     4296.39 │
│ kvikio │ zstd  │ write       │ 1.13     │                      925.32 │
│ local  │ none  │ write       │ 0.40     │                     2618.83 │
│ local  │ zstd  │ write       │ 0.92     │                     1136.39 │
└────────┴───────┴─────────────┴──────────┴─────────────────────────────┘
```

## Profiles

```
nsys profile -t nvtx,cuda --python-sampling=true --force-overwrite=true --output=write-compressed-kvikio ./bench.py write --compress --kvikio --profiling
nsys profile -t nvtx,cuda --python-sampling=true --force-overwrite=true --output=read-compressed-kvikio ./bench.py read --compress --kvikio --profiling
```
