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
$ KVIKIO_NTHREADS=64 ./bench.py all
Working... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:12
┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Task  ┃ Compression ┃ Store  ┃ Duration ┃ Effective Throughput (MB/s) ┃         ┃
┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ read  │ none        │ kvikio │ read     │ 0.15                        │ 6970.26 │
│ read  │ none        │ local  │ read     │ 0.16                        │ 6618.90 │
│ read  │ zstd        │ kvikio │ read     │ 0.38                        │ 2779.28 │
│ read  │ zstd        │ local  │ read     │ 0.38                        │ 2765.37 │
│ write │ none        │ kvikio │ write    │ 3.87                        │  270.84 │
│ write │ none        │ local  │ write    │ 5.36                        │  195.74 │
│ write │ zstd        │ kvikio │ write    │ 7.09                        │  147.97 │
│ write │ zstd        │ local  │ write    │ 4.52                        │  232.02 │
└───────┴─────────────┴────────┴──────────┴─────────────────────────────┴─────────┘
```


On curiousity (NVIDIA A100-SXM4-80GB). Unclear if GDS is actually enabled.

```
$ docker run --gpus all --rm -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 toaugspurger/kvikio-zarr-v3 /bin/bash
$ . .venv/bin/activate
$ KVIKIO_NTHREADS=32 ./bench.py all
Working... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:12
┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Task  ┃ Compression ┃ Store  ┃ Duration ┃ Effective Throughput (MB/s) ┃          ┃
┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ read  │ none        │ kvikio │ read     │ 0.09                        │ 11185.60 │
│ read  │ none        │ local  │ read     │ 0.15                        │  6773.34 │
│ read  │ zstd        │ kvikio │ read     │ 0.33                        │  3208.50 │
│ read  │ zstd        │ local  │ read     │ 0.46                        │  2266.46 │
│ write │ none        │ kvikio │ write    │ 0.23                        │  4559.98 │
│ write │ none        │ local  │ write    │ 0.37                        │  2816.11 │
│ write │ zstd        │ kvikio │ write    │ 0.79                        │  1328.97 │
│ write │ zstd        │ local  │ write    │ 0.94                        │  1113.98 │
└───────┴─────────────┴────────┴──────────┴─────────────────────────────┴──────────┘ 
```

## Profiles

```
nsys profile -t nvtx,cuda --python-sampling=true --force-overwrite=true --output=write-compressed-kvikio ./bench.py write --compress --kvikio --profiling
nsys profile -t nvtx,cuda --python-sampling=true --force-overwrite=true --output=read-compressed-kvikio ./bench.py read --compress --kvikio --profiling
```
