import pathlib

import xarray as xr
import zarr


def main():
    p = pathlib.Path("test.zarr")
    if not p.exists():
        xr.tutorial.open_dataset("air_temperature").to_zarr("test.zarr", zarr_format=3)

    with zarr.config.enable_gpu():
        ds = xr.open_dataset("test.zarr", engine="zarr")
        print(ds)
        print(type(ds.air.data))


if __name__ == "__main__":
    main()
