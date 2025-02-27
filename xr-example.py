import xarray as xr
import zarr

xr.tutorial.open_dataset("air_temperature").to_zarr(
    "test.zarr", zarr_format=3, mode="w"
)
with zarr.config.enable_gpu():
    ds = xr.open_dataset("test.zarr", engine="zarr", decode_times=False)
    print(ds)
    print(type(ds.air.data))
    ds.air.data.mean()
