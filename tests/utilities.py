"""Utilities for use in unit tests. Likely to be migrated to pytest fixtures."""

from os.path import join as path_join

import numpy as np
import rasterio as rio
from netCDF4 import Dataset
from pyproj import CRS


def get_dummy_netcdf_file(parent_directory: str) -> str:
    """Create a simple netCDF4 file and return the path to it."""
    netcdf_file_path = path_join(parent_directory, "dummy.nc")

    with Dataset(netcdf_file_path, "w", format="NETCDF4") as netcdf_ds:
        netcdf_ds.setncatts(
            {
                "title": "Dummy File",
                "comment": "Produced for HGA unit tests only",
                "Conventions": "CF-1.7",
            }
        )
        netcdf_ds.createDimension("lon", 3)
        netcdf_ds.createDimension("lat", 4)

        x_variable = netcdf_ds.createVariable("lon", float, ("lon",))
        x_variable.setncatts(
            {
                "standard_name": "longitude",
                "units": "degrees_east",
            }
        )
        x_variable[:] = np.array([10, 20, 30])

        y_variable = netcdf_ds.createVariable("lat", float, ("lat",))
        y_variable.setncatts(
            {
                "standard_name": "latitude",
                "units": "degrees_north",
            }
        )
        y_variable[:] = np.array([5, 15, 25, 35])

        grid_mapping = netcdf_ds.createVariable("crs", "S1")
        grid_mapping.setncatts(CRS.from_epsg(4326).to_cf())

        science_variable = netcdf_ds.createVariable("science", float, ("lat", "lon"))
        science_variable.setncatts({"grid_mapping": "crs"})

    return netcdf_file_path


def get_dummy_geotiff_file(parent_directory: str) -> str:
    """Create a simple GeoTIFF file with 2 bands and return the path to it."""
    geotiff_file_path = path_join(parent_directory, "dummy.tif")

    height = 100
    width = 100
    data = np.random.rand(height, width).astype(rio.float32)

    # Define the metadata for the GeoTIFF
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": data.dtype,
        "crs": "EPSG:4326",
        "transform": rio.transform.from_origin(
            -100.0,
            50.0,
            0.1,
            0.1,
        ),
        "nodata": -9999,
    }

    # Write the GeoTIFF file
    with rio.open(geotiff_file_path, "w", **profile) as geotiff_ds:
        # Write a band with no standard_name
        geotiff_ds.write(data, 1)

        # Write a band with a standard_name
        geotiff_ds.write(data, 2)
        geotiff_ds.update_tags(2, standard_name="sea_surface_temperature")

    return geotiff_file_path
