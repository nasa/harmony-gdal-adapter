"""Utilities to convert a GeoTIFF file into a netCDF4 output.

This module may be better as a separate microservice within Harmony.

"""

import re
from datetime import datetime

import numpy as np
from netCDF4 import Dataset
from numpy.ma import masked_array
from osgeo import gdal
from pycrs.parse import from_ogc_wkt as parse_crs_from_ogc_wkt
from pyproj import Proj

from gdal_subsetter.utilities import OpenGDAL


def convert_geotiff_to_netcdf(infile: str, outfile: str) -> str:
    """Convert GeoTIFF file to NetCDF-4 file.

    input:
        infile - geotiff file name
    return:
        outfile - netcdf file name

    """

    with (
        OpenGDAL(infile) as input_geotiff,
        Dataset(outfile, mode="w", format="NETCDF4") as output_netcdf4,
    ):
        # define global attributes
        output_netcdf4.title = ""
        output_netcdf4.institution = "Alaska Satellite Facility"
        output_netcdf4.source = ""
        output_netcdf4.references = ""
        output_netcdf4.comment = ""
        output_netcdf4.history = f"{datetime.utcnow():%d/%m/%Y %H:%M:%S} (UTC)"
        output_netcdf4.GDAL = f"Version {gdal.__version__}"

        # create dimensions
        crs = parse_crs_from_ogc_wkt(input_geotiff.GetProjectionRef())
        if crs.cs_type == "Projected":
            process_projected(input_geotiff, output_netcdf4)
        else:
            process_geogcs(input_geotiff, output_netcdf4)

        output_netcdf4.Conventions = "CF-1.7"

    return outfile


def process_projected(ds_in: gdal.Dataset, dst: Dataset):
    """Create a netCDF4 file from a GeoTIFF with non-geographically gridded data."""
    gt = ds_in.GetGeoTransform()
    crs = parse_crs_from_ogc_wkt(ds_in.GetProjectionRef())
    unitname = crs.unit.unitname.proj4
    # define dimensions
    dst.createDimension("x", ds_in.RasterXSize)
    dst.createDimension("y", ds_in.RasterYSize)
    # copy attributes, geotiff metadata is party of attributes in netcdf.
    # Conventions, GDAL, history
    for attribute_name, attribute_value in ds_in.GetMetadata().items():
        if attribute_name.find(r"#") > -1:
            tmp = attribute_name.split(r"#")
            if tmp[0] == "NC_GLOBAL":
                dst.setncattr(tmp[1], attribute_value.rstrip("\n"))
        else:
            if attribute_name not in ("_FillValue", "Conventions"):
                dst.setncattr(attribute_name, attribute_value.rstrip("\n"))

    # create georeference variable
    crs_name = crs.proj.name.ogc_wkt.lower()
    geovar = dst.createVariable(crs_name, "S1")
    geovar.grid_mapping_name = crs_name
    geovar.long_name = "CRS definition"
    for item in crs.params:
        attr_name = str(item).split(".")[-1].split(" ")[0]
        attr_lst = re.findall("[A-Z][^A-Z]*", attr_name)
        name = "_".join(attr_lst).lower()
        geovar.setncattr(name, item.value)

    geovar.longitude_of_prime_meridian = crs.geogcs.prime_mer.value
    if crs.geogcs.datum.ellips:
        geovar.semi_major_axis = crs.geogcs.datum.ellips.semimaj_ax.value
        geovar.inverse_flattening = crs.geogcs.datum.ellips.inv_flat.value

    geovar.spatial_ref = ds_in.GetProjectionRef()
    geovar.GeoTransform = " ".join(map(str, list(gt)))

    # create 1D coordinate variables if the geotiff is not rotated image
    if gt[2] == 0.0 and gt[4] == 0.0:
        x_array = gt[0] + gt[1] * (np.arange(ds_in.RasterXSize) + 0.5)
        y_array = gt[3] + gt[5] * (np.arange(ds_in.RasterYSize) + 0.5)
        xvar = dst.createVariable("x", np.dtype("float64"), ("x"))
        xvar[:] = x_array
        xvar.setncattr("standard_name", "projection_x_coordinate")
        xvar.setncattr("axis", "X")
        xvar.setncattr("long_name", "x-coordinate in projected coordinate system")
        xvar.setncattr("units", unitname)
        yvar = dst.createVariable("y", np.dtype("float64"), ("y"))
        yvar[:] = y_array
        yvar.setncattr("standard_name", "projection_y_coordinate")
        xvar.setncattr("axis", "Y")
        yvar.setncattr("long_name", "y-coordinate in projected coordinate system")
        yvar.setncattr("units", unitname)
        lcc = Proj(ds_in.GetProjectionRef())

        # lon 1D
        tmp_y = np.zeros(x_array.shape, x_array.dtype)
        tmp_y[:] = y_array[0]
        lon, tmp_lat = lcc(x_array, tmp_y, inverse=True)  # pylint: disable=unpacking-non-sequence

        # lat 1D
        tmp_x = np.zeros(y_array.shape, y_array.dtype)
        tmp_x[:] = x_array[0]
        tmp_lon, lat = lcc(tmp_x, y_array, inverse=True)  # pylint: disable=unpacking-non-sequence

        lon_var = dst.createVariable("lon", np.float64, ("x"), zlib=True)
        lon_var[:] = lon
        lon_var.units = "degrees_east"
        lon_var.standard_name = "longitude"
        lon_var.long_name = "longitude"

        lat_var = dst.createVariable("lat", np.float64, ("y"), zlib=True)
        lat_var[:] = lat
        lat_var.units = "degrees_north"
        lat_var.standard_name = "latitude"
        lat_var.long_name = "latitude"

    # create data variables
    for band_index in range(1, ds_in.RasterCount + 1):
        band = ds_in.GetRasterBand(band_index)
        meta = band.GetMetadata()
        mask_band = band.GetMaskBand()
        data = band.ReadAsArray()
        mask = mask_band.ReadAsArray()
        mx = masked_array(data, mask=mask == 0)
        # get varname
        varnames = [item for item in meta if item == "standard_name"]
        if varnames:
            varname = meta[varnames[0]].replace("-", "_")
        else:
            varname = f"Band{band_index}".replace("-", "_")

        vardatatype = mx.data.dtype
        fillvalue = band.GetNoDataValue()
        if fillvalue:
            datavar = dst.createVariable(
                varname, vardatatype, ("y", "x"), zlib=True, fill_value=fillvalue
            )
        else:
            datavar = dst.createVariable(varname, vardatatype, ("y", "x"), zlib=True)

        datavar[:, :] = mx

        # write attrs of the variabale datavar
        for attribute_name, attribute_value in band.GetMetadata().items():
            if attribute_name.find(r"#") > -1:
                tmp = attribute_name.split(r"#")
                if tmp[0] == "NC_GLOBAL":
                    dst.setncattr(tmp[1], attribute_value.rstrip("\n"))

            else:
                if attribute_name != "_FillValue":
                    if attribute_name == "units" and attribute_value == "unitless":
                        datavar.setncattr(attribute_name, "1")
                    else:
                        datavar.setncattr(
                            attribute_name,
                            attribute_value.rstrip("\n").replace("-", "_"),
                        )

        datavar.grid_mapping = crs_name

        # add standard_name no standard_name in datavar
        lst = [
            attr for attr in datavar.ncattrs() if attr in ["standard_name", "long_name"]
        ]

        if not lst:
            datavar.standard_name = varname

        # add units attr
        if "units" not in datavar.ncattrs():
            datavar.setncattr("units", "1")


def process_geogcs(ds_in: gdal.Dataset, dst: Dataset):
    """Create a netCDF4 file from a GeoTIFF with geographically gridded data."""
    gt = ds_in.GetGeoTransform()
    crs = parse_crs_from_ogc_wkt(ds_in.GetProjectionRef())
    # define dimensions
    dst.createDimension("lon", ds_in.RasterXSize)
    dst.createDimension("lat", ds_in.RasterYSize)
    # copy attributes, geotiff metadata is party of attributes in netcdf.
    for attribute_name, attribute_value in ds_in.GetMetadata().items():
        if attribute_name.find(r"#") > -1:
            tmp = attribute_name.split(r"#")
            if tmp[0] == "NC_GLOBAL":
                dst.setncattr(tmp[1], attribute_value.rstrip("\n"))

        else:
            if attribute_name not in ("_FillValue", "Conventions"):
                dst.setncattr(attribute_name, attribute_value.rstrip("\n"))

    # create georeference variable
    crs_name = "latitude_longitude"
    geovar = dst.createVariable(crs_name, "S1")
    geovar.grid_mapping_name = crs_name
    geovar.long_name = "CRS definition"
    geovar.longitude_of_prime_meridian = crs.prime_mer.value
    if crs.datum.ellips:
        geovar.semi_major_axis = crs.datum.ellips.semimaj_ax.value
        geovar.inverse_flattening = crs.datum.ellips.inv_flat.value

    geovar.spatial_ref = ds_in.GetProjectionRef()
    geovar.GeoTransform = " ".join(map(str, list(gt)))
    # create coordinate variables if the geotiff is a non-rotated image
    if gt[2] == 0.0 and gt[4] == 0.0:
        lon_array = gt[0] + gt[1] * (np.arange(ds_in.RasterXSize) + 0.5)
        lat_array = gt[3] + gt[5] * (np.arange(ds_in.RasterYSize) + 0.5)
        lonvar = dst.createVariable("lon", np.dtype("float64"), ("lon"))
        lonvar[:] = lon_array
        lonvar.setncattr("standard_name", "longitude")
        lonvar.setncattr("long_name", "longitude")
        lonvar.setncattr("units", "degrees_east")
        latvar = dst.createVariable("lat", np.dtype("float64"), ("lat"))
        latvar[:] = lat_array
        latvar.setncattr("standard_name", "latitude")
        latvar.setncattr("long_name", "latitude")
        latvar.setncattr("units", "degrees_north")
    # else:
    #  create auxilliary coordinates
    # lcc =Proj(ds_in.GetProjectionRef())
    # J, I = np.meshgrid(np.arange(dst.dimensions['lon'].size), np.arange(dst.dimensions['lat'].size) )
    # lon_array = gt[0] + gt[1]*(J + 0.5) + gt[2]*(I + 0.5)
    # lat_array = gt[3] + gt[4]*(J + 0.5) + gt[5]*(I + 0.5)
    # lon, lat = lcc(lon_array, lat_array,inverse=True )
    # lon_var = dst.createVariable('lon', np.float64, ('lat', 'lon'), zlib=True)
    # lon_var[:,:] = lon
    # lon_var.units = 'degrees_east'
    # lon_var.standard_name = 'longitude'
    # lon_var.long_name = 'longitude'
    # lat_var = dst.createVariable('lat', np.float64, ('lat', 'lon'), zlib=True)
    # lat_var[:,:] = lat
    # lat_var.units = 'degrees_north'
    # lat_var.standard_name = 'latitude'
    # lat_var.long_name = 'latitude'

    # create data variables
    for band_index in range(1, ds_in.RasterCount + 1):
        band = ds_in.GetRasterBand(band_index)
        meta = band.GetMetadata()
        mask_band = band.GetMaskBand()
        data = band.ReadAsArray()
        mask = mask_band.ReadAsArray()
        mx = masked_array(data, mask=mask == 0)
        # get varname
        varnames = [item for item in meta if item == "standard_name"]
        if varnames:
            varname = meta[varnames[0]].replace("-", "_")
        else:
            varname = f"Band{band_index}".replace("-", "_")

        vardatatype = mx.data.dtype
        fillvalue = band.GetNoDataValue()
        if fillvalue:
            datavar = dst.createVariable(
                varname, vardatatype, ("lat", "lon"), zlib=True, fill_value=fillvalue
            )
        else:
            datavar = dst.createVariable(
                varname, vardatatype, ("lat", "lon"), zlib=True
            )

        datavar[:, :] = mx
        # write attrs of the variable datavar
        for attribute_name, attribute_value in band.GetMetadata().items():
            if attribute_name.find(r"#") > -1:
                tmp = attribute_name.split(r"#")
                if tmp[0] == "NC_GLOBAL":
                    dst.setncattr(tmp[1], attribute_value.rstrip("\n"))

            else:
                if attribute_name != "_FillValue":
                    if attribute_name == "units" and attribute_value == "unitless":
                        datavar.setncattr(attribute_name, "1")
                    else:
                        datavar.setncattr(
                            attribute_name,
                            attribute_value.rstrip("\n").replace("-", "_"),
                        )

        datavar.grid_mapping = crs_name

        # add standard_name if there is no standard_name or long_name
        # attributes associated with datavar
        lst = [
            attr for attr in datavar.ncattrs() if attr in ["standard_name", "long_name"]
        ]

        if not lst:
            datavar.standard_name = varname

        # add units attribute
        if "units" not in datavar.ncattrs():
            datavar.setncattr("units", "1")

        if gt[2] == 0.0 and gt[4] == 0.0:
            datavar.coordinates = "lon lat"
