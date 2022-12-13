""" Utilities for handling coordinate transformations and similar functionality
    within the Harmony GDAL Adapter.

"""
from math import isinf
from typing import List, Tuple

from affine import Affine
from osgeo import gdal, osr
from pyproj import Proj

from gdal_subsetter.utilities import OpenGDAL


def boxwrs84_boxproj(boxwrs84, ref_ds):
    """ Convert the box defined in lon/lat to box in projection coordinates
        defined in ref_ds

        inputs:
            boxwrs84, which is defined as [left,low,right,upper] in lon/lat
            ref_ds is reference dataset

        returns:
            boxprj, which is also defined as:

            {"llxy": llxy, "lrxy": lrxy, "urxy": urxy, "ulxy": ulxy},

            where llxy,lrxy, urxy, and ulxy are coordinate pairs in projection
            projection, which is the projection of ref_ds
    """
    projection = ref_ds.GetProjection()
    dst = osr.SpatialReference(projection)

    # get coordinates of four corners of the boxwrs84
    ll_lon, ll_lat = boxwrs84[0], boxwrs84[1]
    lr_lon, lr_lat = boxwrs84[2], boxwrs84[1]
    ur_lon, ur_lat = boxwrs84[2], boxwrs84[3]
    ul_lon, ul_lat = boxwrs84[0], boxwrs84[3]

    # convert all four corners
    dstproj4 = dst.ExportToProj4()
    ct = Proj(dstproj4)
    llxy = ct(ll_lon, ll_lat)
    lrxy = ct(lr_lon, lr_lat)
    urxy = ct(ur_lon, ur_lat)
    ulxy = ct(ul_lon, ul_lat)

    boxproj = {'llxy': llxy, 'lrxy': lrxy, 'urxy': urxy, 'ulxy': ulxy}

    return boxproj, projection


def calc_coord_ij(geotransform, x, y) -> Tuple[int]:
    """ Calculate array (i, j) coordinates from spatial (x, y) coordinates. """
    transform = Affine.from_gdal(*geotransform)
    rev_transform = ~transform
    cols, rows = rev_transform*(x, y)

    return int(cols), int(rows)


def calc_ij_coord(geotransform, col, row):
    """ Calculate spatial (x, y) coordinates of a pixel in the GeoTIFF raster
        given the row and column indices).

    """
    transform = Affine.from_gdal(*geotransform)
    x, y = transform * (col, row)

    return x, y


def get_bbox(filename: str) -> List[float]:
    """
    input: the GeoTIFF file
    return: bbox[left,low,right,upper] of the file
    """
    with OpenGDAL(filename) as dataset:
        geotransform = dataset.GetGeoTransform()
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize

    ul_x, ul_y = calc_ij_coord(geotransform, 0, 0)
    ur_x, ur_y = calc_ij_coord(geotransform, cols, 0)
    lr_x, lr_y = calc_ij_coord(geotransform, cols, rows)
    ll_x, ll_y = calc_ij_coord(geotransform, 0, rows)

    return [min(ul_x, ll_x), min(ll_y, lr_y), max(lr_x, ur_x), max(ul_y, ur_y)]


def is_rotated_geotransform(srcfile: str) -> bool:
    """ Determine if the geotransform associated with the given GeoTIFF file is
        rotated or not by considering the row and column translation elements
        of that geotransform.

    """
    with OpenGDAL(srcfile) as dataset:
        geo_transform = dataset.GetGeoTransform()

    check = geo_transform[2] != 0.0 or geo_transform[4] != 0

    return check


def lonlat_to_projcoord(srcfile, lon, lat):
    """ Convert longitude and latitude coordinates to the projection specified
        in the input GeoTIFF.

    """
    with OpenGDAL(srcfile) as dataset:
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

    dst = osr.SpatialReference(projection)
    dstproj4 = dst.ExportToProj4()
    ct2 = Proj(dstproj4)
    xy = ct2(lon, lat)

    if isinf(xy[0]) or isinf(xy[1]):
        xy = [None, None]

    return [xy[0], xy[1]], geotransform
