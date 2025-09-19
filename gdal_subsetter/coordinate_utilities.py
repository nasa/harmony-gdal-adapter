"""Utilities for handling coordinate transformations and similar functionality
within the Harmony GDAL Adapter.

"""

from math import isinf
from typing import List, Tuple

from affine import Affine
from osgeo import osr
from pyproj import Proj

from gdal_subsetter.utilities import OpenGDAL


def boxwrs84_boxproj(boxwrs84, reference_file: str):
    """Convert the box defined in lon/lat to box in projection coordinates
    defined in the reference file.

    inputs:
        boxwrs84, which is defined as [left,low,right,upper] in lon/lat
        reference_file: Path to a reference dataset

    returns:
        boxprj, which is also defined as:

        {"llxy": llxy, "lrxy": lrxy, "urxy": urxy, "ulxy": ulxy},

        where llxy,lrxy, urxy, and ulxy are coordinate pairs in projection
        projection, which is the projection of ref_ds

    """
    with OpenGDAL(reference_file) as ref_ds:
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

    boxproj = {"llxy": llxy, "lrxy": lrxy, "urxy": urxy, "ulxy": ulxy}

    return boxproj, projection


def calc_coord_ij(
    geotransform: Tuple[float], x_coordinate: float, y_coordinate: float
) -> Tuple[int]:
    """Calculate array (i, j) coordinates from spatial (x, y) coordinates."""
    transform = Affine.from_gdal(*geotransform)
    rev_transform = ~transform
    cols, rows = rev_transform * (x_coordinate, y_coordinate)

    return int(cols), int(rows)


def calc_ij_coord(
    geotransform: Tuple[float], column_index: int, row_index: int
) -> Tuple[float]:
    """Calculate spatial (x, y) coordinates of a pixel in the GeoTIFF raster
    given the row and column indices).

    """
    return Affine.from_gdal(*geotransform) * (column_index, row_index)


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
    """Determine if the geotransform associated with the given GeoTIFF file is
    rotated or not by considering the row and column translation elements
    of that geotransform.

    """
    with OpenGDAL(srcfile) as dataset:
        geo_transform = dataset.GetGeoTransform()

    return geo_transform[2] != 0.0 or geo_transform[4] != 0


def lonlat_to_projcoord(srcfile, lon, lat):
    """Convert longitude and latitude coordinates to the projection specified
    in the input GeoTIFF.

    """
    with OpenGDAL(srcfile) as dataset:
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

    dst = osr.SpatialReference(projection)
    dstproj4 = dst.ExportToProj4()
    ct2 = Proj(dstproj4)
    xy = ct2(lon, lat)

    if isinf(xy[0]) or isinf(xy[1]):  # pylint: disable=unsubscriptable-object
        xy = [None, None]

    return [xy[0], xy[1]], geotransform


def calc_subset_envelope_window(reference_file: str, box, delt=0):
    """
    inputs:
        reference_file: the reference dataset
        box: Defined as:

            {'llxy':llxy, 'lrxy':lrxy, 'urxy':urxy, 'ulxy':ulxy},

            where llxy,lrxy, urxy, and ulxy are coordinate pairs in projection
        delt: the number of deltax and deltay to extend the subsetting
              array which represents the box
    returns: ul_x, ul_y, ul_i, ul_j, cols, rows

    """
    with OpenGDAL(reference_file) as reference_dataset:
        geotransform = reference_dataset.GetGeoTransform()
        cols_img = reference_dataset.RasterXSize
        rows_img = reference_dataset.RasterYSize

    # get (i, j) coordinates in the array of 4 corners of the box
    ul_i, ul_j = calc_coord_ij(geotransform, box["ulxy"][0], box["ulxy"][1])
    ur_i, ur_j = calc_coord_ij(geotransform, box["urxy"][0], box["urxy"][1])
    ll_i, ll_j = calc_coord_ij(geotransform, box["llxy"][0], box["llxy"][1])
    lr_i, lr_j = calc_coord_ij(geotransform, box["lrxy"][0], box["lrxy"][1])

    # adjust box in array coordinates
    ul_i -= delt
    ul_j -= delt
    ur_i += delt
    ur_j -= delt
    lr_i += delt
    lr_j += delt
    ll_i -= delt
    ll_j += delt

    # get the envelop of the box in array coordinates
    ul_i = min(ul_i, ur_i, ll_i, lr_i)
    ul_j = min(ul_j, ur_j, ll_j, lr_j)
    lr_i = max(ul_i, ur_i, ll_i, lr_i)
    lr_j = max(ul_j, ur_j, ll_j, lr_j)

    # get the intersection between box and image in row, col coordinator
    ul_i = max(0, ul_i)
    ul_j = max(0, ul_j)
    lr_i = min(cols_img, lr_i)
    lr_j = min(rows_img, lr_j)
    cols = lr_i - ul_i
    rows = lr_j - ul_j
    ul_x, ul_y = calc_ij_coord(geotransform, ul_i, ul_j)

    return ul_x, ul_y, ul_i, ul_j, cols, rows
