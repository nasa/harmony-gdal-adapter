""" A module to contain utility functions pertaining to GeoJSON shape files
    within the Harmony GDAL Adapter.

"""
from os import remove as remove_file
from os.path import isdir, isfile
from shutil import rmtree
import json

from geopandas import GeoDataFrame
from osgeo import ogr, osr
from pyproj import CRS
from shapely.geometry import shape, mapping
from shapely.ops import cascaded_union
import fiona


def create_shapefile_with_box(box, projection, shapefile):
    """
        input: box {ll, lr, ur, ul} in projection coordinates, where:

        ll = (ll_lon, ll_lat)
        lr = (lr_lon, lr_lat)
        ur = (ur_lon, ur_lat)
        ul = (ul_lon, ul_lat)

    """

    # output: polygon geometry
    llxy = box.get('llxy')
    lrxy = box.get('lrxy')
    urxy = box.get('urxy')
    ulxy = box.get('ulxy')
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(llxy[0], llxy[1])
    ring.AddPoint(lrxy[0], lrxy[1])
    ring.AddPoint(urxy[0], urxy[1])
    ring.AddPoint(ulxy[0], ulxy[1])
    ring.AddPoint(llxy[0], llxy[1])
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    # create output file
    out_driver = ogr.GetDriverByName('ESRI Shapefile')

    if isfile(shapefile):
        remove_file(shapefile)
    elif isdir(shapefile):
        rmtree(shapefile)

    out_data_source = out_driver.CreateDataSource(shapefile)
    out_spatial_ref = osr.SpatialReference(projection)
    out_layer = out_data_source.CreateLayer('boundingbox', out_spatial_ref,
                                            geom_type=ogr.wkbPolygon)
    feature_definition = out_layer.GetLayerDefn()

    # add new geom to layer
    out_feature = ogr.Feature(feature_definition)
    out_feature.SetGeometry(polygon)
    out_layer.CreateFeature(out_feature)
    out_feature.Destroy()
    out_data_source.Destroy()


def convert_to_multipolygon(infile: str, outfile: str, buf=None):
    """ Convert point or line feature GeoJSON file to multi-polygon feature
        GeoJSON file

        input:
            infile - point or line feature GeoJSON file name
            buf - buffer defined in degree or meter for geographic or
                  projected coordinates for line or point features GeoJSON
                  file.
        return:
            outfile - multi-polygon feature ESRI shapefile directory name
    """
    if not buf:
        return infile

    fd_infile = fiona.open(infile)
    # get feature type of infile
    featype = fd_infile.schema.get('geometry')
    # prepare meta for polygon file
    meta = fd_infile.meta
    meta['schema']['geometry'] = 'Polygon'
    meta['schema']['properties'] = {'id': 'int'}
    meta['driver'] = 'GeoJSON'
    with fiona.open(outfile, 'w', **meta) as fd_outfile:
        poly_lst = []
        for index_point, point in enumerate(fd_infile):
            pt = shape(point['geometry'])
            polygon = pt.buffer(buf)
            poly_lst.append(polygon)

        polygons = cascaded_union(poly_lst)
        if polygons.geometryType() == 'Polygon':
            fd_outfile.write({'geometry': mapping(polygons),
                              'properties': {'id': 0}})
        else:
            for index_polygon, polygon in enumerate(polygons):
                fd_outfile.write({'geometry': mapping(polygon),
                                  'properties': {'id': index_polygon}})

    return outfile


def get_coordinates_unit(geojson_file: str):
    """ Parse contents of the request shape file and retrieve its units. """
    try:
        # get unit of the feature in the shapefile
        fd_infile = fiona.open(geojson_file)
        geometry = fd_infile.schema.get('geometry')
        proj = CRS(fd_infile.crs_wkt)
        proj_json = json.loads(proj.to_json())
        unit = proj_json['coordinate_system']['axis'][0]['unit']
    except Exception:
        unit = None

    return geometry, unit


def shapefile_boxproj(shapefile, ref_ds, outputfile):
    """ Convert shape file and calculate the envelop box in the projection
        defined in ref_ds

        inputs:
            shapefile - used to define the AOI
            ref_ds - dataset associate with the reference geotiff file
            outputfile - output shapefile anme
        returns:
            boxproj - extent of the outputfile
            ref_proj - projection of the ref_ds
            outputfile - output shapefile name
            geometryname - geometry name of the features in the outputfile

    """
    ref_proj = ref_ds.GetProjection()
    tmp = GeoDataFrame.from_file(shapefile)
    tmpproj = tmp.to_crs(ref_proj)
    tmpproj.to_file(outputfile)
    shp = ogr.Open(outputfile)
    lyr = shp.GetLayer()
    lyrextent = lyr.GetExtent()
    feature = lyr.GetNextFeature()
    geometry = feature.GetGeometryRef()
    geometryname = geometry.GetGeometryName()
    # Extent[lon_min, lon_max, lat_min, lat_max]
    # boxproj={'llxy': llxy, 'lrxy': lrxy, 'urxy': urxy, 'ulxy': ulxy}
    # where llxy, lrxy, urxy, and ulxy are coordinate pairs in projection
    llxy = (lyrextent[0], lyrextent[2])
    lrxy = (lyrextent[1], lyrextent[2])
    urxy = (lyrextent[1], lyrextent[3])
    ulxy = (lyrextent[0], lyrextent[3])
    boxproj = {'llxy': llxy, 'lrxy': lrxy, 'urxy': urxy, 'ulxy': ulxy}
    return boxproj, ref_proj, outputfile, geometryname
