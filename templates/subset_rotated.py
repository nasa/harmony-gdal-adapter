import os
#import fiona
#import rasterio
#import rasterio.mask
#from shapely.geometry import box
from osgeo import gdal, osr
import math
from affine import Affine


def subset2(tiffile, output_dir, bbox, shapefile=None):

    RasterFormat = 'GTiff'

    raw_file_name = os.path.splitext(os.path.basename(tiffile))[0]
  
    output_file=output_dir+'/subsetted.tif'


    [ll_lon, ll_lat,ur_lon, ur_lat]=bbox

    #covert bbox to tiffile's coordinator
    
    src = osr.SpatialReference()

    src.SetWellKnownGeogCS("WGS84")

    dataset = gdal.Open(tiffile)

    transform=dataset.GetGeoTransform()

    xRes=transform[1]

    yRes=transform[5]

    projection = dataset.GetProjection()

    dst = osr.SpatialReference(projection)

    ct = osr.CoordinateTransformation(src, dst)

    llxy = ct.TransformPoint(ll_lon, ll_lat)

    urxy = ct.TransformPoint(ur_lon, ur_lat)

    shapes=[ llxy[0],llxy[1], urxy[0],urxy[1] ]

    gdal.Warp(output_file, dataset, format=RasterFormat, outputBounds=shapes, xRes=xRes, yRes=yRes, dstSRS=projection, resampleAlg=gdal.GRA_NearestNeighbour, options=['COMPRESS=DEFLATE'])

    dataset=None

    return output_file    
