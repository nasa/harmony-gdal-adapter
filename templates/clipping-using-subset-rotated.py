import os
#import fiona
#import rasterio
#import rasterio.mask
#from shapely.geometry import box
from osgeo import gdal, osr
import math
from affine import Affine

from subset_rotated import subset2

if __name__=="__main__":

    filename="/home/jzhu4/projects/work/harmony-curr/sampledata/avnir/IMG-01-ALAV2A279143000-OORIRFU_000.tif"
    bbox=[-96.898, 29.738, -96.636, 29.876]
    output_dir='.'
    subset2(filename,output_dir,bbox)
    print("complete...")

