import pytest
import sys
sys.path.insert(0, "/home/jzhu4/projects/work/harmony-curr/harmony-service-lib-py")
import harmony
from transform import HarmonyAdapter
from harmony.message import Message
from argparse import ArgumentParser


#@pytest.fixture
#def adapter():
#    def _method(message):
#        return HarmonyAdapter(Message(message))
#    return _method

#define a class adapter

class test_adapter(HarmonyAdapter):
    def __init__(self, messagestr):
        self.adapter = HarmonyAdapter(Message(messagestr))
        self.downloaded_file=None
        self.downloaded_succes=False
        self.subsetted_file=None
        self.subsetted_success=False
   
def get_file_info(infile):
    from osgeo import gdal
    import osr
    ds = gdal.Open(infile)
    proj_wkt=ds.GetProjection()
    proj = osr.SpatialReference(wkt=proj_wkt)
    gcs = proj.GetAttrValue('GEOGCS',0)
    authority = proj.GetAttrValue('AUTHORITY',0)
    epsg = proj.GetAttrValue('AUTHORITY',1)
    width = ds.RasterXSize
    height = ds.RasterYSize
    xy_size = [width, height]
    bands = ds.RasterCount
    meta = ds.GetMetadata()
    gt = ds.GetGeoTransform()
    minx = round(gt[0],2)
    miny = round(gt[3] + width*gt[4] + height*gt[5],2)
    maxx = round(gt[0] + width*gt[1] + height*gt[2],2)
    maxy = round(gt[3],2)
    extent = [miny, maxy, minx, maxx]   
    ds = None
    information={'proj_wkt':proj_wkt,
                 'gcs':gcs,
                 'authority':authority,
                 'epsg':epsg,
                 'width':width,
                 'height':height,
                 'xy_size':[width,height],
                 'bands':bands,
                 'meta':meta,
                 'gt':gt,
                 'minx':minx,
                 'miny':miny,
                 'maxx':maxx,
                 'maxy':maxy,
                 'extent':extent
                 }
    return information
