"""Utilities for general use through unit tests."""

import json
import sys
from os.path import abspath, dirname

from harmony_service_lib.message import Message
from harmony_service_lib.logging import setup_stdout_log_formatting
from harmony_service_lib.util import config, create_decrypter
from osgeo import gdal
from osgeo.osr import SpatialReference

sys.path.append(dirname(abspath(__file__)) + "/../gdal_subsetter/")
from transform import HarmonyAdapter


class UnittestAdapter(HarmonyAdapter):
    def __init__(self, message_string):
        cfg = config()
        setup_stdout_log_formatting(cfg)
        secret_key = cfg.shared_secret_key
        decrypter = create_decrypter(bytes(secret_key, "utf-8"))
        message_data = json.loads(message_string)
        self.adapter = HarmonyAdapter(Message(message_data, decrypter))
        self.adapter.set_config(cfg)

        self.downloaded_file = None
        self.downloaded_success = False
        self.subsetted_file = None
        self.subsetted_success = False
        self.var_basename = None


def get_file_info(infile):
    ds = gdal.Open(infile)
    proj_wkt = ds.GetProjection()
    proj = SpatialReference(wkt=proj_wkt)
    gcs = proj.GetAttrValue("GEOGCS", 0)
    authority = proj.GetAttrValue("AUTHORITY", 0)
    epsg = proj.GetAttrValue("AUTHORITY", 1)
    width = ds.RasterXSize
    height = ds.RasterYSize
    bands = ds.RasterCount
    meta = ds.GetMetadata()
    gt = ds.GetGeoTransform()
    minx = round(gt[0], 2)
    miny = round(gt[3] + width * gt[4] + height * gt[5], 2)
    maxx = round(gt[0] + width * gt[1] + height * gt[2], 2)
    maxy = round(gt[3], 2)
    extent = [miny, maxy, minx, maxx]
    ds = None

    return {
        "proj_wkt": proj_wkt,
        "gcs": gcs,
        "authority": authority,
        "epsg": epsg,
        "width": width,
        "height": height,
        "xy_size": [width, height],
        "bands": bands,
        "meta": meta,
        "gt": gt,
        "minx": minx,
        "miny": miny,
        "maxx": maxx,
        "maxy": maxy,
        "extent": extent,
    }
