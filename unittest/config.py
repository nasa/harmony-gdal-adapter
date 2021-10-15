import pytest
import sys
import os
import json

sys.path.append(os.path.dirname(
    os.path.abspath(__file__)) + '/../gdal_subsetter/'
)
from transform import HarmonyAdapter

import logging
import harmony
from harmony import BaseHarmonyAdapter
from harmony.message import Message
from argparse import ArgumentParser
from harmony.logging import setup_stdout_log_formatting
from harmony.util import (CanceledException, HarmonyException, receive_messages, delete_message,
                          change_message_visibility, config, create_decrypter)


class UnittestAdapter(HarmonyAdapter):
    def __init__(self, message_string):
        cfg = config()
        setup_stdout_log_formatting(cfg)
        secret_key = cfg.shared_secret_key
        decrypter = create_decrypter(bytes(secret_key, 'utf-8'))
        message_data = json.loads(message_string)
        self.adapter = HarmonyAdapter(Message(message_data, decrypter))
        self.adapter.set_config(cfg)

        self.downloaded_file = None
        self.downloaded_succes = False
        self.subsetted_file = None
        self.subsetted_success = False
        self.var_basename = None

    def _create_adapter(self, message_string):
        cfg = config()
        setup_stdout_log_formatting(cfg)
        secret_key = cfg.shared_secret_key
        decrypter = create_decrypter(bytes(secret_key, 'utf-8'))
        message_data = json.loads(message_string)
        adapter = BaseHarmonyAdapter(Message(message_data, decrypter))
        adapter.set_config(cfg)

        return adapter


def get_file_info(infile):
    from osgeo import gdal
    import osr
    ds = gdal.Open(infile)
    proj_wkt = ds.GetProjection()
    proj = osr.SpatialReference(wkt=proj_wkt)
    gcs = proj.GetAttrValue('GEOGCS', 0)
    authority = proj.GetAttrValue('AUTHORITY', 0)
    epsg = proj.GetAttrValue('AUTHORITY', 1)
    width = ds.RasterXSize
    height = ds.RasterYSize
    xy_size = [width, height]
    bands = ds.RasterCount
    meta = ds.GetMetadata()
    gt = ds.GetGeoTransform()
    minx = round(gt[0], 2)
    miny = round(gt[3] + width*gt[4] + height*gt[5], 2)
    maxx = round(gt[0] + width*gt[1] + height*gt[2], 2)
    maxy = round(gt[3], 2)
    extent = [miny, maxy, minx, maxx]
    ds = None

    return {
        'proj_wkt': proj_wkt,
        'gcs': gcs,
        'authority': authority,
        'epsg': epsg,
        'width': width,
        'height': height,
        'xy_size': [width, height],
        'bands': bands,
        'meta': meta,
        'gt': gt,
        'minx': minx,
        'miny': miny,
        'maxx': maxx,
        'maxy': maxy,
        'extent': extent
    }
