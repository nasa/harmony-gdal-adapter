import json
import os
import sys
import pystac
from datetime import datetime

sys.path.append(os.path.dirname(
    os.path.abspath(__file__)) + '/../gdal_subsetter/'
                )
from transform import HarmonyAdapter

import harmony
from harmony.message import Message
from harmony.logging import setup_stdout_log_formatting
from harmony.util import (CanceledException, config, create_decrypter)


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


class UnittestAdapterNoDownload(HarmonyAdapter):
    def __init__(self, message_string):
        cfg = UnittestAdapterNoDownload.config_fixture()
        setup_stdout_log_formatting(cfg)
        message_data = json.loads(message_string)
        message = Message(message_data)

        catalog = pystac.Catalog('0', 'Catalog 0')
        item = pystac.Item('1', {}, [0, 0, 1, 1], datetime(2020, 1, 1, 0, 0, 0), {})
        catalog.add_item(item)
        granule = message.granules[0]
        if granule.name == 'ALAV2A104483330':
            granule.local_filename = 'file://./data/granules/anvir2/ALAV2A104483330-OORIRFU.zip'
            item.add_asset('data', pystac.Asset(granule.local_filename, roles=['data']))
        elif granule.name == 'S1-GUNW-D-R-083-tops-20141116_20141023-095646-360325S_38126S-PP-24b3-v2_0_2':
            granule.local_filename = 'file://./data/granules/gfrn/S1-GUNW-D-R-083-tops-20141116_20141023-095646-360325S_38126S-PP-24b3-v2_0_2.nc'
            item.add_asset('data', pystac.Asset(granule.local_filename, roles=['data']))
        elif granule.name == 'UA_gulfco_32010_09045_001_090617_L090_CX_01-PAULI':
            granule.local_filename = 'file://./data/granules/uavsar/gulfco_32010_09045_001_090617_L090_CX_01_pauli.tif'
            item.add_asset('data', pystac.Asset(granule.local_filename, roles=['data']))

        self.adapter = HarmonyAdapter(message, catalog=catalog, config=cfg)
        self.adapter.get_version = lambda: "unittest" \
                                           ""
        self.downloaded_file = None
        self.downloaded_succes = False
        self.subsetted_file = None
        self.subsetted_success = False
        self.var_basename = None

    @staticmethod
    def config_fixture(fallback_authn_enabled=False,
                       edl_username='yoda',
                       edl_password='password_this_is',
                       use_localstack=False,
                       staging_bucket='UNKNOWN',
                       staging_path='UNKNOWN',
                       oauth_client_id=None,
                       user_agent=None,
                       app_name=None):
        c = harmony.util.config(validate=False)
        return harmony.util.Config(
            # Override
            fallback_authn_enabled=fallback_authn_enabled,
            edl_username=edl_username,
            edl_password=edl_password,
            use_localstack=use_localstack,
            staging_path=staging_path,
            staging_bucket=staging_bucket,
            oauth_client_id=oauth_client_id,
            app_name=app_name,
            # Default
            env=c.env,
            oauth_host=c.oauth_host,
            oauth_uid=c.oauth_uid,
            oauth_password=c.oauth_password,
            oauth_redirect_uri=c.oauth_redirect_uri,
            backend_host=c.backend_host,
            localstack_host=c.localstack_host,
            aws_default_region=c.aws_default_region,
            text_logger=c.text_logger,
            health_check_path=c.health_check_path,
            shared_secret_key=c.shared_secret_key,
            # Override if provided, else default
            user_agent=c.user_agent if user_agent is None else user_agent
        )


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
    miny = round(gt[3] + width * gt[4] + height * gt[5], 2)
    maxx = round(gt[0] + width * gt[1] + height * gt[2], 2)
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
