import get_uavsar_info
import harmony_requests
import pytest
from pytest import mark
import os.path

@pytest.fixture()
def info():
    global expected
    global product
    expected =  {'cs': 'Geographic',
                 'proj_cs':'NA',
                 'gcs': 'WGS 84',
                 'authority': 'EPSG',
                 'proj_epsg': 'NA',
                 'gcs_epsg': '4326',
                 'subset': [28.9, 29.0, -89.1, -89.0],
                 'bands': 3,
                 'variables': ['Band1', 'Band2', 'Band3', 'NA']
                 }

    product_info = get_uavsar_info.get_uavsar_info(outfile)
    product = {'GDAL_CS': product_info.get('gdal_cs'),
               'GDAL_PROJ_CS': product_info.get('gdal_proj_cs'),
               'GDAL_GCS': product_info.get('gdal_gcs'),
               'GDAL_AUTHORITY': product_info.get('gdal_authority'),
               'GDAL_PROJ_EPSG': product_info.get('gdal_proj_epsg'),
               'GDAL_GCS_EPSG': product_info.get('gdal_gcs_epsg'),
               'GDAL_SUBSET': product_info.get('gdal_spatial_extent'),
               'GDAL_BANDS': product_info.get('gdal_n_bands'),
               'GDAL_VARIABLES': product_info.get('gdal_variables')
               }

@mark.uavsar    
@mark.status
def test_uavsar_status(harmony_url_config):
    base = harmony_url_config.base_url
    uavsar_id = harmony_url_config.uavsar_id
    path_flag = 'uavsar'

    if harmony_url_config.env_flag == 'prod':
        granule_id = 'G1366852113-ASF'
    else:
        granule_id = 'G1233284377-ASF'

    harmony_url = base + uavsar_id + '/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?subset=lat(28.9:29.0)&subset=lon(-89.1:-89)&format=image%2Ftiff&granuleID=' + granule_id
    global outfile
    outfile = harmony_url_config.env_flag + '_uavsar_query6.tiff'
    get_data_and_status = harmony_requests.harmony_requests(harmony_url, path_flag, outfile)
    assert get_data_and_status == 200

@mark.uavsar
@mark.existing_data
def test_uavsar_existing_data(harmony_url_config):
    path = './uavsar/uavsar_products/'
    global outfile
    outfile = harmony_url_config.env_flag + '_uavsar_query6.tiff'
    assert os.path.exists(path+outfile) == True

@mark.uavsar
@mark.cs
def test_uavsar_cs(info):
    assert product['GDAL_CS'] == expected['cs']

@mark.uavsar
@mark.projection
def test_uavsar_projection(info):
    assert product['GDAL_PROJ_CS'] == expected['proj_cs']

@mark.uavsar
@mark.proj_epsg
def test_uavsar_proj_epsg(info):
    assert product['GDAL_PROJ_EPSG'] == expected['proj_epsg']

@mark.uavsar
@mark.gcs
def test_uavsar_gcs(info):
    assert product['GDAL_GCS'] == expected['gcs']

@mark.uavsar
@mark.gcs_epsg
def test_uavsar_gcs_epsg(info):
    assert product['GDAL_GCS_EPSG'] == expected['gcs_epsg']

@mark.uavsar
@mark.authority
def test_uavsar_authority(info):
    assert product['GDAL_AUTHORITY'] == expected['authority']

@mark.uavsar
@mark.subset
def test_uavsar_subset(info):
    assert product['GDAL_SUBSET'] == expected['subset']

@mark.uavsar
@mark.bands
def test_uavsar_bands(info):
    assert product['GDAL_BANDS'] == expected['bands']

#@mark.uavsar
#@mark.variables
#def test_uavsar_variables(info):
#    assert product['GDAL_VARIABLES'] == expected['variables']
