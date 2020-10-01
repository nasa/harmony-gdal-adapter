import get_avnir_info
import harmony_requests
import pytest
from pytest import mark
import os.path

#Anti-meridian crossing spatial subset

@pytest.fixture()
def info():
    global expected
    global product
    expected = {'cs': 'Projected',
                'proj_cs':'WGS 84 / UTM zone  1N',
                'gcs': 'WGS 84',
                'authority': 'EPSG',
                'proj_epsg': '32601',
                'gcs_epsg': '4326',
                'subset': [66.5, 66.75, -179.85, 179.75],
                'bands': 4,
                'variables': ['Band1', 'Band2', 'Band3', 'Band4']
                }
    product_info = get_avnir_info.get_avnir_info(outfile)
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

@mark.avnir
@mark.status
def test_avnir_status(harmony_url_config):
    base = harmony_url_config.base_url
    avnir_id = harmony_url_config.avnir_id
    path_flag = 'avnir'

    if harmony_url_config.env_flag == 'prod':
        granule_id = 'G1835935580-ASF'
    else:
        granule_id = 'G1236469523-ASF'


    harmony_url = base + avnir_id + '/ogc-api-coverages/1.0.0/collections/Band1%2CBand2/coverage/rangeset?subset=lat(66.5:66.75)&subset=lon(-179.85:179.75)&format=image%2Ftiff&granuleID=' + granule_id
    global outfile
    outfile = harmony_url_config.env_flag + '_avnir_query3.tiff'
    get_data_and_status = harmony_requests.harmony_requests(harmony_url, path_flag, outfile)
    assert get_data_and_status == 200

@mark.avnir
@mark.existing_data
def test_avnir_existing_data(harmony_url_config):
    path = './avnir/avnir_products/'
    global outfile
    outfile = harmony_url_config.env_flag + '_avnir_query3.tiff'
    assert os.path.exists(path+outfile) == True

@mark.avnir
@mark.cs
def test_avnir_cs(info):
    assert product['GDAL_CS'] == expected['cs']

@mark.avnir
@mark.projection
def test_avnir_projection(info):
    assert product['GDAL_PROJ_CS'] == expected['proj_cs']

@mark.avnir
@mark.proj_epsg
def test_avnir_proj_epsg(info):
    assert product['GDAL_PROJ_EPSG'] == expected['proj_epsg']

@mark.avnir
@mark.gcs
def test_avnir_gcs(info):
    assert product['GDAL_GCS'] == expected['gcs']

@mark.avnir
@mark.gcs_epsg
def test_avnir_gcs_epsg(info):
    assert product['GDAL_GCS_EPSG'] == expected['gcs_epsg']

@mark.avnir
@mark.authority
def test_avnir_authority(info):
    assert product['GDAL_AUTHORITY'] == expected['authority']

@mark.avnir
@mark.subset
def test_avnir_subset(info):
    assert product['GDAL_SUBSET'] == expected['subset']

@mark.avnir
@mark.bands
def test_avnir_bands(info):
    assert product['GDAL_BANDS'] == expected['bands']

#@mark.avnir
#@mark.variables
#def test_avnir_variables(info):
#    assert product['GDAL_VARIABLES'] == expected['variables']
