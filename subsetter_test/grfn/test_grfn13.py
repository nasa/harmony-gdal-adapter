import get_grfn_info
import harmony_requests
import pytest
from pytest import mark
import os.path

# Temporal test to find single granule in subset=time("2020-02-01T23:27:00Z":"2020-02-01T23:27:30Z")
@pytest.fixture()
def info():
    global expected
    global product
    expected = {'cs': 'Geographic',
                'proj_cs':'NA',
                'gcs': 'WGS 84',
                'authority': 'EPSG',
                'proj_epsg': 'NA',
                'gcs_epsg': '4326',
                'subset': [-37.2, -36.8, -69.6, -69.2],
                'bands': 1,
                'variables': ['amplitude', 'NA', 'NA', 'NA']
                }

    product_info = get_grfn_info.get_grfn_info(outfile)
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

@mark.grfn
@mark.status
def test_grfn_status(harmony_url_config):
    base = harmony_url_config.base_url
    grfn_id = harmony_url_config.grfn_id
    path_flag = 'grfn'
    if harmony_url_config.env_flag == 'prod':
        harmony_url = base + grfn_id + '/ogc-api-coverages/1.0.0/collections/science%2Fgrids%2Fdata%2Famplitude/coverage/rangeset?subset=time(%222020-08-13T12%3A03%3A44Z%22%3A%222020-08-13T12%3A03%3A44Z%22)&format=image%2Ftiff'
    else:
        harmony_url = base + grfn_id + '/ogc-api-coverages/1.0.0/collections/science%2Fgrids%2Fdata%2Famplitude/coverage/rangeset?subset=lat(-37.2:-36.8)&subset=lon(-69.6:-69.2)&subset=time(%222020-02-01T23%3A27%3A00Z%22%3A%222020-02-01T23%3A27%3A30Z%22)&format=image%2Ftiff'
    global outfile
    outfile = harmony_url_config.env_flag + '_grfn_query13.tiff'
    get_data_and_status = harmony_requests.harmony_requests(harmony_url, path_flag, outfile)
    assert get_data_and_status == 200

@mark.grfn
@mark.existing_data
def test_grfn_existing_data(harmony_url_config):
    path = './grfn/grfn_products/'
    global outfile
    outfile = harmony_url_config.env_flag + '_grfn_query13.tiff'
    assert os.path.exists(path+outfile) == True

@mark.grfn
@mark.cs
def test_grfn_cs(info):
    assert product['GDAL_CS'] == expected['cs']

@mark.grfn
@mark.projection
def test_grfn_projection(info):
    assert product['GDAL_PROJ_CS'] == expected['proj_cs']

@mark.grfn
@mark.proj_epsg
def test_grfn_proj_epsg(info):
    assert product['GDAL_PROJ_EPSG'] == expected['proj_epsg']

@mark.grfn
@mark.gcs
def test_grfn_gcs(info):
    assert product['GDAL_GCS'] == expected['gcs']

@mark.grfn
@mark.gcs_epsg
def test_grfn_gcs_epsg(info):
    assert product['GDAL_GCS_EPSG'] == expected['gcs_epsg']

@mark.grfn
@mark.authority
def test_grfn_authority(info):
    assert product['GDAL_AUTHORITY'] == expected['authority']

#@mark.grfn
#@mark.subset
#def test_grfn_subset(info):
#    assert product['GDAL_SUBSET'] == expected['subset']

@mark.grfn
@mark.bands
def test_grfn_bands(info):
    assert product['GDAL_BANDS'] == expected['bands']

@mark.grfn
@mark.variables
def test_grfn_variables(info):
    assert product['GDAL_VARIABLES'] == expected['variables']
