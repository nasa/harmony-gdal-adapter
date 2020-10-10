import get_uavsar_info
import harmony_requests
import pytest
from pytest import mark
import os.path
import imghdr

#Test that tiff file format works
@mark.uavsar    
@mark.status
def test_uavsar_status(harmony_url_config):
    base = harmony_url_config.base_url
    uavsar_id = harmony_url_config.uavsar_id
    path_flag = 'uavsar'

    if harmony_url_config.env_flag == 'prod':
        granule_id = 'G1422449017-ASF'
    else:
        granule_id = 'G1233284367-ASF'

    harmony_url = base + uavsar_id + '/ogc-api-coverages/1.0.0/collections/Band1/coverage/rangeset?subset=lat(63.7:64.1)&subset=lon(-145.9:-145.7)&format=image%2Ftiff&granuleID=' + granule_id
    global outfile
    outfile = harmony_url_config.env_flag + '_uavsar_query18'
    get_data_and_status = harmony_requests.harmony_requests(harmony_url, path_flag, outfile)
    assert get_data_and_status == 200

@mark.uavsar
@mark.existing_data
def test_uavsar_existing_data(harmony_url_config):
    path = './uavsar/uavsar_products/'
    global outfile
    outfile = harmony_url_config.env_flag + '_uavsar_query18'
    assert os.path.exists(path+outfile) == True

@mark.uavsar
@mark.tiff_out
def test_uavsar_tiff_out(harmony_url_config):
    image_type =''
    path = './uavsar/uavsar_products/'
    global outfile
    outfile = harmony_url_config.env_flag + '_uavsar_query18'
    image_type = imghdr.what(path+outfile)
    assert image_type == 'tiff'
