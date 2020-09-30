import get_grfn_info
import harmony_requests
import pytest
from pytest import mark
import os.path
import imghdr
# Test to verify that tiff file output works
@mark.grfn    
@mark.status
def test_grfn_status(harmony_url_config):
    base = harmony_url_config.base_url
    grfn_id = harmony_url_config.grfn_id
    path_flag = 'grfn'

    if harmony_url_config.env_flag == 'prod':
        granule_id = 'G1715962900-ASF'
    else:
        granule_id = 'G1234646236-ASF'

    harmony_url = base + grfn_id + '/ogc-api-coverages/1.0.0/collections/science%2Fgrids%2Fdata%2Famplitude/coverage/rangeset?subset=lat(33:33.1)&subset=lon(-115.5:-115.25)&format=image%2Ftiff&granuleID=' + granule_id
    global outfile
    outfile = harmony_url_config.env_flag + '_grfn_query17'
    get_data_and_status = harmony_requests.harmony_requests(harmony_url, path_flag, outfile)
    assert get_data_and_status == 200

@mark.grfn
@mark.existing_data
def test_grfn_existing_data(harmony_url_config):
    path = './grfn/grfn_products/'
    global outfile
    outfile = harmony_url_config.env_flag + '_grfn_query17'
    assert os.path.exists(path+outfile) == True

@mark.grfn
@mark.tiff_out
def test_grfn_tiff_out(harmony_url_config):
    image_type =''
    path = './grfn/grfn_products/'
    global outfile
    outfile = harmony_url_config.env_flag + '_grfn_query17'
    image_type = imghdr.what(path+outfile)
    assert image_type == 'tiff'
