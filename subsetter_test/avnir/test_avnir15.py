import get_avnir_info
import harmony_requests
import pytest
from pytest import mark
import os.path
import imghdr

# Test to verify that gif file output works
@mark.avnir
@mark.status
def test_avnir_status(harmony_url_config):
    base = harmony_url_config.base_url
    avnir_id = harmony_url_config.avnir_id
    path_flag = 'avnir'

    if harmony_url_config.env_flag == 'prod':
        granule_id = 'G1813212660-ASF'
    else:
        granule_id = 'G1236469528-ASF'


    harmony_url = base + avnir_id + '/ogc-api-coverages/1.0.0/collections/Band1/coverage/rangeset?subset=lat(-.05:0.25)&subset=lon(-51.0:-50.75)&format=image%2Fgif&granuleID=' + granule_id
    global outfile
    outfile = harmony_url_config.env_flag + '_avnir_query15'
    get_data_and_status = harmony_requests.harmony_requests(harmony_url, path_flag, outfile)
    assert get_data_and_status == 200

@mark.avnir
@mark.existing_data
def test_avnir_existing_data(harmony_url_config):
    path = './avnir/avnir_products/'
    global outfile
    outfile = harmony_url_config.env_flag + '_avnir_query15'
    assert os.path.exists(path+outfile) == True

@mark.avnir
@mark.gif_out
def test_avnir_gif_out(harmony_url_config):
    image_type =''
    path = './avnir/avnir_products/'
    global outfile
    outfile = harmony_url_config.env_flag + '_avnir_query15'
    image_type = imghdr.what(path+outfile)
    assert image_type == 'gif'
