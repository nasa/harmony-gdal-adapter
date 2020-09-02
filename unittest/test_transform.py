#This script does unit test for transform.py
#Input is the meassage


import pytest
import os,sys
from config import test_adapter
########################
#in debug mode
#import pdb
#pdb.set_trace()
########################


#get two environment variables: TEST_MESSAGE_FILE, TEST_OUTPUT_DIR

message_file = os.getenv("TEST_MESSAGE_FILE")

output_dir = os.getenv("TEST_OUTPUT_DIR")

if not (message_file and output_dir):

    print("need set TEST_MESSAGE_FILE and TEST_OUTPUT_DIR")

    sys.exit(1)


with open(message_file) as msg_file:
    messagestr = msg_file.read().rstrip()

adapter = test_adapter(messagestr).adapter

#two test functions

def test_download_file():

    message = adapter.message

    granules = message.granules

    granules = granules[:1]

    adapter.prepare_output_dir(output_dir)

    granule = granules[0]    

    adapter.download_granules( [ granule ] )

    assert os.path.exists(granule.local_filename)


def test_subsetter():
    
    message = adapter.message

    granules = message.granules

    granules = granules[:1]

    granule  = granules[0]

    layernames = []

    operations = dict(
        is_variable_subset=True,
        is_regridded=bool(message.format.crs),
        is_subsetted=bool(message.subset and message.subset.bbox)
        )

    result = None
        
    file_type = adapter.get_filetype(granule.local_filename)

    if file_type == 'tif':
        layernames, result = adapter.process_geotiff(
                granule,output_dir,layernames,operations,message.isSynchronous
                )

    elif file_type == 'nc':
        layernames, result = adapter.process_netcdf(
                granule,output_dir,layernames,operations,message.isSynchronous
                )
    elif file_type == 'zip':
        layernames, result = adapter.process_zip(
                granule,output_dir,layernames,operations,message.isSynchronous
                )
    else:
        logger.exception(e)
        adapter.completed_with_error('No reconized file foarmat, not process')

    #test the result
    assert result

