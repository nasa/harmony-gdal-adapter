"""
Description:
This script does unit test for transform.py. It includes three tests:
test if download is success, if subset is success, and if subsetted file 
agrees with downloaded file. This script limits to test the message 
with the first variable and the first granule and the tiff format output. 

Paramters:
meassage_file and output_dir, where message_file is created with get_message_file.py,
it captures the message pass to the gdal-subsetter by harmony front service.
output_dir defines the directory where downaloded and subsetted files are stored.

Returns:
None

"""
import argparse
import pytest
import os
import sys
from config import test_adapter, get_file_info
########################
#in debug mode
import pdb
pdb.set_trace()
########################


#get two environment variables: TEST_MESSAGE_FILE, TEST_OUTPUT_DIR

message_file = os.getenv("TEST_MESSAGE_FILE")

output_dir = os.getenv("TEST_OUTPUT_DIR")

if not (message_file and output_dir):

    print("need set TEST_MESSAGE_FILE and TEST_OUTPUT_DIR")

    sys.exit(1)

output_dir=output_dir.rstrip(r"/")

with open(message_file) as msg_file:

    messagestr = msg_file.read().rstrip()

test_adapter = test_adapter(messagestr)

#function

def compare_files(message,downloaded_file,subsetted_file):

    info_downloaded=get_file_info(downloaded_file)
    info_subsetted=get_file_info(subsetted_file)
    #test reprojection
    if not message.format.crs:
        assert info_downloaded['proj_wkt'] == info_subsetted['proj_wkt']
    #test spatial subset
    if message.subset.bbox == None and message.subset.shape == None:
        assert info_downloaded['gt'] == info_subsetted['gt']
        assert info_downloaded['xy_size'] == info_subsetted['xy_size']
    else:
        pass
    #test number of bands
    if message.sources[0].variables:
        variables=message.sources[0].variables
        assert len(variables) == info_subsetted['bands']
    else:
        assert info_downloaded['bands'] == info_subsetted['bands']



#two test functions

def test_download_file():
    """
    This function test if the file pointed by url is downloaded successfully 
    to the local space. The url is in the message, which is an attribute in 
    the object adapter. This object is created with message file as a global 
    object before this function is called. At the last of this function, 
    use assert to check if the file is downloaded.
    """

    adapter =test_adapter.adapter

    message = adapter.message

    granule = message.granules[0]

    adapter.prepare_output_dir(output_dir)

    adapter.download_granules( [ granule ] )

    assert os.path.exists(granule.local_filename)

    if  os.path.exists(granule.local_filename):

        test_adapter.downloaded_file=granule.local_filename

        test_adapter.downloaded_success=True


def test_subsetter():
    """
    This function test the subset process. It use the functions in 
    the global object adapter to do the subset process. At the end 
    of this function, use assert to check if the subsetted file exist.
    """

    adapter =test_adapter.adapter

    message = adapter.message

    granule = message.granules[0]

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

    if result:
  
        test_adapter.subsetted_file=result

        test_adapter.subsetted_success=True


def test_subset_result():
    """
    This function verifies if the subsetted file experiences 
    required process defined by message.
    """
    adapter=test_adapter.adapter

    message = adapter.message

    granule = message.granules[0]

    if message.sources[0].variables:
        
        variables=message.sources[0].variables

    else:

        variables=None
 
    #check if the download is success 

    if not (test_adapter.downloaded_success and test_adapter.subsetted_success):

        assert False

    #compare output file if it is geotiff


    if adapter.get_filetype(test_adapter.subsetted_file) != "tif":

        assert True

        return

    downloaded_file = test_adapter.downloaded_file
    
    subsetted_file = test_adapter.subsetted_file

    file_type = adapter.get_filetype(downloaded_file)
    
    if file_type == 'zip':

        [tiffile, ncfile]=adapter.pack_zipfile(downloaded_file, output_dir)
        
        if tiffile:

            compare_files(message, tiffile,subsetted_file)

        if ncfile:

            for variable in variables:

                layer_format = adapter.read_layer_format(
                        granule.collection,
                        granule.local_filename,
                        variable.name
                        )

                filename = layer_format.format(
                        granule.local_filename)

                layer_id = granule.id + '__' + variable.name

                tifffile=adapter.nc2tiff(layer_id, filename, output_dir)

                compare_files(message, tifffile,subsetted_file)



    elif file_type == "nc":

        for variable in variables:

            layer_format = adapter.read_layer_format(
                        granule.collection,
                        granule.local_filename,
                        variable.name
                        )

            filename = layer_format.format(
                        granule.local_filename)

            layer_id = granule.id + '__' + variable.name
            
            tifffile=adapter.nc2tiff(layer_id, filename, output_dir)

            compare_files(message, tifffile,subsetted_file)


    elif file_type == "tif":
      
        compare_files(message, downloaded_file,subsetted_file)

    else:

        print("downloaded file has unknown format")

        assert False


