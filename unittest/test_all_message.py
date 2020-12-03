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
from harmony.util import stage, bbox_to_geometry, download, generate_output_filename
from config import test_adapter, get_file_info
########################
#in debug mode
#import pdb
#pdb.set_trace()
########################


#define futures
#@pytest.fixture
#def message_file():
#    messagefile="/home/unittest/data/messages/gfrn/G1234646236-ASF.msg"
#    return messagefile

#@pytest.fixture
#def output_dir():
#    outputdir="/home/unittest/data/results"
#    return outputdir

#general function called by test_funtion

def newadapter(message_file, output_dir):
    if not os.path.isfile(message_file):
        return None
    output_dir=output_dir.rstrip(r"/")
    with open(message_file) as msg_file:
        messagestr = msg_file.read().rstrip()
    return test_adapter(messagestr)

def compare_files(message, downloaded_file,subsetted_file):
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

def download_file(newadapter,output_dir):
    """
    This function test if the file pointed by url is downloaded successfully
    to the local space. The url is in the message, which is an attribute in
    the object adapter. This object is created with message file as a global
    object before this function is called. At the last of this function,
    use assert to check if the file is downloaded.
    """

    adapter = newadapter.adapter
    message = adapter.message
    granule = message.granules[0]
    input_filename = download(granule.url, output_dir, logger=adapter.logger, access_token=message.accessToken)

    adapter.prepare_output_dir(output_dir)
    assert os.path.exists(input_filename)
    if os.path.exists(input_filename):
        newadapter.downloaded_file = input_filename
        newadapter.downloaded_success = True


def subsetter(newadapter, output_dir):
    """
    This function test the subset process. It use the functions in
    the global object adapter to do the subset process. At the end
    of this function, use assert to check if the subsetted file exist.
    """
    adapter = newadapter.adapter
    message = adapter.message
    granule = message.granules[0]
    layernames = []
    operations = dict(
        variable_subset=message.sources[0].variables,
        is_regridded=bool(message.format.crs),
        is_subsetted=bool(message.subset and message.subset.bbox)
    )

    result = None
    basename = os.path.basename(generate_output_filename(granule.url, **operations))
    input_filename = newadapter.downloaded_file
    file_type = adapter.get_filetype(input_filename)

    if file_type == 'tif':
        layernames, result = adapter.process_geotiff(
                message.sources[0], basename, input_filename, output_dir, layernames
                )
    elif file_type == 'nc':
        layernames, result = adapter.process_netcdf(
                message.sources[0], basename, input_filename, output_dir, layernames
                )
    elif file_type == 'zip':
        layernames, result = adapter.process_zip(
                message.sources[0], basename, input_filename, output_dir, layernames
                )
    else:
        adapter.completed_with_error('No reconized file foarmat, not process')
    # test the result
    newadapter.subsetted_file = None
    newadapter.subsetted_success = False
    if result:
        newadapter.subsetted_file = result
        newadapter.subsetted_success = True
        newadapter.var_basename = basename
    assert result


def subset_result(newadapter, output_dir):
    """
    This function verifies if the subsetted file experiences
    required process defined by message.
    """
    adapter = newadapter.adapter
    message = adapter.message
    source = message.sources[0]
    input_filename = adapter.downloaded_file
    basename = newadapter.var_basename

    # granule = message.granules[0]
    if source.variables:
        variables = source.variables
    else:
        variables = None
    # check if the download is success
    if not (newadapter.downloaded_success and newadapter.subsetted_success):
        assert False
    # compare output file if it is geotiff
    if adapter.get_filetype(newadapter.subsetted_file) != "tif":
        assert True
    downloaded_file = newadapter.downloaded_file
    subsetted_file = newadapter.subsetted_file
    file_type = adapter.get_filetype(downloaded_file)
    if file_type == 'zip':
        [tiffile, ncfile] = adapter.pack_zipfile(downloaded_file, output_dir)
        if tiffile:
            compare_files(message, tiffile, subsetted_file)
        if ncfile:
            for variable in variables:
                layer_format = adapter.read_layer_format(
                        source.collection,
                        input_filename,
                        variable.name
                        )
                filename = layer_format.format(
                        input_filename)
                layer_id = basename + '__' + variable.name
                tiffile = adapter.nc2tiff(layer_id, filename, output_dir)
                compare_files(message, tiffile, subsetted_file)
    elif file_type == "nc":
        for variable in variables:
            layer_format = adapter.read_layer_format(
                        source.collection,
                        input_filename,
                        variable.name
                        )
            filename = layer_format.format(
                        input_filename)
            layer_id = basename + '__' + variable.name
            tiffile = adapter.nc2tiff(layer_id, filename, output_dir)
            compare_files(message, tiffile, subsetted_file)
    elif file_type == "tif":
        compare_files(message, downloaded_file, subsetted_file)
    else:
        print("downloaded file has unknown format")
        assert False


########entry of the test script##################

if not os.getenv("EDL_USERNAME"):
    with open(".env_unittest") as envfile:
        test = envfile.read().splitlines()

    res = [i for i in test if "EDL_USERNAME" in i]
    username = res[0].split("=")[1]
    os.environ["EDL_USERNAME"] = username
    res = [i for i in test if "EDL_PASSWORD" in i]
    userpass = res[0].split("=")[1]
    os.environ["EDL_PASSWORD"] = userpass

output_dir = "/home/unittest/data/results"
message_dir = "/home/unittest/data/messages"
i = 0
param_names = "message_file, output_dir"
param_list = []
# r=root, d=directories, f = files
for r, d, f in os.walk(message_dir):
    for file in f:
        if file.endswith(".msg"):
            i = i+1
            message_file = os.path.join(r, file)
            param_list.append((message_file, output_dir))
            # print("unit test against message "+str(i)+": " + message_file)


# dynamically produces the paramters

@pytest.mark.parametrize(param_names, param_list)
def test_one_message(message_file, output_dir):
    adapter_obj = newadapter(message_file, output_dir)
    assert adapter_obj
    download_file(adapter_obj, output_dir)
    subsetter(adapter_obj, output_dir)
    subset_result(adapter_obj, output_dir)


"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--message-file', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    message_file = args.message_file
    output_dir = args.output_dir
    test_one_message(message_file,output_dir)
"""
