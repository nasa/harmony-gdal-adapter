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
from pathlib import Path
import sys
from harmony.util import stage, bbox_to_geometry, download, generate_output_filename
from config import UnittestAdapter, get_file_info
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture()
def output_dir():
    return 'data/results'


@pytest.fixture()
def message_files():
    message_path = './data/messages/prod'

    return list(Path(message_path).rglob("*.msg"))


@pytest.fixture()
def adapters(message_files):
    unittest_adapters = []

    for message_file in message_files:
        with open(message_file) as msg_file:
            messagestr = msg_file.read().rstrip()

        unittest_adapters.append(UnittestAdapter(messagestr))

    return unittest_adapters


def test_message(adapters, output_dir):
    for unittest_adapter in adapters:
        download_file(unittest_adapter, output_dir)
        subsetter(unittest_adapter, output_dir)
        subset_result(unittest_adapter, output_dir)


def download_file(unittest_adapter, output_dir):
    """
    This function test if the file pointed by url is downloaded successfully
    to the local space. The url is in the message, which is an attribute in
    the object adapter. This object is created with message file as a global
    object before this function is called. At the last of this function,
    use assert to check if the file is downloaded.
    """
    adapter = unittest_adapter.adapter
    message = adapter.message
    granule = message.granules[0]
    adapter.prepare_output_dir(output_dir)

    url = granule.url

    granule.local_filename = download(
        url, output_dir, logger=None, access_token=None, data=None, cfg=None
    )

    assert os.path.exists(granule.local_filename)

    if os.path.exists(granule.local_filename):
        unittest_adapter.downloaded_file = granule.local_filename
        unittest_adapter.downloaded_success = True


def subsetter(unittest_adapter, output_dir):
    """
    This function test the subset process. It use the functions in
    the global object adapter to do the subset process. At the end
    of this function, use assert to check if the subsetted file exist.
    """

    adapter = unittest_adapter.adapter
    message = adapter.message
    source = message.sources[0]
    granule = source.granules[0]

    input_filename = granule.local_filename
    basename = os.path.basename(input_filename)
    layernames = []

    operations = dict(
        is_variable_subset=True,
        is_regridded=bool(message.format.crs),
        is_subsetted=bool(message.subset and message.subset.bbox)
    )

    result = None
    file_type = adapter.get_filetype(input_filename)

    if file_type == 'tif':
        layernames, result = adapter.process_geotiff(
            source, basename, input_filename, output_dir, layernames
        )

    elif file_type == 'nc':
        layernames, result = adapter.process_netcdf(
            source, basename, input_filename, output_dir, layernames
        )
    elif file_type == 'zip':
        layernames, result = adapter.process_zip(
            source, basename, input_filename, output_dir, layernames
        )
    else:
        logger.exception(e)
        adapter.completed_with_error(
            'No reconized file foarmat, not process')

    # test the result
    assert result

    if result:
        unittest_adapter.subsetted_file = result
        unittest_adapter.subsetted_success = True


def subset_result(unittest_adapter, output_dir):
    """
    This function verifies if the subsetted file experiences
    required process defined by message.
    """
    adapter = unittest_adapter.adapter
    message = adapter.message
    granule = message.granules[0]

    if message.sources[0].variables:
        variables = message.sources[0].variables
    else:
        variables = None

    # check if the download is success
    if not (unittest_adapter.downloaded_success and unittest_adapter.subsetted_success):
        assert False

    # compare output file if it is geotiff
    if adapter.get_filetype(unittest_adapter.subsetted_file) != "tif":
        assert True
        return

    downloaded_file = unittest_adapter.downloaded_file
    subsetted_file = unittest_adapter.subsetted_file
    file_type = adapter.get_filetype(downloaded_file)

    if file_type == 'zip':
        [tiffile, ncfile] = adapter.pack_zipfile(downloaded_file, output_dir)

        if tiffile:
            compare_files(message, tiffile, subsetted_file)

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
                tifffile = adapter.nc2tiff(layer_id, filename, output_dir)
                compare_files(message, tifffile, subsetted_file)

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
            tifffile = adapter.nc2tiff(layer_id, filename, output_dir)
            compare_files(message, tifffile, subsetted_file)

    elif file_type == "tif":
        compare_files(message, downloaded_file, subsetted_file)
    else:
        assert False


def compare_files(message, downloaded_file, subsetted_file):

    info_downloaded = get_file_info(downloaded_file)
    info_subsetted = get_file_info(subsetted_file)

    # test reprojection
    if not message.format.crs:
        assert info_downloaded['proj_wkt'] == info_subsetted['proj_wkt']

    # test spatial subset
    if message.subset.bbox == None and message.subset.shape == None:
        assert info_downloaded['gt'] == info_subsetted['gt']
        assert info_downloaded['xy_size'] == info_subsetted['xy_size']

    # test number of bands
    if message.sources[0].variables:
        variables = message.sources[0].variables
        assert len(variables) == info_subsetted['bands']
    else:
        assert info_downloaded['bands'] == info_subsetted['bands']
