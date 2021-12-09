"""
Description:
This script does unit test for transform.py. It includes two tests:
if subset is success, and if subsetted file
agrees with downloaded file.

Paramters:
meassage_file and output_dir, where message_file is created with get_message_file.py,
it captures the message pass to the gdal-subsetter by harmony front service.
output_dir defines the directory where downaloded and subsetted files are stored.

Returns:
None

"""
import unittest.mock
from pathlib import Path

import pytest

from config import UnittestAdapterNoDownload, get_file_info


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

        unittest_adapters.append(UnittestAdapterNoDownload(messagestr))

    return unittest_adapters


@unittest.mock.patch('harmony.aws.stage')
def test_message(stage, adapters, output_dir):
    # Instead of staging to aws, just return the local filename
    stage.side_effect = lambda config, local_filename, remote_filename, mime, logger, location=None: local_filename

    for unittest_adapter in adapters:
        subsetter(unittest_adapter)
        subset_result(unittest_adapter, output_dir)


@unittest.mock.patch('harmony.aws.stage')
def test_world_file_in_output(stage, output_dir):
    # Instead of staging to aws, just return the local filename
    stage.side_effect = lambda config, local_filename, remote_filename, mime, logger, location=None: local_filename

    adapter = UnittestAdapterNoDownload(open('./data/messages/prod/mur/G2145874703-POCLOUD.msg').read().rstrip())

    result = subsetter(adapter)

    assert result.assets['data'].href.endswith('.png')
    assert result.assets['metadata'].href.endswith('.wld')
    assert 'metadata' in result.assets['metadata'].roles


def subsetter(unittest_adapter):
    """
    This function test the subset process. It use the functions in
    the global object adapter to do the subset process. At the end
    of this function, use assert to check if the subsetted file exist.
    """

    adapter = unittest_adapter.adapter
    message = adapter.message
    source = message.sources[0]
    item = next(adapter.catalog.get_items())

    result = adapter.process_item(item, source)

    # test the result
    assert result

    if result:
        unittest_adapter.subsetted_file = result.assets['data'].href
        unittest_adapter.subsetted_success = True

    return result


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
    if message.subset.bbox is None and message.subset.shape is None:
        assert info_downloaded['gt'] == info_subsetted['gt']
        assert info_downloaded['xy_size'] == info_subsetted['xy_size']

    # test number of bands
    if message.sources[0].variables:
        variables = message.sources[0].variables
        assert len(variables) == info_subsetted['bands']
    else:
        assert info_downloaded['bands'] == info_subsetted['bands']
