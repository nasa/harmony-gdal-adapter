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
from os.path import abspath, dirname, join as join_path
from pathlib import Path
import unittest.mock

import pytest

from gdal_subsetter.utilities import get_file_type
from tests.config import UnittestAdapterNoDownload, get_file_info


@pytest.fixture
def data_dir():
    """ A test fixture pointing at the directory containing test data. """
    return join_path(dirname(abspath(__file__)), 'data')


@pytest.fixture
def output_dir(data_dir):
    """ A test fixture pointing to an output directory for test restults. """
    return join_path(data_dir, 'results')


@pytest.fixture
def messages_dir(data_dir):
    """ A test fixture pointing to a directory containing Harmony messages. """
    return join_path(data_dir, 'messages/prod')

@pytest.fixture
def message_files(messages_dir):
    """ A test fixture returning a list of Harmony message files. """
    return list(Path(messages_dir).rglob("*.msg"))


@pytest.fixture
def adapters(message_files):
    unittest_adapters = []

    for message_file in message_files:
        with open(message_file, 'r') as msg_file:
            messagestr = msg_file.read().rstrip()

        unittest_adapters.append(UnittestAdapterNoDownload(messagestr))

    return unittest_adapters


@unittest.mock.patch('harmony.aws.stage')
def test_message(stage, adapters, output_dir, message_files):
    # Instead of staging to aws, just return the local filename
    stage.side_effect = lambda config, local_filename, remote_filename, mime, logger, location=None: local_filename

    for adapter_index, unittest_adapter in enumerate(adapters):
        # Omit MUR test, as it takes too long
        print(message_files[adapter_index].name)
        if not message_files[adapter_index].name.endswith('G2145874703-POCLOUD.msg'):
            print('Running test')
            subsetter(unittest_adapter)
            subset_result(unittest_adapter, output_dir)


@pytest.mark.skip(reason='MUR tests currently take too long')
@unittest.mock.patch('harmony.aws.stage')
def test_world_file_in_output(stage, output_dir, messages_dir):
    # Instead of staging to aws, just return the local filename
    stage.side_effect = lambda config, local_filename, remote_filename, mime, logger, location=None: local_filename

    with open(join_path(messages_dir, 'mur/G2145874703-POCLOUD.msg')) as file_handler:
        message = file_handler.read().rstrip()

    adapter = UnittestAdapterNoDownload(message)

    result = subsetter(adapter)

    assert result.assets['data'].href.endswith('.png')
    assert result.assets['metadata'].href.endswith('.wld')
    assert 'metadata' in result.assets['metadata'].roles


def subsetter(unittest_adapter):
    """
    This function tests the subset process. It uses the functions in
    the global object adapter to do the subset process. At the end
    of this function,  assert to check if the subsetted file exist.
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
    downloaded_file_type = get_file_type(unittest_adapter.downloaded_file)
    subsetted_file_type = get_file_type(unittest_adapter.subsetted_file)

    if subsetted_file_type != 'tif':
        # this test only checks tif files everything else passes.
        assert True
        return

    if downloaded_file_type == 'zip':
        [tiffile, ncfile] = adapter.pack_zipfile(
            unittest_adapter.downloaded_file, output_dir
        )

        if tiffile:
            compare_files(message, tiffile, unittest_adapter.subsetted_file)

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
                compare_files(message, tifffile,
                              unittest_adapter.subsetted_file)

    elif downloaded_file_type == 'nc':
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
            compare_files(message, tifffile, unittest_adapter.subsetted_file)

    elif downloaded_file_type == 'tif':
        compare_files(message, unittest_adapter.downloaded_file,
                      unittest_adapter.subsetted_file)
    else:
        assert False, 'Unexpected download type'


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
