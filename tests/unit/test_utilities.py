""" Unit tests for functions in the `gdal_subsetter.utilities.py` module. """
from os.path import exists, join as path_join
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase

from gdal_subsetter.utilities import get_file_type, OpenGDAL, rename_file


class TestUtilities(TestCase):
    """ A class to test the functions in the utilities.py module. """
    def setUp(self):
        self.temp_dir = mkdtemp()

    def tearDown(self):
        if exists(self.temp_dir):
            rmtree(self.temp_dir)

    def test_get_file_type(self):
        """ Ensure the type of file can be recognised from the input extension.
            If the type is unknown, that extension should be returned. If the
            file is absent, or the name is None the function should return
            None.

        """
        test_args = [['NetCDF-4 file (.nc4)', 'file.nc4', 'nc'],
                     ['NetCDF-4 file (.nc)', 'file.nc', 'nc'],
                     ['GeoTIFF file (.tif)', 'file.tif', 'tif'],
                     ['GeoTIFF file (.tiff)', 'file.tiff', 'tif'],
                     ['Zip file (.zip)', 'file.zip', 'zip'],
                     ['Unknown extension (.other)', 'file.other', '.other']]

        for description, input_file_basename, expected_file_type in test_args:
            with self.subTest(description):
                full_input_path = path_join(self.temp_dir, input_file_basename)
                # Touch the file to make sure it exists:
                Path(full_input_path).touch()

                self.assertEqual(get_file_type(full_input_path),
                                 expected_file_type)

        with self.subTest('Filename is None, returns None'):
            self.assertIsNone(get_file_type(None))

        with self.subTest('Non-existent file returned None'):
            self.assertIsNone(get_file_type('non_existent.tif'))

    def test_rename_file(self):
        """ Ensure a file is renamed as expected using the harmony-service-lib
            library. The original file should no longer exist, as it should
            have been moved to the new location.

        """
        asset_href = 'https://example.com/ATL03_20200101T000103.nc4'
        expected_output_name = path_join(self.temp_dir,
                                         'ATL03_20200101T000103.nc4')
        input_file_name = path_join(self.temp_dir, 'test.tmp')
        with open(input_file_name, 'a', encoding='utf-8') as file_handler:
            file_handler.write('File content')

        self.assertTrue(exists(input_file_name))

        output_file_name = rename_file(input_file_name, asset_href)
        self.assertEqual(output_file_name, expected_output_name)
        self.assertTrue(exists(output_file_name))
        self.assertFalse(exists(input_file_name))

        with open(output_file_name, 'r', encoding='utf-8') as file_handler:
            output_content = file_handler.read()

        self.assertEqual(output_content, 'File content')

    def test_open_gdal(self):
        """ Ensure that the context manager implementation of OpenGDAL allows
            access to a GeoTIFF file.

        """
        uavsar_granule = ('tests/data/granules/uavsar/'
                          'gulfco_32010_09045_001_090617_L090_CX_01_pauli.tif')

        with OpenGDAL(uavsar_granule) as uavsar_gdal_object:
            gdal_metadata = uavsar_gdal_object.GetMetadata()

        self.assertDictEqual(gdal_metadata,
                             {'AREA_OR_POINT': 'Area',
                              'TIFFTAG_RESOLUTIONUNIT': '1 (unitless)',
                              'TIFFTAG_XRESOLUTION': '1',
                              'TIFFTAG_YRESOLUTION': '1'})
