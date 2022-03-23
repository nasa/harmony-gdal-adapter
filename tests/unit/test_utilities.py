""" Unit tests for functions in the `gdal_subsetter.utilities.py` module. """
from os.path import exists, join as path_join
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase

from gdal_subsetter.utilities import get_file_type


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
