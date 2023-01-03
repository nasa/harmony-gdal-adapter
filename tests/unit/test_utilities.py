""" Unit tests for functions in the `gdal_subsetter.utilities.py` module. """
from os.path import exists, join as path_join
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase

from gdal_subsetter.utilities import (get_file_type, get_files_from_unzipfiles,
                                      has_world_file, is_geotiff, OpenGDAL,
                                      rename_file)


class TestUtilities(TestCase):
    """ A class to test the functions in the utilities.py module. """
    @classmethod
    def setUpClass(cls):
        """ Define items that can be shared between tests. """
        cls.granule_dir = 'tests/data/granules'
        cls.uavsar_granule = path_join(
            cls.granule_dir, 'uavsar',
            'gulfco_32010_09045_001_090617_L090_CX_01_pauli.tif'
        )
        cls.sentinel_granule = path_join(
            cls.granule_dir, 'gfrn',
            'S1-GUNW-D-R-083-tops-20141116_20141023-095646-360325S_38126S-PP-24b3-v2_0_2.nc'
        )

    def setUp(self):
        """ Define items that need to be unique to each test. """
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
        with OpenGDAL(self.uavsar_granule) as uavsar_gdal_object:
            gdal_metadata = uavsar_gdal_object.GetMetadata()

        self.assertDictEqual(gdal_metadata,
                             {'AREA_OR_POINT': 'Area',
                              'TIFFTAG_RESOLUTIONUNIT': '1 (unitless)',
                              'TIFFTAG_XRESOLUTION': '1',
                              'TIFFTAG_YRESOLUTION': '1'})

    def test_is_geotiff(self):
        """ Ensure that a file is correctly recognised as a GeoTIFF. """
        with self.subTest('A GeoTIFF granule returns True'):
            self.assertTrue(is_geotiff(self.uavsar_granule))

        with self.subTest('A NetCDF-4 granules returns False'):
            self.assertFalse(is_geotiff(self.sentinel_granule))

    def test_get_files_from_unzipfiles(self):
        """ Ensure that files extracted from a zip file can be filtered based
            on their extensions. In addition, if variables are specified, and
            are not "Band1", "Band2", etc, the output file list should be
            further filtered to only those paths that match to a variable name.

            The tests below verify that files with ".nc" and ".nc4" extensions
            are both recognised as having type "nc", while files with ".tif"
            and ".tiff" extensions are both recognised as GeoTIFFs.

        """
        netcdf4_files = [path_join(self.temp_dir, 'granule_amplitude.nc'),
                         path_join(self.temp_dir, 'granule_coherence.nc4'),
                         path_join(self.temp_dir, 'granule_variable-one.nc'),
                         path_join(self.temp_dir, 'granule_variable_two.nc')]
        geotiff_files = [path_join(self.temp_dir, 'granule_amplitude.tif'),
                         path_join(self.temp_dir, 'granule_coherence.tiff')]

        for netcdf4_file in netcdf4_files:
            with open(netcdf4_file, 'a', encoding='utf-8') as file_handler:
                file_handler.write(netcdf4_file)

        for geotiff_file in geotiff_files:
            with open(geotiff_file, 'a', encoding='utf-8') as file_handler:
                file_handler.write(geotiff_file)

        with self.subTest('No variables, all GeoTIFF files are retrieved.'):
            self.assertListEqual(
                get_files_from_unzipfiles(self.temp_dir, 'tif',
                                          variable_names=[]),
                geotiff_files
            )

        with self.subTest('No variables, all NetCDF-4 files are retrieved.'):
            self.assertListEqual(
                get_files_from_unzipfiles(self.temp_dir, 'nc'),
                netcdf4_files
            )

        with self.subTest('Only files matching variable names are retrieved.'):
            self.assertListEqual(
                get_files_from_unzipfiles(self.temp_dir, 'nc',
                                          variable_names=['amplitude']),
                [netcdf4_files[0]]
            )

        with self.subTest('No files matching variables returns empty list.'):
            self.assertListEqual(
                get_files_from_unzipfiles(self.temp_dir, 'nc',
                                          variable_names=['wind_speed']),
                []
            )

        with self.subTest('Variable names are ignored if they contain "Band"'):
            self.assertListEqual(
                get_files_from_unzipfiles(self.temp_dir, 'tif',
                                          variable_names=['Band1', 'Band2']),
                geotiff_files
            )

        with self.subTest('Variable name hyphens converted to underscores.'):
            self.assertListEqual(
                get_files_from_unzipfiles(self.temp_dir, 'nc',
                                          variable_names=['variable-two']),
                [netcdf4_files[3]]
            )

        with self.subTest('File name hyphens converted to underscores.'):
            self.assertListEqual(
                get_files_from_unzipfiles(self.temp_dir, 'nc',
                                          variable_names=['variable_one']),
                [netcdf4_files[2]]
            )

    def test_has_world_file(self):
        """ Ensure that files are correctly identified as having an associated
            ESRI world file based on their MIME type.

        """
        with self.subTest('PNG returns True.'):
            self.assertTrue(has_world_file('image/png'))

        with self.subTest('JPEG returns True.'):
            self.assertTrue(has_world_file('image/jpeg'))

        with self.subTest('GeoTIFF returns False.'):
            self.assertFalse(has_world_file('image/tiff'))

        with self.subTest('NetCDF-4 returns False.'):
            self.assertFalse(has_world_file('application/x-netcdf4'))
