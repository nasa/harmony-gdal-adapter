from string import ascii_letters, digits
from random import choice
from unittest import TestCase
from unittest.mock import ANY, call, MagicMock, patch

from harmony.message import Message
from harmony.util import config
from osgeo.gdal import Band as GdalBand, Dataset as GdalDataset

from gdal_subsetter.transform import HarmonyAdapter
from gdal_subsetter.exceptions import IncompatibleVariablesError


def random_file(size=6, chars=ascii_letters + digits):
    return ''.join(choice(chars) for _ in range(size))


class TestRecolor(TestCase):

    def test_recolor_on_tif_does_not_call_gdaldem(self):
        test_message = Message({
            "format": {
                "mime": 'tif'
            },
            "sources": [{
                "collection": "fake_collection",
                "variables": []
            }]
        })
        test_adapter = HarmonyAdapter(test_message, '', None)

        expected = 'sourcefile'
        with patch.object(test_adapter, 'cmd') as mock_cmd:
            actual = test_adapter.recolor('granId__subsetted', expected,
                                          'dstdir')
            self.assertEqual(expected, actual, 'file has changed')
            mock_cmd.assert_not_called()

    @patch('gdal_subsetter.transform.copyfile')
    def test_recolor_on_png_calls_gdaldem(self, copyfile_mock):
        test_message = Message({
            "format": {
                "mime": 'image/png'
            },
            "sources": [{
                "collection": "fake_collection",
                "variables": []
            }]
        })
        test_adapter = HarmonyAdapter(test_message, '', None)

        layer_id = 'granId__subsetted'
        dest_dir = 'destdir'
        expected_outfile = f'{dest_dir}/{layer_id}__colored.png'

        with patch.object(test_adapter, 'cmd') as cmd_mock:
            actual = test_adapter.recolor(layer_id, 'srcfile', dest_dir)
            self.assertEqual(expected_outfile, actual,
                             'incorrect output file generated')
            cmd_mock.assert_called_once_with('gdaldem', 'color-relief',
                                             '-alpha', '-of', 'PNG', '-co',
                                             'WORLDFILE=YES', 'srcfile', ANY,
                                             expected_outfile)

            copyfile_mock.assert_any_call(expected_outfile,
                                          f'{dest_dir}/result.png')
            copyfile_mock.assert_any_call(
                expected_outfile.replace('.png', '.wld'),
                f'{dest_dir}/result.wld')


class TestAddToList(TestCase):

    @patch.object(HarmonyAdapter, 'stack_multi_file_with_metadata')
    @patch.object(HarmonyAdapter, 'is_stackable')
    def test_add_to_result_stacks_if_stackable_with_netcdf(self,
                                                           is_stackable_mock,
                                                           stack_multi_mock):

        filelist = [random_file(), random_file()]
        dstdir = random_file()
        expected = f'{dstdir}/result.tif'

        test_adapter = HarmonyAdapter(
            Message({"format": {"mime": "application/x-netcdf4"}}), '', None
        )

        stack_multi_mock.return_value = expected
        is_stackable_mock.return_value = True

        actual = test_adapter.add_to_result(filelist, dstdir)

        stack_multi_mock.assert_called_once_with(filelist, expected)
        self.assertEqual(expected, actual)

    @patch.object(HarmonyAdapter, 'stack_multi_file_with_metadata')
    @patch.object(HarmonyAdapter, 'is_stackable')
    def test_add_to_result_raises_if_unstackable_with_netcdf(self,
                                                             is_stackable_mock,
                                                             stack_multi_mock):

        filelist = [random_file(), random_file()]
        dstdir = random_file()
        dstfile = f'{dstdir}/result.tif'

        test_adapter = HarmonyAdapter(
            Message({"format": {"mime": "application/x-netcdf4"}}), '', None
        )

        stack_multi_mock.return_value = dstfile
        is_stackable_mock.return_value = False

        with self.assertRaises(IncompatibleVariablesError) as error:
            test_adapter.add_to_result(filelist, dstdir)

        stack_multi_mock.assert_not_called()
        self.assertEqual(
            'Request cannot be completed: datasets are incompatible and cannot be combined.',
            str(error.exception)
        )

    @patch.object(HarmonyAdapter, 'stack_multi_file_with_metadata')
    @patch.object(HarmonyAdapter, 'is_stackable')
    def test_add_to_result_stacks_with_png(self, is_stackable_mock,
                                           stack_multi_mock):

        filelist = [random_file(), random_file()]
        dstdir = random_file()
        expected = f'{dstdir}/result.png'

        test_adapter = HarmonyAdapter(
            Message({"format": {"mime": "png"}}), '', None
        )

        is_stackable_mock.return_value = False

        actual = test_adapter.add_to_result(filelist, dstdir)

        stack_multi_mock.assert_not_called()
        self.assertEqual(expected, actual)

    @patch.object(HarmonyAdapter, 'stack_multi_file_with_metadata')
    @patch.object(HarmonyAdapter, 'is_stackable')
    def test_add_to_result_stacks_with_jpeg(self, is_stackable_mock,
                                            stack_multi_mock):

        filelist = [random_file(), random_file()]
        dstdir = random_file()
        expected = f'{dstdir}/result.jpeg'

        test_adapter = HarmonyAdapter(
            Message({"format": {"mime": "jpeg"}}), '', None
        )

        is_stackable_mock.return_value = False

        actual = test_adapter.add_to_result(filelist, dstdir)

        stack_multi_mock.assert_not_called()
        self.assertEqual(expected, actual)

class TestIsStackable(TestCase):
    """ Ensure that the method correctly identifies when files can be stacked,
        according to their geotransform, projection, raster size and file
        formats.

    """

    @classmethod
    def setUpClass(cls):
        """ Define test fixtures to be shared between tests. """
        cls.config = config(validate=False)
        cls.projection = 'PROJCS[CEA...]'
        cls.geo_transform = (-10000, 0, 1000, 10000, 0, -1000)
        cls.data_type = '1'
        cls.raster_x_size = 1000
        cls.raster_y_size = 500

    def setUp(self):
        """ Define test fixtures that are not shared between tests. """
        self.raster_band_one = MagicMock(spec=GdalBand)
        self.raster_band_one.DataType = self.data_type

        self.geotiff_one = MagicMock(spec=GdalDataset)
        self.geotiff_one.RasterXSize = self.raster_x_size
        self.geotiff_one.RasterYSize = self.raster_y_size
        self.geotiff_one.GetRasterBand.return_value = self.raster_band_one
        self.geotiff_one.GetProjection.return_value = self.projection
        self.geotiff_one.GetGeoTransform.return_value = self.geo_transform

    @patch('gdal_subsetter.transform.gdal.Open')
    def test_no_inputs_returns_false(self, mock_gdal_open):
        """ The checks against each list of input options (e.g., projection,
            geotransform, etc) should each return False.

        """
        test_adapter = HarmonyAdapter(Message({}), self.config, None)

        self.assertFalse(test_adapter.is_stackable([]))
        mock_gdal_open.assert_not_called()

    @patch('gdal_subsetter.transform.gdal.Open')
    def test_stackable_inputs_return_true(self, mock_gdal_open):
        """ Files with the same geotransform, projection, data type and
            raster size should return true (if they are not PNG files).

        """
        mock_raster_band_two = MagicMock(spec=GdalBand)
        mock_raster_band_two.DataType = self.data_type

        mock_geotiff_two = MagicMock(spec=GdalDataset)
        mock_geotiff_two.RasterXSize = self.raster_x_size
        mock_geotiff_two.RasterYSize = self.raster_y_size
        mock_geotiff_two.GetRasterBand.return_value = mock_raster_band_two
        mock_geotiff_two.GetProjection.return_value = self.projection
        mock_geotiff_two.GetGeoTransform.return_value = self.geo_transform

        mock_gdal_open.side_effect = [self.geotiff_one, mock_geotiff_two]

        test_adapter = HarmonyAdapter(Message({}), self.config, None)

        self.assertTrue(test_adapter.is_stackable(['file1.tif', 'file2.tif']))
        self.assertEqual(mock_gdal_open.call_count, 2)
        mock_gdal_open.assert_has_calls([call('file1.tif'), call('file2.tif')])

    @patch('gdal_subsetter.transform.gdal.Open')
    def test_stackable_png_returns_false(self, mock_gdal_open):
        """ PNG files should always return False, even if their other
            information matches.

        """
        mock_raster_band_two = MagicMock(spec=GdalBand)
        mock_raster_band_two.DataType = self.data_type

        mock_geotiff_two = MagicMock(spec=GdalDataset)
        mock_geotiff_two.RasterXSize = self.raster_x_size
        mock_geotiff_two.RasterYSize = self.raster_y_size
        mock_geotiff_two.GetRasterBand.return_value = mock_raster_band_two
        mock_geotiff_two.GetProjection.return_value = self.projection
        mock_geotiff_two.GetGeoTransform.return_value = self.geo_transform

        mock_gdal_open.side_effect = [self.geotiff_one, mock_geotiff_two]

        test_adapter = HarmonyAdapter(Message({}), self.config, None)

        self.assertFalse(test_adapter.is_stackable(['file1.png', 'file2.png']))
        self.assertEqual(mock_gdal_open.call_count, 2)
        mock_gdal_open.assert_has_calls([call('file1.png'), call('file2.png')])

    @patch('gdal_subsetter.transform.gdal.Open')
    def test_stackable_diff_geotransform_returns_false(self, mock_gdal_open):
        """ Files with differing geo-transforms are not stackable. """
        mock_raster_band_two = MagicMock(spec=GdalBand)
        mock_raster_band_two.DataType = self.data_type

        mock_geotiff_two = MagicMock(spec=GdalDataset)
        mock_geotiff_two.RasterXSize = self.raster_x_size
        mock_geotiff_two.RasterYSize = self.raster_y_size
        mock_geotiff_two.GetRasterBand.return_value = mock_raster_band_two
        mock_geotiff_two.GetProjection.return_value = self.projection
        mock_geotiff_two.GetGeoTransform.return_value = (-5000, 0, 500,
                                                         5000, 0, -500)

        mock_gdal_open.side_effect = [self.geotiff_one, mock_geotiff_two]

        test_adapter = HarmonyAdapter(Message({}), self.config, None)

        self.assertFalse(test_adapter.is_stackable(['file1.tif', 'file2.tif']))
        self.assertEqual(mock_gdal_open.call_count, 2)
        mock_gdal_open.assert_has_calls([call('file1.tif'), call('file2.tif')])

    @patch('gdal_subsetter.transform.gdal.Open')
    def test_stackable_diff_projection_returns_false(self, mock_gdal_open):
        """ Files with differing projections are not stackable. """
        mock_raster_band_two = MagicMock(spec=GdalBand)
        mock_raster_band_two.DataType = self.data_type

        mock_geotiff_two = MagicMock(spec=GdalDataset)
        mock_geotiff_two.RasterXSize = self.raster_x_size
        mock_geotiff_two.RasterYSize = self.raster_y_size
        mock_geotiff_two.GetRasterBand.return_value = mock_raster_band_two
        mock_geotiff_two.GetProjection.return_value = 'PROJCS[LAEA...]'
        mock_geotiff_two.GetGeoTransform.return_value = self.geo_transform

        mock_gdal_open.side_effect = [self.geotiff_one, mock_geotiff_two]

        test_adapter = HarmonyAdapter(Message({}), self.config, None)

        self.assertFalse(test_adapter.is_stackable(['file1.tif', 'file2.tif']))
        self.assertEqual(mock_gdal_open.call_count, 2)
        mock_gdal_open.assert_has_calls([call('file1.tif'), call('file2.tif')])

    @patch('gdal_subsetter.transform.gdal.Open')
    def test_stackable_different_dtype_returns_false(self, mock_gdal_open):
        """ Files with differing dtypes are not stackable. """
        mock_raster_band_two = MagicMock(spec=GdalBand)
        mock_raster_band_two.DataType = '2'

        mock_geotiff_two = MagicMock(spec=GdalDataset)
        mock_geotiff_two.RasterXSize = self.raster_x_size
        mock_geotiff_two.RasterYSize = self.raster_y_size
        mock_geotiff_two.GetRasterBand.return_value = mock_raster_band_two
        mock_geotiff_two.GetProjection.return_value = self.projection
        mock_geotiff_two.GetGeoTransform.return_value = self.geo_transform

        mock_gdal_open.side_effect = [self.geotiff_one, mock_geotiff_two]

        test_adapter = HarmonyAdapter(Message({}), self.config, None)

        self.assertFalse(test_adapter.is_stackable(['file1.png', 'file2.png']))
        self.assertEqual(mock_gdal_open.call_count, 2)
        mock_gdal_open.assert_has_calls([call('file1.png'), call('file2.png')])

    @patch('gdal_subsetter.transform.gdal.Open')
    def test_stackable_different_xsize_returns_false(self, mock_gdal_open):
        """ Files with differing raster x-sizes are not stackable. """
        mock_raster_band_two = MagicMock(spec=GdalBand)
        mock_raster_band_two.DataType = self.data_type

        mock_geotiff_two = MagicMock(spec=GdalDataset)
        mock_geotiff_two.RasterXSize = 500
        mock_geotiff_two.RasterYSize = self.raster_y_size
        mock_geotiff_two.GetRasterBand.return_value = mock_raster_band_two
        mock_geotiff_two.GetProjection.return_value = self.projection
        mock_geotiff_two.GetGeoTransform.return_value = self.geo_transform

        mock_gdal_open.side_effect = [self.geotiff_one, mock_geotiff_two]

        test_adapter = HarmonyAdapter(Message({}), self.config, None)

        self.assertFalse(test_adapter.is_stackable(['file1.png', 'file2.png']))
        self.assertEqual(mock_gdal_open.call_count, 2)
        mock_gdal_open.assert_has_calls([call('file1.png'), call('file2.png')])

    @patch('gdal_subsetter.transform.gdal.Open')
    def test_stackable_different_ysize_returns_false(self, mock_gdal_open):
        """ Files with differing raster y-sizes are not stackable. """
        mock_raster_band_two = MagicMock(spec=GdalBand)
        mock_raster_band_two.DataType = self.data_type

        mock_geotiff_two = MagicMock(spec=GdalDataset)
        mock_geotiff_two.RasterXSize = self.raster_x_size
        mock_geotiff_two.RasterYSize = 1000
        mock_geotiff_two.GetRasterBand.return_value = mock_raster_band_two
        mock_geotiff_two.GetProjection.return_value = self.projection
        mock_geotiff_two.GetGeoTransform.return_value = self.geo_transform

        mock_gdal_open.side_effect = [self.geotiff_one, mock_geotiff_two]

        test_adapter = HarmonyAdapter(Message({}), self.config, None)

        self.assertFalse(test_adapter.is_stackable(['file1.png', 'file2.png']))
        self.assertEqual(mock_gdal_open.call_count, 2)
        mock_gdal_open.assert_has_calls([call('file1.png'), call('file2.png')])

    @patch('gdal_subsetter.transform.gdal.Open')
    def test_stackable_multiple_diffs_returns_false(self, mock_gdal_open):
        """ Files which differ in all properties are not stackable. """
        mock_raster_band_two = MagicMock(spec=GdalBand)
        mock_raster_band_two.DataType = '2'

        mock_geotiff_two = MagicMock(spec=GdalDataset)
        mock_geotiff_two.RasterXSize = 2000
        mock_geotiff_two.RasterYSize = 1000
        mock_geotiff_two.GetRasterBand.return_value = mock_raster_band_two
        mock_geotiff_two.GetProjection.return_value = 'PROJCS[LAEA...]'
        mock_geotiff_two.GetGeoTransform.return_value = (-5000, 0, 500,
                                                         5000, 0, -500)

        mock_gdal_open.side_effect = [self.geotiff_one, mock_geotiff_two]

        test_adapter = HarmonyAdapter(Message({}), self.config, None)

        self.assertFalse(test_adapter.is_stackable(['file1.tif', 'file2.tif']))
        self.assertEqual(mock_gdal_open.call_count, 2)
        mock_gdal_open.assert_has_calls([call('file1.tif'), call('file2.tif')])

    @patch('gdal_subsetter.transform.gdal.Open')
    def test_stackable_png_and_diff_ysize_returns_false(self, mock_gdal_open):
        """ This test confirms that a PNG file, when combined with a difference
            between the read files (raster Y size) will still return False,
            indicating the files cannot be stacked.

        """
        mock_raster_band_two = MagicMock(spec=GdalBand)
        mock_raster_band_two.DataType = self.data_type

        mock_geotiff_two = MagicMock(spec=GdalDataset)
        mock_geotiff_two.RasterXSize = self.raster_x_size
        mock_geotiff_two.RasterYSize = 1000
        mock_geotiff_two.GetRasterBand.return_value = mock_raster_band_two
        mock_geotiff_two.GetProjection.return_value = self.projection
        mock_geotiff_two.GetGeoTransform.return_value = self.geo_transform

        mock_gdal_open.side_effect = [self.geotiff_one, mock_geotiff_two]

        test_adapter = HarmonyAdapter(Message({}), self.config, None)

        self.assertFalse(test_adapter.is_stackable(['file1.tif', 'file2.png']))
        self.assertEqual(mock_gdal_open.call_count, 2)
        mock_gdal_open.assert_has_calls([call('file1.tif'), call('file2.png')])
