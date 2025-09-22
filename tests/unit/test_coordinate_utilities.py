from unittest import TestCase
from unittest.mock import Mock, patch

from osgeo import gdal

from gdal_subsetter.coordinate_utilities import (
    calc_coord_ij,
    calc_ij_coord,
    is_rotated_geotransform,
)


class TestCoordinateUtilities(TestCase):
    """A class to test functions in the coordinate_utilities.py module."""

    def test_calc_coord_ij(self):
        """Ensure the expected row and column indices are retrieved for a
        given geotransform, x coordinate and y coordinate.

        """
        self.assertTupleEqual(calc_coord_ij((10, 1, 0, 20, 0, 2), 20, 60), (10, 20))

    def test_calc_ij_coord(self):
        """Ensure the expected values are retrieved for a given geotransform,
        row index and column index.

        In the geotransform below, the corner coordinates are (10, 20),
        each column is separated by 1 unit and each row by 2 units.

        """
        self.assertTupleEqual(calc_ij_coord((10, 1, 0, 20, 0, 2), 10, 20), (20, 60))

    @patch("gdal_subsetter.utilities.gdal.Open")
    def test_is_rotated_geotransform(self, mock_gdal_open):
        """Ensure a geotransform is correctly identified as having a non-zero
        components relating to rotation.

        """
        mock_gdal_open.return_value = Mock(spec=gdal.Dataset)

        with self.subTest("Non rotated transform returns false"):
            mock_gdal_open.return_value.GetGeoTransform.return_value = (
                -10,
                1,
                0,
                20,
                0,
                -5,
            )
            self.assertFalse(is_rotated_geotransform("file.tif"))

        with self.subTest("Rotational x component returns true"):
            mock_gdal_open.return_value.GetGeoTransform.return_value = (
                -10,
                1,
                2,
                20,
                0,
                -5,
            )
            self.assertTrue(is_rotated_geotransform("file.tif"))

        with self.subTest("Rotational y component returns true"):
            mock_gdal_open.return_value.GetGeoTransform.return_value = (
                -10,
                1,
                0,
                20,
                3,
                -5,
            )
            self.assertTrue(is_rotated_geotransform("file.tif"))

        with self.subTest("Rotational x and y components return true"):
            mock_gdal_open.return_value.GetGeoTransform.return_value = (
                -10,
                1,
                2,
                20,
                3,
                -5,
            )
            self.assertTrue(is_rotated_geotransform("file.tif"))
