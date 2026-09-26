"""Preserve zero-valued no-data metadata during real GeoTIFF conversions."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from netCDF4 import Dataset
from osgeo import gdal, osr

from gdal_subsetter.reformat import convert_geotiff_to_netcdf


class TestZeroNoData(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "source.tif"
        self.output = Path(self.directory.name) / "output.nc"

    def create_geotiff(self, values, epsg, nodata, rotated=False, bands=1):
        gdal_type = {
            np.dtype("uint8"): gdal.GDT_Byte,
            np.dtype("int16"): gdal.GDT_Int16,
            np.dtype("float32"): gdal.GDT_Float32,
        }[values.dtype]
        dataset = gdal.GetDriverByName("GTiff").Create(
            str(self.source), values.shape[1], values.shape[0], bands, gdal_type
        )
        crs = osr.SpatialReference()
        crs.ImportFromEPSG(epsg)
        dataset.SetProjection(crs.ExportToWkt())
        origin_x, origin_y, step = (
            (-80, 40, 0.1) if epsg == 4326 else (500000, 4400000, 100)
        )
        transform = (
            origin_x,
            step,
            step / 4 if rotated else 0,
            origin_y,
            step / 8 if rotated else 0,
            -step,
        )
        dataset.SetGeoTransform(transform)
        dataset.SetMetadataItem("source_marker", "unchanged")
        for number in range(1, bands + 1):
            band = dataset.GetRasterBand(number)
            if nodata is not None:
                band.SetNoDataValue(nodata)
            band.SetMetadataItem("standard_name", f"signal_{number}")
            band.SetMetadataItem("units", "unitless")
            band.WriteArray(values)
        dataset = None
        return transform

    def convert(self):
        before = sha256(self.source.read_bytes()).digest()
        self.assertEqual(
            convert_geotiff_to_netcdf(str(self.source), str(self.output)),
            str(self.output),
        )
        self.assertEqual(sha256(self.source.read_bytes()).digest(), before)

    def check_output(self, values, epsg, nodata, bands=1):
        missing = np.isnan(values) if np.isnan(nodata) else values == nodata
        with Dataset(self.output) as dataset:
            dimensions = ("lat", "lon") if epsg == 4326 else ("y", "x")
            self.assertEqual(dataset.source_marker, "unchanged")
            self.assertEqual(dataset.Conventions, "CF-1.7")
            for number in range(1, bands + 1):
                variable = dataset[f"signal_{number}"]
                decoded = variable[:]
                # In particular, valid 255/-32767 values must not acquire the
                # implicit NetCDF fill mask when the GeoTIFF sentinel was zero.
                np.testing.assert_array_equal(np.ma.getmaskarray(decoded), missing)
                np.testing.assert_array_equal(decoded.compressed(), values[~missing])
                self.assertEqual(variable.dimensions, dimensions)
                self.assertEqual(variable.dtype, values.dtype)
                self.assertTrue(variable.filters()["zlib"])
                self.assertEqual(variable.units, "1")
                self.assertEqual(variable.standard_name, f"signal_{number}")
                self.assertIn(variable.grid_mapping, dataset.variables)
                if np.isnan(nodata):
                    self.assertTrue(np.isnan(variable._FillValue))
                else:
                    self.assertEqual(variable._FillValue, nodata)
                variable.set_auto_mask(False)
                np.testing.assert_array_equal(variable[:], values)

    def test_zero_nodata_does_not_mask_valid_default_fill_values(self):
        for epsg in (4326, 32618):
            for dtype, valid in (("uint8", 255), ("int16", -32767), ("float32", 123.5)):
                with self.subTest(epsg=epsg, dtype=dtype):
                    values = np.array([[0, valid, 1], [2, 0, 4]], dtype=dtype)
                    self.create_geotiff(values, epsg, 0)
                    self.convert()
                    self.check_output(values, epsg, 0)

    def test_all_nodata_and_no_nodata_pixels_retain_zero_sentinel(self):
        for epsg in (4326, 32618):
            for value in (0, 255):
                with self.subTest(epsg=epsg, value=value):
                    values = np.full((2, 3), value, dtype="uint8")
                    self.create_geotiff(values, epsg, 0)
                    self.convert()
                    self.check_output(values, epsg, 0)

    def test_rotated_geotiffs_preserve_the_same_mask_and_raw_values(self):
        for epsg in (4326, 32618):
            with self.subTest(epsg=epsg):
                values = np.array([[0, 255], [1, 0]], dtype="uint8")
                self.create_geotiff(values, epsg, 0, rotated=True)
                self.convert()
                self.check_output(values, epsg, 0)

    def test_every_band_preserves_zero_nodata(self):
        for epsg in (4326, 32618):
            with self.subTest(epsg=epsg):
                values = np.array([[0, 2, 255]], dtype="uint8")
                self.create_geotiff(values, epsg, 0, bands=3)
                self.convert()
                self.check_output(values, epsg, 0, bands=3)

    def test_nonzero_and_nan_nodata_keep_existing_behavior(self):
        for epsg in (4326, 32618):
            for nodata in (-9999.0, float("nan")):
                with self.subTest(epsg=epsg, nodata=nodata):
                    values = np.array([[nodata, 0, 1], [2, nodata, 4]], dtype="float32")
                    self.create_geotiff(values, epsg, nodata)
                    self.convert()
                    self.check_output(values, epsg, nodata)

    def test_absent_nodata_does_not_create_an_explicit_fill_attribute(self):
        for epsg in (4326, 32618):
            with self.subTest(epsg=epsg):
                values = np.array([[0, 1, 2], [3, 4, 0]], dtype="uint8")
                self.create_geotiff(values, epsg, None)
                self.convert()
                with Dataset(self.output) as dataset:
                    variable = dataset["signal_1"]
                    self.assertNotIn("_FillValue", variable.ncattrs())
                    np.testing.assert_array_equal(variable[:], values)
                    self.assertFalse(np.ma.getmaskarray(variable[:]).any())
