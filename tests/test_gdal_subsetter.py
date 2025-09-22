"""A module for end-to-end tests of the Harmony GDAL Adapter."""

from datetime import datetime
from os.path import exists, join as path_join
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch

from harmony_service_lib.message import Message
from harmony_service_lib.util import config as harmony_config
from pystac import Asset, Catalog, Item

from gdal_subsetter.exceptions import DownloadError, UnsupportedFileFormatError
from gdal_subsetter.transform import HarmonyAdapter


class TestEndToEnd(TestCase):
    @classmethod
    def setUpClass(cls):
        """Set test fixtures that can be shared between tests."""
        cls.config = harmony_config(validate=False)

    def setUp(self):
        """Create test fixtures that must be unique to each test."""
        self.temp_dir = mkdtemp()

    def tearDown(self):
        """Remove test fixtures that are unique to each test."""
        if exists(self.temp_dir):
            rmtree(self.temp_dir)

    @patch("gdal_subsetter.transform.download")
    def test_failed_download(self, mock_download):
        """Ensure a failed download raises a service exception."""
        raw_exception_message = "Download exception message"
        mock_download.side_effect = Exception(raw_exception_message)

        stac_catalog = Catalog("input catalog", "description")
        stac_item = Item("input", {}, [0, 0, 1, 1], datetime(2020, 1, 1), {})
        stac_item.add_asset("data", Asset("url.com/file.nc4", roles=["data"]))
        stac_catalog.add_item(stac_item)

        harmony_message = Message(
            {
                "accessToken": "fake-token",
                "callback": "https://example.com",
                "format": {"mime": "image/png"},
                "sources": [{"collection": "C1234-XYZ"}],
                "stagingLocation": "s3://example-bucket",
                "user": "ascientist",
            }
        )

        adapter = HarmonyAdapter(
            harmony_message, catalog=stac_catalog, config=self.config
        )

        with self.assertRaisesRegex(
            DownloadError,
            f"Could not download resource: url.com/file.nc4, {raw_exception_message}",
        ):
            adapter.invoke()

    @patch("gdal_subsetter.transform.download")
    def test_unsupported_file_format(self, mock_download):
        """An unsupported input file extension raises a service exception."""
        local_file = path_join(self.temp_dir, "file.xyz")
        Path(local_file).touch()
        mock_download.return_value = local_file

        stac_catalog = Catalog("input catalog", "description")
        stac_item = Item("input", {}, [0, 0, 1, 1], datetime(2020, 1, 1), {})
        stac_item.add_asset("data", Asset("url.com/file.xyz", roles=["data"]))
        stac_catalog.add_item(stac_item)

        harmony_message = Message(
            {
                "accessToken": "fake-token",
                "callback": "https://example.com",
                "format": {"mime": "image/png"},
                "sources": [{"collection": "C1234-XYZ"}],
                "stagingLocation": "s3://example-bucket",
                "user": "ascientist",
            }
        )

        adapter = HarmonyAdapter(
            harmony_message, catalog=stac_catalog, config=self.config
        )

        with self.assertRaisesRegex(
            UnsupportedFileFormatError,
            'Cannot process unsupported file format: ".xyz"',
        ):
            adapter.invoke()
