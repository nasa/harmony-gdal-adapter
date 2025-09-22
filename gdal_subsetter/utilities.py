"""A module to contain helper functions offering utility functionality for
the Harmony GDAL Adapter Service.

"""

from glob import glob
from os import rename
from os.path import dirname, exists, join as path_join, splitext

from harmony_service_lib.util import generate_output_filename
from osgeo import gdal

known_file_types = {
    ".nc": "nc",
    ".nc4": "nc",
    ".tif": "tif",
    ".tiff": "tif",
    ".zip": "zip",
}

mime_to_gdal = {
    "image/tiff": "GTiff",
    "image/png": "PNG",
    "image/gif": "GIF",
    "application/x-netcdf4": "NETCDF",
    "application/x-zarr": "zarr",
}

mime_to_extension = {
    "image/tiff": "tif",
    "image/png": "png",
    "image/gif": "gif",
    "application/x-netcdf4": "nc",
    "application/x-zarr": "nc",
}

process_flags = {"subset": False, "maskband": False}

resampling_methods = [
    "nearest",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "average",
    "rms",
    "mode",
]


def get_file_type(input_file_name: str) -> str:
    """Determine the input file type according to its extension. If the
    format is not recognised the extension is returned so it can be
    provided in a raised exception and the message returned to the
    end-user.

    """
    if input_file_name is None or not exists(input_file_name):
        file_type = None
    else:
        file_extension = splitext(input_file_name)[-1]
        file_type = known_file_types.get(file_extension, file_extension)

    return file_type


def get_version() -> str:
    """Retrieve the version of the HGA Docker image from version.txt."""
    with open("version.txt", mode="r", encoding="utf-8") as file_version:
        version = ",".join(file_version.readlines())

    return version


def has_world_file(file_mime_type: str) -> bool:
    """Determine if the given MIME type for a transformed output is expected
    to have an accompanying ESRI world file.

    """
    return any(world_mime in file_mime_type.lower() for world_mime in ["png", "jpeg"])


def is_geotiff(file_name: str) -> bool:
    """Determine if the given file is a GeoTIFF via `gdalinfo`."""
    gdalinfo_lines = gdal.Info(file_name).splitlines()
    return gdalinfo_lines[0] == "Driver: GTiff/GeoTIFF"


def get_unzipped_geotiffs(
    extract_dir: str, variable_names: list[str] = []
) -> list[str]:
    """Get a list of GeoTIFFs unzipped from a zip-format granule.

    If a list of variables names is specified, and the first variable name does
    not include "Band", the list of extracted GeoTIFFs will be further filtered
    to only return those file names that include one of the requested variable
    names.

    """
    geotiff_file_extensions = [
        file_extension
        for file_extension, known_file_type in known_file_types.items()
        if known_file_type == "tif"
    ]

    geotiff_files = [
        unzipped_file
        for unzipped_file in glob(f"{extract_dir}/**", recursive=True)
        if unzipped_file.endswith(tuple(geotiff_file_extensions))
    ]
    geotiff_files.sort()

    if len(variable_names) > 0 and "Band" not in variable_names[0]:
        formatted_variable_names = [
            variable_name.replace("-", "_") for variable_name in variable_names
        ]

        filtered_geotiff_files = [
            file_name
            for file_name in geotiff_files
            if any(
                variable_name in file_name.replace("-", "_")
                for variable_name in formatted_variable_names
            )
        ]
    else:
        filtered_geotiff_files = geotiff_files

    return filtered_geotiff_files


def rename_file(input_filename: str, stac_asset_href: str) -> str:
    """Rename a given file to a name determined for the input STAC Asset
    by the harmony-service-lib Python library.

    This function is used to rename the input file downloaded to the
    Docker container from a randomly generated temporary file name, to one
    that pertains to the original STAC asset URL.

    """
    output_filename = path_join(
        dirname(input_filename), generate_output_filename(stac_asset_href)
    )
    rename(input_filename, output_filename)
    return output_filename


class OpenGDAL:
    """A class that allows invocation of `osgeo.gdal.Open` via a context
    manager. When the context manager exits, the file opened with GDAL
    will be closed.

    Usage:

    ```
    with OpenGDAL('file.tiff') as input_file:
        gdal_metadata = input_file.GetMetadata()
    ```

    """

    def __init__(self, file_path: str, *gdal_args):
        """Save `osgeo.gdal.Open` arguments as class attributes for use in
        `self.__enter__`.

        """
        self.file_path = file_path
        self.gdal_args = gdal_args
        self.gdal_object = None

    def __enter__(self):
        """Return a file opened with `osgeo.gdal.Open`."""
        self.gdal_object = gdal.Open(self.file_path, *self.gdal_args)
        return self.gdal_object

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the file opened via `osgeo.gdal.Open`, if is still open."""
        if self.gdal_object:
            del self.gdal_object
