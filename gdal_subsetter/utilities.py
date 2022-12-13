""" A module to contain helper functions offering utility functionality for
    the Harmony GDAL Adapter Service.

"""
from glob import glob
from os import rename
from os.path import dirname, exists, join as path_join, splitext
from typing import List
import re

from harmony.util import generate_output_filename
from osgeo import gdal

known_file_types = {'.nc': 'nc', '.nc4': 'nc', '.tif': 'tif', '.tiff': 'tif',
                    '.zip': 'zip'}

mime_to_gdal = {'image/tiff': 'GTiff',
                'image/png': 'PNG',
                'image/gif': 'GIF',
                'application/x-netcdf4': 'NETCDF',
                'application/x-zarr': 'zarr'}

mime_to_extension = {'image/tiff': 'tif',
                     'image/png': 'png',
                     'image/gif': 'gif',
                     'application/x-netcdf4': 'nc',
                     'application/x-zarr': 'nc'}

process_flags = {'subset': False,
                 'maskband': False}

resampling_methods = ['nearest', 'bilinear', 'cubic', 'cubicspline', 'lanczos',
                      'average', 'rms', 'mode']


def get_file_type(input_file_name: str) -> str:
    """ Determine the input file type according to its extension. If the
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
    """ Retrieve the version of the HGA Docker image from version.txt. """
    with open('version.txt', mode='r', encoding='utf-8') as file_version:
        version = ','.join(file_version.readlines())

    return version


def get_files_from_unzipfiles(extract_dir: str, filetype: str,
                              variables=None) -> List[str]:
    """
    inputs: extract_dir which include geotiff files, filetype is
    either 'tif' or 'nc', variables is the list of variable names.
    return: filelist for variables.
    """
    tmpexp = path_join(extract_dir, f'*.{filetype}')
    filelist = sorted(glob(tmpexp))
    ch_filelist = []
    if filelist:
        if variables:
            if 'Band' not in variables[0]:
                for variable in variables:
                    variable_tmp = variable.replace('-', '_')
                    variable_raw =fr'{variable_tmp}'
                    for filename in filelist:
                        if re.search(variable_raw, filename.replace('-', '_')):
                            ch_filelist.append(filename)
                            break
            else:
                ch_filelist = filelist
        else:
            ch_filelist = filelist
    return ch_filelist


def rename_file(input_filename: str, stac_asset_href: str) -> str:
    """ Rename a given file to a name determined for the input STAC Asset
        by the harmony-service-lib Python library.

        TODO: `generate_output_filename` should be called with appropriate
              values for `variable_subset`, `is_regridded` and `is_subsetted`.
              These kwargs allow the function to determine any required
              suffices for the file, e.g., `<input_file>_subsetted.nc4`.

    """
    output_filename = path_join(dirname(input_filename),
                                generate_output_filename(stac_asset_href))
    rename(input_filename, output_filename)
    return output_filename


class OpenGDAL:
    """ A class that allows invocation of `osgeo.gdal.Open` via a context
        manager. When the context manager exits, the file opened with GDAL
        will be closed.

        Usage:

        ```
        with OpenGDAL('file.tiff') as input_file:
            gdal_metadata = input_file.GetMetadata()
        ```

    """
    def __init__(self, file_path: str, *gdal_args):
        """ Save `osgeo.gdal.Open` arguments as class attributes for use in
            `self.__enter__`.

        """
        self.file_path = file_path
        self.gdal_args = gdal_args
        self.gdal_object = None

    def __enter__(self):
        """ Return a file opened with `osgeo.gdal.Open`. """
        self.gdal_object = gdal.Open(self.file_path, *self.gdal_args)
        return self.gdal_object

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Close the file opened via `osgeo.gdal.Open`, if is still open. """
        if self.gdal_object:
            del self.gdal_object
