""" A module to contain helper functions offering utility functionality for
    the Harmony GDAL Adapter Service.

"""
from os.path import exists as file_exists, splitext


known_file_types = {'.nc': 'nc', '.nc4': 'nc', '.tif': 'tif', '.tiff': 'tif',
                    '.zip': 'zip'}


def get_file_type(input_file_name: str) -> str:
    """ Determine the input file type according to its extension. If the
        format is not recognised the extension is returned so it can be
        provided in a raised exception and the message returned to the
        end-user.

    """
    if input_file_name is None or not file_exists(input_file_name):
        file_type = None
    else:
        file_extension = splitext(input_file_name)[-1]
        file_type = known_file_types.get(file_extension, file_extension)

    return file_type
