"""This module contains custom exceptions specific to the Harmony GDAL Adapter
    service. These exceptions are intended to allow for clearer messages to the
    end-user and easier debugging of expected errors that arise during an
    invocation of the service.

"""
# noqa: D107

from harmony.exceptions import HarmonyException


class HGAException(HarmonyException):
    """Base class for exceptions in the Harmony GDAL Adapter."""

    def __init__(self, message):
        super().__init__(message, 'nasa/harmony-gdal-adapter')


class DownloadError(HGAException):
    """ This exception is raised when the Harmony GDAL Adapter cannot retrieve
        input data.

    """
    def __init__(self, url, message):
        super().__init__(f'Could not download resource: {url}, {message}')


class UnknownFileFormatError(HGAException):
    """ This is raised when the input file format is one that cannot by
        processed by the Harmony GDAL Adapter.

    """
    def __init__(self, file_format):
        super().__init__('Cannot process unrecognised file format: '
                         f'"{file_format}"')


class IncompatibleVariablesError(HGAException):
    """This exception is raised when the dataset variables requested are not
    compatible, i.e. they have different projections, geotransforms, sizes or
    data types.

    """
    def __init__(self, message):
        super().__init__(f'Incompatible variables: {message}')
