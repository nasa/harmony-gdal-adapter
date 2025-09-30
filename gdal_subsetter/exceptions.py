"""Contains custom exceptions specific to the Harmony GDAL Adapter service.

These exceptions are intended to allow for clearer messages to the
end-user and easier debugging of expected errors that arise during an
invocation of the service.

"""
# noqa: D107

from harmony_service_lib.exceptions import HarmonyException, NoRetryException


class HGANoRetryException(NoRetryException):
    """Base class for exceptions in the Harmony GDAL Adapter.

    This exception is inherited by errors that should not cause Harmony to
    retry the HGA step in a workflow chain, because it is known they will fail
    upon that retry.

    """

    def __init__(self, message=None):
        super().__init__(message)


class DownloadError(HarmonyException):
    """Raised when the Harmony GDAL Adapter cannot retrieve input data.

    This does not inherit from NoRetryException, as it may be due to intermittent
    network issues.

    """

    def __init__(self, url, message):
        super().__init__(f"Could not download resource: {url}, {message}")


class UnsupportedFileFormatError(HGANoRetryException):
    """Raised when the input file format is cannot processed by the HGA."""

    def __init__(self, file_format):
        super().__init__(f'Cannot process unsupported file format: "{file_format}"')


class IncompatibleVariablesError(HGANoRetryException):
    """Raised when the dataset variables requested are not compatible.

    i.e. they have different projections, geotransforms, sizes or data types.

    """

    def __init__(self, message):
        super().__init__(f"Incompatible variables: {message}")


class MissingVariableError(HGANoRetryException):
    """Raised when a requested variable is absent from the input GeoTIFF."""

    def __init__(self, variable_name):
        super().__init__(f"Missing variable in input file: {variable_name}")
