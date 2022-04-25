#!/bin/bash

# Exit status used to report back to caller
#
STATUS=0

# Directory containing this script, used to ensure artefacts are placed in the
# mounted volumes.
#
TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

export HDF5_DISABLE_VERSION_CHECK=1


# Run the standard set of unit tests, producing JUnit compatible output
#
pytest ./tests -s --cov=gdal_subsetter --junitxml=${TEST_DIR}/reports/hga_junit.xml --cov-report=html:${TEST_DIR}/coverage

RESULT=$?
if [ "$RESULT" -ne "0" ]; then
    STATUS=1
    echo "ERROR: pytest generated errors"
fi

# Run pylint (uncomment block below to enable, or use a pylint/flake8 plugin
# for pytest)
#

# pylint gdal_subsetter --disable=E0401 --extension-pkg-whitelist=netCDF4
# RESULT=$?
# RESULT=$((3 & $RESULT))
# if [ "$RESULT" -ne "0" ]; then
#     STATUS=1
#     echo "ERROR: pylint generated errors"
# fi
echo "pylint check currently disabled - lots of linting errors need fixing."

exit $STATUS
