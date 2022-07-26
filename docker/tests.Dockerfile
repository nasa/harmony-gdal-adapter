###############################################################################
#
# Test image for the harmonyservices/harmony-gdal-adapter service. This image
# will run a Python `pytest` suite. The `run_tests.sh` script also contains
# a `pylint` check on the code contained in the `gdal_subsetter` directory.
#
# 2022-01-27: Dockerfile created.
#
###############################################################################
FROM osgeo/gdal:ubuntu-full-3.4.3

WORKDIR "/home"

# Ensure Python is installed, add Pip for Python 3.
RUN ln -sf /usr/bin/python3 /usr/bin/python \
	&& apt-get update \
	&& apt-get install -y python3-pip

# Copy both service and testing requirements files into image.
COPY requirements*.txt /home/

# Ensure Python is installed, add Pip for Python3, install dependencies.
RUN pip install -r requirements.txt -r requirements_dev.txt

# Copy service code into image
COPY gdal_subsetter gdal_subsetter

# Copy version file into image - for use in logging
COPY version.txt .

# Copy test directory into image
COPY tests tests

# Set entrypoint to run test script.
ENTRYPOINT ["/home/tests/run_tests.sh"]
