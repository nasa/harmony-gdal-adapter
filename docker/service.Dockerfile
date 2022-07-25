###############################################################################
#
# Image for the harmonyservices/harmony-gdal-adapter service. This image
# will be executed as a container, and will run a Harmony request.
# The ordering of Docker commands is optimised to prevent re-running the more
# time-consuming package installation when only service code is updated.
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

# Copy service requirements file into image.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install -r requirements.txt

# Copy service code into image
COPY gdal_subsetter gdal_subsetter

# Copy version file into image - for use in logging
COPY version.txt .

# Set entrypoint to invoke service
ENTRYPOINT ["python3", "-m", "gdal_subsetter"]
