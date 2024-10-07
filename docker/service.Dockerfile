###############################################################################
#
# Image for the ghcr.io/nasa/harmony-gdal-adapter service. This image
# will be executed as a container, and will run a Harmony request.
# The ordering of Docker commands is optimised to prevent re-running the more
# time-consuming package installation when only service code is updated.
#
# 2022-01-27: Dockerfile created.
# 2024-10-04: Updated to Miniconda base image, and install GDAL via conda.
#
###############################################################################
FROM continuumio/miniconda3

WORKDIR "/home"

# Create conda environment
RUN conda create -y --name hga python=3.11 --channel conda-forge -q -y && conda clean -a

# Install GDAL
RUN conda run --name hga conda install gdal=3.6.2

# Copy service requirements file into image.
COPY requirements.txt .

# Install Python dependencies.
RUN conda run --name hga pip install --no-input --no-cache-dir -r requirements.txt

# Copy service code into image
COPY gdal_subsetter gdal_subsetter

# Copy version file into image - for use in logging
COPY version.txt .

# Set conda environment for HGA, as `conda run` will not stream logging.
# Setting these environment variables is the equivalent of `conda activate`.
ENV _CE_CONDA='' \
    _CE_M='' \
    CONDA_DEFAULT_ENV=hga \
    CONDA_EXE=/opt/conda/bin/conda \
    CONDA_PREFIX=/opt/conda/envs/hga \
    CONDA_PREFIX_1=/opt/conda \
    CONDA_PROMPT_MODIFIER=(hga) \
    CONDA_PYTHON_EXE=/opt/conda/bin/python \
    CONDA_ROOT=/opt/conda \
    CONDA_SHLVL=2 \
    PATH="/opt/conda/envs/hga/bin:${PATH}" \
    SHLVL=1

# Set GDAL related environment variables.
ENV CPL_ZIP_ENCODING=UTF-8 \
    GDAL_DATA=/opt/conda/envs/hga/share/gdal \
    GSETTINGS_SCHEMA_DIR=/opt/conda/envs/hga/share/glib-2.0/schemas \
    GSETTINGS_SCHEMA_DIR_CONDA_BACKUP='' \
    PROJ_LIB=/opt/conda/envs/hga/share/proj

# Set entrypoint to invoke service
ENTRYPOINT ["python3", "-m", "gdal_subsetter"]
