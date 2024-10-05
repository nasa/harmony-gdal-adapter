###############################################################################
#
# Test image for the harmonyservices/harmony-gdal-adapter service. This image
# will run a Python `pytest` suite. The `run_tests.sh` script also contains
# a `pylint` check on the code contained in the `gdal_subsetter` directory.
#
# 2022-01-27: Dockerfile created.
# 2024-10-04: Updated to the service image as a base.
#
###############################################################################
FROM ghcr.io/nasa/harmony-gdal-adapter

# Install test requirements in hga conda environment.
COPY requirements_dev.txt /home/
RUN conda run --name hga pip install -r requirements_dev.txt

# Copy test directory into image
COPY tests tests

# Set conda environment to maskfill, as `conda run` will not stream logging.
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

# GDAL specific environment variables
ENV CPL_ZIP_ENCODING=UTF-8 \
    GDAL_DATA=/opt/conda/envs/hga/share/gdal \
    GSETTINGS_SCHEMA_DIR=/opt/conda/envs/hga/share/glib-2.0/schemas \
    GSETTINGS_SCHEMA_DIR_CONDA_BACKUP='' \
    PROJ_LIB=/opt/conda/envs/hga/share/proj

# Set entrypoint to run test script.
ENTRYPOINT ["/home/tests/run_tests.sh"]
