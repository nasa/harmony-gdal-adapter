# harmony-gdal-adapter

![](https://data-services-github-badges.s3.amazonaws.com/cov.svg?dummy=true)

A Harmony (https://harmony.earthdata.nasa.gov/) backend service that transforms input images using GDAL.

The harmony-gdal-adapter is deployed to [ghcr.io](https://github.com/nasa/harmony-gdal-adapter/pkgs/container/harmony-gdal-adapter) GitHub's Container registry.

The harmony-gdal-adapter is invoked by [harmony](https://github.com/nasa/harmony) when the harmony server is configured, via harmony's [service.yml](https://github.com/nasa/harmony/blob/main/config/services.yml) or by UMM-S/C associations in CMR, to handle an incoming request for the collection. You can see examples of requests that harmony dispatches to the harmony-gdal-adapter by examining the [regression test notebook for hga](https://github.com/nasa/harmony-regression-tests/blob/main/test/hga/HGA_regression.ipynb).


## Test with Docker

### Build harmony-gdal-adapter image
```bash
bin/build-image
```
Creates the image `ghcr.io/nasa/harmony-gdal-adapter`.

### Build Tests
```bash
bin/build-test
```
Creates the `nasa/harmony-gdal-adapter-test` test image.

### Run Tests
```bash
bin/run-test
```
The `run-test` script mounts `test-reports` and `coverage` directories and run the test script `tests/run_tests.sh` inside of a docker test container.


## Test Locally

### Create isolated environment

```bash
conda create --name hga python=3.8 --channel conda-forge
conda activate hga
```

### Install requirements

```bash
conda install gdal==3.4.2
pip install -r requirements.txt -r requirements_dev.txt
```

### Run Tests

```bash
./tests/run_tests.sh
```
This script runs pytest on the `./tests` directory.
