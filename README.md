# harmony-gdal-adapter

![](https://data-services-github-badges.s3.amazonaws.com/cov.svg?dummy=true)

A Harmony (https://harmony.earthdata.nasa.gov/) backend service that transforms input images using GDAL.

The harmony-gdal-adapter is deployed to [ghcr.io](https://github.com/nasa/harmony-gdal-adapter/pkgs/container/harmony-gdal-adapter) GitHub's Container registry.


## Run with Docker
### Build image
```bash
bin/build-image
```

### Build Tests
```bash
bin/build-test
```

### Run Tests
```bash
bin/run-test
```

## Run Locally

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
