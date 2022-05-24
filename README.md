# harmony-gdal-adapter (HGA)

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

## Contributions:

### Pull requests:

Contributions to HGA can be made via submitting a pull request (PR) to this
repository. Developers with contributor privileges to the Harmony team within
the NASA GitHub organisation should have the ability to create a PR directly
within this repository. Other developers will need to follow the
[fork-and-pull model](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model).

In addition to any DAAC stakeholders, please add members of the EED Data
Services team (currently: David Auty, Ken Cockerill, Owen Littlejohns and Matt
Savoie) as PR reviewers. One of these developers must approve the PR before
it is merged.

NASA GitHub contributers will have the required GitHub permissions to merge
their PRs after the peer review is complete. Please consider using the
[squash-and-merge option](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-pull-request-commits) when merging a PR to maintain a clean Git history.

The EED Data Services team will merge approved PRs for developers without write
access to the HGA repository.

### Semantic versioning:

When making changes to the HGA service code, it is important to make updates to
`version.txt` and `CHANGE.md`. Every time code is merged to the `main` branch,
a Docker image is published to [ghcr.io](https://github.com/nasa/harmony-gdal-adapter/pkgs/container/harmony-gdal-adapter). The semantic version number listed in `version.txt` must
be iterated to avoid overwriting an existing Docker image.

When writing or updating service code, please also update the existing test
suite with unit tests to ensure the new functionality performs as expected, and
continues to do so following subsequent development.

## Deployment of new HGA versions:

After a new Docker image has been published, it will need to be deployed as
part of the Harmony Kubernetes cluster. The EED Data Services team will
coordinate with Harmony to ensure HGA is updated. If there are specific
deployment requirements, such as test data only being available in UAT or time
constraints, please communicate these to the Data Services team.

Initial deployment will be to test environments (SIT, then UAT), to allow
changes to be tested by all DAACs that are using HGA with their data
collections.

It is possible to determine the version of HGA that is deployed to a given
Harmony environment via the `/versions` endpoint, e.g.:
<https://harmony.earthdata.nasa.gov/versions>.
