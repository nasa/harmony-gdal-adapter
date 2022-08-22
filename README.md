# Harmony GDAL Adapter (HGA)

![](https://data-services-github-badges.s3.amazonaws.com/cov.svg?dummy=true)

A Harmony (https://harmony.earthdata.nasa.gov/) backend service that transforms
input images using GDAL.

HGA is published to [ghcr.io](https://github.com/nasa/harmony-gdal-adapter/pkgs/container/harmony-gdal-adapter)
GitHub's Container registry.

HGA is invoked by [harmony](https://github.com/nasa/harmony)
when the harmony server is configured, via Harmony's [service.yml](https://github.com/nasa/harmony/blob/main/config/services.yml)
or by UMM-S/C associations in CMR, to handle an incoming request for the
collection. You can see examples of requests that harmony dispatches to the
harmony-gdal-adapter by examining the [regression test notebook for hga](https://github.com/nasa/harmony-regression-tests/blob/main/test/hga/HGA_regression.ipynb).


## Test with Docker

### Build HGA image
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
The `run-test` script mounts `test-reports` and `coverage` directories and run
the test script `tests/run_tests.sh` inside of a Docker test container.


## Test Locally

### Create isolated environment

```bash
conda create --name hga python=3.8 --channel conda-forge
conda activate hga
```

### Install requirements

```bash
conda install gdal==3.4.3
pip install -r requirements.txt -r requirements_dev.txt
```

### Run Tests

```bash
./tests/run_tests.sh
```
This script runs pytest on the `./tests` directory.

## Debugging

It is possible to debug this service for development by attaching a debugger
that follows the
[debugger-adapter-protocol](https://microsoft.github.io/debug-adapter-protocol/)
to a harmony stack running in a local kubernetes cluster.  These instructions
are for developers of this service in order to help them understand the code.

Basic steps for debugging are:

1. Add [debugpy](https://github.com/microsoft/debugpy) to `requirements.txt` file and reubild this image `./bin/build-image`.
     - add debugpy to the `requirements.txt` file:

     -  requirements.txt diff:
          ```diff
          +debugpy==1.6.3
          ```

1. Edit your harmony `.env` file to use
 `debugpy` and relaunch harmony services
 to enable this change.  The default invocation args are for this service are
 `python -m gdal_subsetter` and you must change the default params to run
 through debugpy listening on all interfaces at port 5678.

    ```sh
    HARMONY_GDAL_ADAPTER_INVOCATION_ARGS='python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m gdal_subsetter'
    ```
1. Determine the name of your service pod in kubernetes, finding the one that
   is named like `harmony-gdal-adapter-58b6f98b57-sv5vm` with different
   trailing hashes.

     ```sh
     kubectl get pods -n harmony
     ```
1. Open a port from your local machine to the kubernetes pod, subsituting your
   pod's name. This allows your local debugger to attach to the running process
   on the pod.

     ```sh
     kubectl port-forward harmony-gdal-adapter-58b6f98b57-sv5vm -n harmony 5678:5678
     ```

1. Submit a harmony client command that will trigger this service.
    - The first time after a restart of the harmony services, you might not
     have to submit a command because harmony submits a fake request to prime
     the system and that priming request should be waiting for a debugger to
     attach.

1. Attach your debugger using a `launch.json` file like this one

     ```json
        {
          "name": "Harmony GDAL Adapter Attach",
          "type": "python",
          "request": "attach",
          "connect": {
            "host": "localhost",
            "port": 5678
          },
          "pathMappings": [
            {
            "localRoot": "${workspaceFolder}",
            "remoteRoot": "/home/"
            }
          ]
        }
     ```






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
`version.txt` and `CHANGE.md`. Every time code is merged to the `main` branch
and the merged commits contain changes to `version.txt`, a Docker image is
published to [ghcr.io](https://github.com/nasa/harmony-gdal-adapter/pkgs/container/harmony-gdal-adapter).
By only triggering image publication when `version.txt` is incremented, the
existing Docker images for HGA will not be overwritten.

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
