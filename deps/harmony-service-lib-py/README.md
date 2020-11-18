# harmony-service-lib

A library for Python-based Harmony services to parse incoming messages, fetch data, stage data, and call back to Harmony

## Installing

### Using pip

Install the latest version of the package from the Earthdata Nexus repo using pip:

    $ pip install --extra-index-url https://maven.earthdata.nasa.gov/repository/python-repo/simple/ harmony-service-lib

Or a specific version:

    $ pip install --extra-index-url https://maven.earthdata.nasa.gov/repository/python-repo/simple/ harmony-service-lib==0.0.7

In a requirements.txt file:

```
--extra-index-url https://maven.earthdata.nasa.gov/repository/python-repo/simple/
harmony-service-lib~=0.0.9
```

### Using conda

In a conda environment, you may still use pip to install the harmony service lib, as shown above.
However, in a conda environment.yml file, in addition to the conda dependencies, you can specify
pip dependencies as shown here:

```
dependencies:
  - ...
  - pip:
    - --extra-index-url https://maven.earthdata.nasa.gov/repository/python-repo/simple/
    - harmony-service-lib~=0.0.9
    - ...
```

### Other methods:

The package is installable from source via

    $ pip install git+https://git.earthdata.nasa.gov/scm/harmony/harmony-service-lib-py.git

If using a local source tree, run the following in the source root directory instead:

    $ pip install .

## Usage

Services that want to work with Harmony can make use of this library to ease
interop and upgrades.  To work with Harmony, services must:

1. Receive incoming messages from Harmony.  Currently the CLI is the only
supported way to receive messages, though HTTP is likely to follow.  `harmony.cli`
provides helpers for setting up CLI parsing while being unobtrusive to non-Harmony
CLIs that may also need to exist.
2. Extend `harmony.BaseHarmonyAdapter` and implement the `#invoke` to
adapt the incoming Harmony message to a service call and adapt the service
result to call to one of the adapter's `#completed_with_*` methods. The adapter
class provides helper methods for retrieving data, staging results, and cleaning
up temporary files, though these can be overridden or ignored if a service
needs different behavior, e.g. if it operates on data in situ and does not
want to download the remote file.

A full example of these two requirements with use of helpers can be found in
[example/example_service.py](example/example_service.py)

## Environment

The following environment variables can be used to control the behavior of the
library and allow easier testing:

* `STAGING_BUCKET`: When using helpers to stage service output and pre-sign URLs, this
  indicates the S3 bucket where data will be staged
* `STAGING_PATH`: When using helpers to stage output, this indicates the path within
  `STAGING_BUCKET` under which data will be staged
* `AWS_DEFAULT_REGION`: (Default: `"us-west-2"`) The region in which S3 calls will be made
* `ENV`: The name of the environment.  If 'dev' or 'test', callbacks to Harmony are
       not made and data is not staged unless also using localstack
* `USE_LOCALSTACK`: (Development) If 'true' will perform S3 calls against localstack rather
       than AWS
* `LOCALSTACK_HOST`: (Development) If `USE_LOCALSTACK` `true` and this is set, will
       establish `boto` client connections for S3 & SQS operations using this hostname.
* `EDL_USERNAME`, `EDL_PASSWORD`: (Better solution on the roadmap)  If using helpers to
       fetch granules over HTTPS and those granules are behind Earthdata Login, these
       variables are the credentials that will be used to authenticate to Earthdata Login
       to fetch the data.  Be sure that the user has accepted any relevant EULAs and
       has permission to get the data.  Services using data in S3 do not need to set this.
       A future release will do away with these variables by having the Harmony frontend
       provide authentication information.
* `TEXT_LOGGER`: Setting this to true will cause all log messages to use a text string
       format. By default log messages will be formatted as JSON.
* `HEALTH_CHECK_PATH`: Set this to the path where the health check file should be stored. This
       file's mtime is set to the current time whenever a successful attempt is made to to read the
       message queue (whether or not a message is retrieved). This file can be used by a container's
       health check command. The container is considered unhealthy if the mtime of the file is old -
       where 'old' is configurable in the service container. If this variable is not set the path 
       defaults to '/tmp/health.txt'.

## Development Setup

Prerequisites:
  - Python 3.7+, ideally installed via a virtual environment such as `pyenv`
  - A local copy of the code

Install dependencies:

    $ make develop

Run tests:

    $ make test

Build & publish the package:

    $ make publish

## Releasing

Update the CHANGELOG with a short bulleted description of the changes to be
built & deployed. Replace `VERSION` and `DATE` with the version being built
by the Bamboo job and the date on which the build is run.

TODO NOTE: There is currently a possible disconnect between the version and 
date entries in the CHANGELOG and the version generated by Bamboo. We need to
resolve this so that a release version and date is either set or determined
in a consistent way, including version tags in git that are pushed to 
BitBucket.

New entries to the CHANGELOG should be of the form:

```
## [VERSION] - DATE

Changes:

* Now uses a flux capacitor in the hydro-flange motivator circuit
* Tastes better & is less filling than previous release
```

The [Harmony Python Service Library Bamboo Build](https://ci.earthdata.nasa.gov/browse/HARMONY-PSL)
will be triggered on commits pushed to the
[BitBucket repo](https://git.earthdata.nasa.gov/projects/HARMONY/repos/harmony-service-lib-py/browse).
New versions of the Python package artifact will then be pushed to the [Earthdata Nexus Repository](https://ci.earthdata.nasa.gov/browse/HARMONY-PSL). It may then be installed using `pip`.
