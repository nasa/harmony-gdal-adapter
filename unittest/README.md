The unittest does test according to the message json files in `data/prod`.

## Setup

if you run the container version unittest in a local environment, you need connect to ASF VPN-Full.

### Environment

Copy `env.unittest` to `.env` and set `EDL_USERNAME` and `EDL_PASSWORD`

```
  cp env.example .env
```

### Build

build the docker iamge asfdataservices/gdal-subsetter on the host

```
cd gdal-subsetter
bin/build_iamge
```

### Run

from host

```
./run-interactive.bash
```

This brings you in to the docker container.

Then from insdier the container,

```
cd /home/unittest
pytest

```
