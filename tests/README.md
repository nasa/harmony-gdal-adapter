The unittest does test according to the message json files in `data/prod`.

## Setup

if you run the container version unittest in a local environment, you need connect to ASF VPN-Full.

### Environment

Copy `env.example` to `.env` and set `EDL_USERNAME` and `EDL_PASSWORD`

```
  cp env.example .env
```

### Build

build the docker image asfdataservices/gdal-subsetter on the host

```
cd gdal-subsetter
bin/build_image
```

### Run

First, edit `tests/run-interactive.bash` and set `gdalsubsetter` to the absolute filepath on the host that contains the source code
Then, from host

```
cd tests
./run-interactive.bash
```

This brings you in to the docker container.

Then from inside the container,

```
pip install -r requirements_dev.txt
cd /home/tests
pytest
```
