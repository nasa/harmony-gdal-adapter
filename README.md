# harmony-gdal

![](https://data-services-github-badges.s3.amazonaws.com/cov.svg#dummy)

A demonstration of a subsetter capability to be used with Harmomy. Deployed on [dockerhub](https://hub.docker.com/repository/docker/asfdataservices/gdal-subsetter/general)

## Installing dependencies - make sure to clone harmony-service-lib-py locally
pip3 install ../harmony-service-lib-py/ --target deps/harmony

## Build image
bin/build-image

## Deploy image
bin/push-image
