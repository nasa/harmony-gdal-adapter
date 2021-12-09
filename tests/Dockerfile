From osgeo/gdal:ubuntu-full-3.1.2

RUN ln -sf /usr/bin/python3 /usr/bin/python \
  && apt-get update \
  && apt-get install -y python3-pip \
  && pip3 install boto3 \
  && pip3 install awscli
