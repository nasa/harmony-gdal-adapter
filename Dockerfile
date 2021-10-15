FROM osgeo/gdal:ubuntu-full-3.3.1

RUN ln -sf /usr/bin/python3 /usr/bin/python && apt-get update && apt-get install -y python3-pip nco && pip3 --no-cache-dir install boto3

WORKDIR "/home"

# Bundle app source
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt

# To run locally during dev, build the image and run, e.g.:
# docker run --rm -it -e ENV=dev -v $(pwd):/home harmony/gdal --harmony-action invoke --harmony-input "$(cat ../harmony/example/service-operation.json)"
ENTRYPOINT ["python3", "-m", "gdal_subsetter"]
