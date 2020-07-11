FROM osgeo/gdal:ubuntu-full-latest

RUN ln -s /usr/bin/python3 /usr/bin/python && apt-get update && apt-get install -y python3-pip && pip3 install boto3

WORKDIR "/home"

# Bundle app source
COPY . .
RUN pip3 install deps/harmony-service-lib-py
RUN pip3 install -r requirements.txt

# To run locally during dev, build the image and run, e.g.:
# docker run --rm -it -e ENV=dev -v $(pwd):/home harmony/gdal --harmony-action invoke --harmony-input "$(cat ../harmony/example/service-operation.json)"
# Or if also working on harmony-service-lib-py in a peered directory:
# docker run --rm -it -e ENV=dev -v $(pwd):/home -v $(dirname $(pwd))/harmony-service-lib-py:/home/deps/harmony harmony/gdal --harmony-action invoke --harmony-input "$(cat ../harmony/example/service-operation.json)"
ENTRYPOINT ["python3", "-m", "harmony_gdal"]
