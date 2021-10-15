gdalsubsetter="/path/to/asf-harmony-gdal"

docker run --rm -it --entrypoint /bin/bash  -v ${gdalsubsetter}:/home asfdataservices/gdal-subsetter
