gdalsubsetter="/home/jzhu4/projects/work/harmony-curr/gdal-subsetter"

harmonyservicelibpy="/home/jzhu4/projects/work/harmony-curr/harmony-service-lib-py"

docker run -it --entrypoint /bin/bash  -v ${gdalsubsetter}:/home -v ${harmonyservicelibpy}/harmony:/usr/lib/harmony-service-lib-py/harmony asfdataservices/gdal-subsetter  
