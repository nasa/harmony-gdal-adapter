gdalsubsetter="/home/jzhu4/projects/work/harmony-curr/gdal-subsetter"

harmonyservicelibpy="/home/jzhu4/projects/work/harmony-curr/harmony-service-lib-py"

docker run --entrypoint /home/unittest/test_whole_messages.bash -v ${gdalsubsetter}:/home -v ${harmonyservicelibpy}/harmony:/usr/lib/harmony-service-lib-py/harmony harmony/gdal-subsetter  
