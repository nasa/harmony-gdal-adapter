[version 1.0.13] 2021-03-16
This version does not define nodata if the original image does not define the nodata. If the geotiff gile doe not define the nodata, the output netcdf file also does not define the nodata. but the default filling_value to be used to fill the data part of the variable before the geotiff data is written to the variable. For geotiff/netcdf without nodata definaition inside, the GIS software should use the mask data which has the same name as the data to correctly display the image. 

[version 1.0.12] 2021-03-15
This version is an experimental version. For the original image with byte type data, if it does not define the nodata, the valid data range is 0-255. We map the range of 0-255 to 0-254, and define the nodata value=255. This way the output image get well display in the GIS software.
 
[version 1.0.11] 2021-03-02
This version imporved the metedata in the output file.

[version 1.0.10] 2021-02-24
This version marks the subset with either filling of nodata value directly to data bands or creating the mask bands which are associated with the data bands. It also implements the reproject/resize with optional input of resampling method from users. The available resampling methods are the same as those defined in the gdal library. They are nearest, bilinear, cubic, cubicspline, lanczos, average, rms, and mode.

[version 1.0.9] 2021-02-10
This version changes the authentication in the version 1.0.8 to new access-token authentication.

[version 1.0.8] 2021-02-03
modify transform.py to adopt to harmoy with STAC-catalog.

[version 1.0.7] 2021-01-15
add the geotiff to netcdf conversion functuionality in the gdal-subsetter. Users may request the harmony to output netcdf file if they provide the parameter &format=application/x-netcdf4 in the request url.

[version 1.0.6]  2020-12-16
updated: addition of the functionality of subsetting with shapefile. The shapefile can be point,line, and polygon feature shapefiles. It can be in three format: geojsion, ESRI shapefile, and KML file. 

[version 1.0.5]  2020-12-03
modify the gdal-subsetter code to work with new harmony-service-library-py (2020/11/12), and new harmony ()  

[version 1.0.2]
updated: improve subset for rotated image. set values as 0 of these pixels outside the user's box.

[version 1.0.1]

[version 1.0.0]
