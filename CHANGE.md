# Changelog

HGA follows semantic versioning. All notable changes to this project will be
documented in this file. The format is based on [Keep a
Changelog](http://keepachangelog.com/en/1.0.0/).


## [v1.2.13] - 2024-10-04

### Changed:

* Updates base Docker image to use Miniconda, and installs GDAL via conda.
* Updates service image to be built on AMD64.

## [v1.2.12] - 2023-04-26

### Changed:

* Rename function `combin_transfer` -> `perform_transforms`
* Tidies Readme

### Removed:

* Removes unused code module `capture_operation.py`

## [v1.2.11] - 2022-12-23

### Added:

* Attempt to add more type hints and documentation strings, to enable better
  understanding of the code.

### Changed:

* Refactor `HarmonyAdapter.process_item` to move staging and STAC record
  generation to separate class method.
* Refactor `HarmonyAdapter.process_zip` and `HarmonyAdapter.unpack_zipfile` to
  remove redundant variables.
* Refactor `get_files_fromunzipfiles` to capture all NetCDF-4 and GeoTIFF
  extension options, while implementing list comprehensions for increased
  readability.

## [v1.2.10] - 2022-12-16

### Added:

* Increased unit test coverage.

### Changed:

* Migrate `calc_subset_envelope_window` and `box_to_shapefile` out of HarmonyAdapter.

## [v1.2.9] - 2022-12-14

### Changed:

* Update base Docker image to `osgeo/gdal:ubuntu-full-3.6.0` (GDAL 3.6.0).
* Update geospatial Python packages to support GDAL upgrade.

## [v1.2.8] - 2022-12-13

### Changed:

* Implement `OpenGDAL` context manager for `osgeo.gdal.Open`.

## [v1.2.7] - 2022-12-10

### Changed:

* Migration of static methods from HarmonyAdapter class to utility modules.
* Implementation of Python native functions to copy, move and delete files or
  directories.
* General linting clean-up.

## [v1.2.6] - 2022-11-25

### Changed:

* Improved detection of NetCDF-4 subdatasets.
* Update harmony-service-lib-py to 1.0.22.
* General code linting clean-up.

## [v1.2.5] - 2022-08-05

### Added:

* Reimplements logging statements removed in HGA v1.2.4.

### Changed:

* Updates `harmony-service-lib` dependency to 1.0.21, to ensure correct
  propagation of error messages to Harmony core.

## [v1.2.4] - 2022-08-01

### Removed:

* Removes logging that is interfering with exception handling.

## [v1.2.3] - 2022-07-25

### Changed:

* Update GDAL to version 3.4.3

### Removed:

* Remove unused tests and code.

## [v1.2.2] - 2022-07-11

### Changed:

* Update harmony-service-library-py to 1.0.20

## [v1.2.1] - 2022-07-11

### Changed:

* Fix logging command so that correct exception is raised.

## [v1.2.0] - 2022-07-08

### Changed:

* Modify transform.py to raise an exception when the set of requested variables is
  incompatible. e.g. they have different geocoordinates, geotransforms, x or y
  dimensions, or different data types.
* Changes behavior of process_item to raise HGAException when a stac\_record is
  not created. Previously the code quietly logged a warning and succeeded.

## [v1.1.5] - 2022-06-13

### Changed:

* Update transform.py to add a default greyscale colormap to image-type tiff.
  Fixes a regression introduced in 1.1.4.
* TIFF files are no longer colored with gdaldem (as of 1.1.4), but a test has
  been added to make that expectation explicit.

## [v1.1.4] - 2022-05-20

### Changed:

* Change to GDAL subsetter to use colormap URLs found in Harmony message
  instead of hard coded ones.

## [v1.1.3] - 2022-05-09

### Changed:

* Renames some code variables and functions, there should be no changes to the
  functionality of the service.

## [v1.1.2] - 2022-04-21

### Changed:

* Updates service library dependencies.

```text
Fiona          1.8.17 => 1.8.21
geopandas       0.9.0 => 0.10.2
rasterio        1.1.5 => 1.2.10
numpy     unsepcified => 1.22.3
```

### Removed:

* Removes unused code from `tests` and `bin`

## [v1.1.1] - 2022-03-23

### Changed:

* Input files with `.nc4` extensions are now recognised as NetCDF-4 files.
* An exception is raised for unknown input file formats.
* Download error messages are propagated more transparently to the end-user in
  cases of failure.

## [v1.1.0] - 2021-12-09

### Added:

* New functionality added for being able to produce PNG output from NetCDF input.

## [v1.0.17] - 2021-05-05

This is the final version. We will stop the development for a while.

## [v1.0.16] - 2021-04-12

### Added:

* This version integrates the regridding functionality. Users can set regridding
  parameters in the url query for the harmony API. They are outputCrs, scaleSize
  (xres/yres in output coordinates), scaleExtent (xmin,ymin,xmax,ymax) in output
  coordinates, and width/height (columnes/rows) of output file. You are not
  allowed to use scaleSize and width/height in the same time.

## [v1.0.15] - 2021-04-09

### Added:

* This version extends the collection of datasets that gdal-subsetter can
  process.The new datasets are: ALOS_PALSAR_Level2.2, ALOS_PALSAR_RTC_HIGH_RES,
  and ALOS_PALSAR_RTC_LOW_RES.

## [v1.0.14] - 2021-03-30

### Changed:

* This version improves the geotiff to netcdf conversion code to make the output
  netcdf file be CF compliant. For a non-rotated image, we define two 1D
  coordinate variables. For a rotated image, we do not define the coordinate
  variables. We could define two 2D coordinate variables for the rotated image,
  but we found that the GIS software does not use the 2D coordinate variables to
  decide the pixel locations. So we do not output 2D coordinate variables in
  this version for sake of reducing the file size.

## [v1.0.13] - 2021-03-16

### Changed:

* This version does not define nodata if the original image does not define the
  nodata. If the geotiff gile doe not define the nodata, the output netcdf file
  also does not define the nodata. But the default filling_value to be used to
  fill the data part of the variable before the geotiff data is written to the
  variable. For geotiff/netcdf without nodata definaition inside, the GIS
  software should use the mask data which has the same name as the data to
  correctly display the image.

## [v1.0.12] - 2021-03-15

### Changed:

* This version is an experimental version. For the original image with byte type
  data, if it does not define the nodata, the valid data range is 0-255. We map
  the range of 0-255 to 0-254, and define the nodata value=255. This way the
  output image get well display in the GIS software.

## [v1.0.11] - 2021-03-02

### Changed:

* This version improves the metadata in the output file.

## [v1.0.10] - 2021-02-24

### Added:

* This version marks the subset with either filling of nodata value directly to
  data bands or creating the mask bands which are associated with the data bands.
* It also implements the reproject/resize with optional input of resampling method
  from users. The available resampling methods are the same as those defined in
  the gdal library. They are nearest, bilinear, cubic, cubicspline, lanczos,
  average, rms, and mode.

## [v1.0.9] - 2021-02-10

### Changed:

* This version changes the authentication in the version 1.0.8 to new
  access-token authentication.

## [v1.0.8] - 2021-02-03

### Changed:

* Modify transform.py to adopt to Harmony with STAC input.

## [v1.0.7] - 2021-01-15

### Added:

* Geotiff to netcdf conversion functionality has been added to the gdal-subsetter.
  Users may request Harmony to output netcdf file if they provide the parameter
  &format=application/x-netcdf4 in the request url.

## [v1.0.6]  2020-12-16

### Added:

* Addition of the functionality of subsetting with shapefile. The shapefile can
  be point,line, and polygon feature shapefiles. It can be in three formats:
  geojsion, ESRI shapefile, and KML file.

## [v1.0.5] - 2020-12-03

### Changed:

* The gdal-subsetter code has been modified to work with new
  harmony-service-library-py (2020/11/12), and new harmony ().

## [v1.0.2]

### Changed:

* Improve subset for rotated image. set values as 0 of these pixels outside the
  user's box.

## [v1.0.1]

## [v1.0.0]
