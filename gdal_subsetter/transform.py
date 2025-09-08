""" CLI for adapting a Harmony operation to GDAL. """
from datetime import datetime
from shutil import copyfile, rmtree
from subprocess import check_output
from tempfile import mkdtemp
from typing import List
from zipfile import ZipFile
import os
import re

from harmony.adapter import BaseHarmonyAdapter
from harmony.message import Source as HarmonySource, Variable as HarmonyVariable
from harmony.util import (bbox_to_geometry, download, generate_output_filename,
                          stage)
from netCDF4 import Dataset
from numpy.ma import masked_array
from osgeo import gdal, osr, ogr, gdal_array
from osgeo.gdalconst import GA_ReadOnly, GA_Update, GMF_PER_DATASET
from pycrs.parse import from_ogc_wkt as parse_crs_from_ogc_wkt
from pyproj import Proj
from pystac import Asset, Item
import numpy as np

from gdal_subsetter.coordinate_utilities import (boxwrs84_boxproj,
                                                 calc_coord_ij, calc_ij_coord,
                                                 calc_subset_envelope_window,
                                                 get_bbox, lonlat_to_projcoord)
from gdal_subsetter.exceptions import (DownloadError,
                                       HGAException,
                                       IncompatibleVariablesError,
                                       MultipleZippedNetCDF4FilesError,
                                       UnknownFileFormatError)
from gdal_subsetter.shape_file_utilities import (box_to_shapefile,
                                                 convert_to_multipolygon,
                                                 get_coordinates_unit,
                                                 shapefile_boxproj)
from gdal_subsetter.utilities import (get_file_type, get_files_from_unzipfiles,
                                      get_version, has_world_file, is_geotiff,
                                      mime_to_extension, mime_to_gdal,
                                      OpenGDAL, process_flags, rename_file,
                                      resampling_methods)


class HarmonyAdapter(BaseHarmonyAdapter):
    """
    See https://github.com/nasa/harmony-service-lib-py
    for documentation and examples.
    """

    def process_item(self, input_item: Item, source: HarmonySource) -> Item:
        """
        Converts an input STAC Item's data into Zarr, returning an output STAC item

        Parameters
        ----------
        input_item : pystac.Item
            the item that should be converted
        source : harmony.message.Source, imported as HarmonySource
            the input source defining the variables, if any, to subset from the item

        Returns
        -------
        pystac.Item
            a STAC item containing the Zarr output
        """
        if self.message.subset and self.message.subset.shape:
            self.logger.warning('Ignoring subset request for user shapefile '
                                f'{self.message.subset.shape.href}')

        layernames = []
        operations = {'variable_subset': source.variables,
                      'is_regridded': bool(self.message.format.crs
                                           or self.message.format.width
                                           or self.message.format.height),
                      'is_subsetted': bool(self.message.subset
                                           and (self.message.subset.bbox
                                                or self.message.subset.shape))}

        output_dir = mkdtemp()
        try:
            # Get the data file
            asset = next(item_asset
                         for item_asset in input_item.assets.values()
                         if 'data' in (item_asset.roles or []))

            if self.config.fallback_authn_enabled:
                access_token = None
            else:
                access_token = self.message.accessToken

            try:
                temporary_filename = download(asset.href, output_dir,
                                              logger=self.logger,
                                              cfg=self.config,
                                              access_token=access_token)
            except Exception as exception:
                raise DownloadError(asset.href, str(exception)) from exception

            input_filename = rename_file(temporary_filename, asset.href)

            output_basename = os.path.splitext(os.path.basename(
                generate_output_filename(asset.href, **operations))
            )[0]

            file_type = get_file_type(input_filename)

            if file_type is None:
                return input_item
            elif file_type == 'tif':
                layernames, transformed_filename = self.process_geotiff(
                    source, output_basename, input_filename, output_dir,
                    layernames
                )
            elif file_type == 'nc':
                layernames, transformed_filename, process_msg = self.process_netcdf(
                    source, output_basename, input_filename, output_dir,
                    layernames
                )

            elif file_type == 'zip':
                layernames, transformed_filename, process_msg = self.process_zip(
                    source, output_basename, input_filename, output_dir,
                    layernames
                )
            else:
                self.logger.warning('Will not process Unrecognised input file '
                                    f'format, "{file_type}".')
                raise UnknownFileFormatError(file_type)

            if layernames and transformed_filename:
                transformed_bbox = self.get_bbox_lonlat(transformed_filename)
                # May need to convert transformed output to the MIME type
                # specified in the request (e.g., GeoTIFF to NetCDF-4).
                transformed_filename = self.reformat(transformed_filename,
                                                     output_dir)
                return self.stage_and_get_output_stac(transformed_filename,
                                                      transformed_bbox,
                                                      input_item,
                                                      output_basename)
            else:
                raise HGAException(f'No stac_record created: {process_msg}')

        except Exception as exception:
            self.logger.exception(exception)
            raise exception

        finally:
            rmtree(output_dir)

    def stage_and_get_output_stac(self, transformed_filename: str,
                                  transformed_bbox: List[float],
                                  input_stac_item: Item,
                                  staged_basename: str) -> Item:
        """ Stage the output file(s). This will be the transformed object and,
            for PNG and JPEG formats, a world file. A STAC Item is returned
            containing Assets for all staged outputs.

        """
        output_stac_item = input_stac_item.clone()
        output_stac_item.assets = {}
        output_stac_item.bbox = transformed_bbox
        output_stac_item.geometry = bbox_to_geometry(output_stac_item.bbox)

        staged_filename = (staged_basename
                           + os.path.splitext(transformed_filename)[-1])

        staged_url = stage(transformed_filename,
                           staged_filename,
                           self.message.format.mime,
                           location=self.message.stagingLocation,
                           logger=self.logger,
                           cfg=self.config)

        output_stac_item.assets['data'] = Asset(
            staged_url, title=staged_filename,
            media_type=self.message.format.mime, roles=['data']
        )

        if has_world_file(self.message.format.mime):
            world_filename = os.path.splitext(transformed_filename)[0] + '.wld'
            staged_world_filename = os.path.splitext(staged_filename)[0] + '.wld'
            world_url = stage(world_filename,
                              staged_world_filename,
                              'text/plain',
                              location=self.message.stagingLocation,
                              logger=self.logger,
                              cfg=self.config)

            output_stac_item.assets['metadata'] = Asset(
                world_url, title=staged_world_filename,
                media_type='text/plain', roles=['metadata']
            )

        return output_stac_item

    def process_geotiff(self, source: HarmonySource, basename: str,
                        input_filename: str, output_dir: str,
                        layernames: List[str]):
        filelist = []
        if not source.variables:
            # process geotiff and all bands

            filename = input_filename
            layer_id = basename + '__all'
            band = None
            layer_id, filename, output_dir = self.perform_transforms(
                layer_id, filename, output_dir, band)

            filelist.append(filename)
            result = self.add_to_result(filelist, output_dir)
            layernames.append(layer_id)
        else:
            # all possible variables (band names) in the input_filenanme
            variables = self.get_variables(input_filename)
            # source.process('variables') is user's requested variables (bands)
            for variable in source.process('variables'):

                band = None

                for key, value in variables.items():
                    if 'band' in variable.name.lower():
                        if variable.name in key:
                            band = int(key.split('Band')[-1])
                            break
                    else:
                        if variable.name.lower() in value.lower():
                            band = int(key.split('Band')[-1])
                            break

                if band:
                    filename = input_filename
                    layer_id = basename + '__' + variable.name
                    layer_id, filename, output_dir = self.perform_transforms(
                        layer_id, filename, output_dir, band)
                    layernames.append(layer_id)
                    filelist.append(filename)

            result = self.add_to_result(filelist, output_dir)

        return layernames, result

    def process_netcdf(self, source: HarmonySource, basename:str,
                       input_filename: str, output_dir: str,
                       layernames: List[str]):

        filelist = []

        variables = (source.process('variables')
                     or self.get_variables(input_filename))

        for variable in variables:
            band = None
            # For non-geotiffs, we refer to variables by appending a file path
            layer_format = self.read_layer_format(source.collection,
                                                  input_filename,
                                                  variable.name)
            filename = layer_format.format(input_filename)

            layer_id = '__'.join([basename, variable.name])

            # convert a subdataset in the nc file into the GeoTIFF file
            filename = self.nc2tiff(layer_id, filename, output_dir)
            if filename:
                layer_id, filename, output_dir = self.perform_transforms(
                    layer_id, filename, output_dir, band
                )

                layernames.append(layer_id)

                filelist.append(filename)

        result = self.add_to_result(filelist, output_dir)

        if result:
            process_msg = 'OK'
        else:
            process_msg = 'subsets in the nc file are not stackable, not processed.'

        return layernames, result, process_msg

    def process_zip(self, source: HarmonySource, basename: str,
                    zipfile_name: str, output_dir: str, layernames: List[str]):
        """ Unpack the input zip file and process the individual files it
            contains. If there are GeoTIFF files, only these will be
            processed.

            `HarmonyAdapter.unpack_zipfile` will attempt to stack multiple
            GeoTIFFs contained within the zip file, and process those as a
            single file. The possibility of multiple NetCDF-4 files does not
            seem to be accounted for here.

        """
        variable_names = [item.name for item in source.process('variables')]
        tiffile, msg_tif, ncfile, msg_nc = self.unpack_zipfile(zipfile_name,
                                                               output_dir,
                                                               variable_names)

        if len(tiffile) > 0:
            layernames, result = self.process_geotiff(source, basename,
                                                      tiffile, output_dir,
                                                      layernames)
            message = msg_tif
        elif len(ncfile) > 1:
            raise MultipleZippedNetCDF4FilesError(zipfile_name)
        elif len(ncfile) == 1:
            layernames, result = self.process_netcdf(source, basename,
                                                     ncfile[0], output_dir,
                                                     layernames)
            message = msg_nc
        else:
            result = None
            message = ('The granule does not have GeoTIFF or NetCDF-4 files, '
                       'will not process.')

        return layernames, result, message

    def update_layernames(self, file_name, layer_names):
        """
        Updates the layers in the given file to match the list of layernames provided

        Parameters
        ----------
        filename : string
            The path to file whose layernames should be updated
        layernames : string[]
            An array of names, in order, to apply to the layers
        """
        with OpenGDAL(file_name) as dataset:
            for index, layer_name in enumerate(layer_names, start=1):
                dataset.GetRasterBand(index).SetDescription(layer_name)

    def execute_gdal_command(self, *args) -> List[str]:
        """ This is used to execute gdal* commands. """
        self.logger.info(
            args[0] + ' ' + ' '.join(["'{}'".format(arg) for arg in args[1:]])
        )
        result_str = check_output(args).decode('utf-8')
        return result_str.split('\n')

    def nc2tiff(self, layerid, filename, dstdir):

        def search(input_dict, lookup_key):
            for dictionary_key, dictionary_value in input_dict.items():
                if lookup_key in dictionary_key:
                    return dictionary_value

        normalized_layerid = layerid.replace('/', '_')
        dstfile = os.path.join(dstdir, f'{normalized_layerid}__nc2tiff.tif')

        try:
            with OpenGDAL(filename) as dataset:
                crs_wkt = search(dataset.GetMetadata(), 'crs_wkt')

            if crs_wkt is None:
                crs_wkt = 'EPSG:4326'

            command = ['gdal_translate', '-a_srs']
            command.extend([crs_wkt])
            command.extend([filename, dstfile])
            self.execute_gdal_command(*command)

            return dstfile
        except Exception as error:
            self.logger.exception(error)
            raise HGAException('Could not convert NetCDF-4 to GeoTIFF')

    def varsubset(self, srcfile, dstfile, band=None):
        if not band:
            return srcfile

        command = ['gdal_translate']
        command.extend(['-b', str(band)])
        command.extend([srcfile, dstfile])
        self.execute_gdal_command(*command)

        return dstfile

    def subset(self, layerid, srcfile, dstdir, band=None):
        """Subset layer to region defined in message.subset."""
        normalized_layerid = layerid.replace('/', '_')
        subset = self.message.subset
        if subset.bbox is None and subset.shape is None:
            dstfile = os.path.join(dstdir,
                                   f'{normalized_layerid}__varsubsetted.tif')
            dstfile = self.varsubset(srcfile, dstfile, band=band)
            return dstfile

        if subset.bbox:
            [left, bottom, right, top] = get_bbox(srcfile)

            # subset.bbox is defined as [left/west,low/south,right/east,upper/north]
            subsetbbox = subset.process('bbox')

            [b0, b1], transform = lonlat_to_projcoord(srcfile, subsetbbox[0],
                                                      subsetbbox[1])
            [b2, b3], transform = lonlat_to_projcoord(srcfile, subsetbbox[2],
                                                      subsetbbox[3])

            if any(x is None for x in [b0, b1, b2, b3]):
                dstfile = os.path.join(
                    dstdir, f'{normalized_layerid}__varsubsetted.tif'
                )
                dstfile = self.varsubset(srcfile, dstfile, band=band)
            elif b0 < left and b1 < bottom and b2 > right and b3 > top:
                # user's input subset totally covers the image bbox, do not do subset
                dstfile = os.path.join(
                    dstdir, f'{normalized_layerid}__varsubsetted.tif'
                )
                dstfile = self.varsubset(srcfile, dstfile, band=band)
            else:
                dstfile = os.path.join(
                    dstdir, f'{normalized_layerid}__subsetted.tif'
                )
                dstfile = self.subset2(srcfile, dstfile, bbox=subsetbbox,
                                       band=band)

            return dstfile

        if subset.shape:
            # need know what the subset.shape is. Is it a file? do we need
            # pre-process like we did for subset.bbox?
            dstfile = os.path.join(dstdir,
                                   f'{normalized_layerid}__subsetted.tif')
            # get the shapefile
            shapefile = self.get_shapefile(subset.shape, dstdir)
            dstfile = self.subset2(srcfile, dstfile, shapefile=shapefile,
                                   band=band)
            return dstfile

    def get_shapefile(self, subsetshape, dstdir):
        """
        read the shapefile passing from harmony, it is geojson file, if it is
        point or line feature file, convert it into and polygon geojson,
        then convert it into a ESRI shapefile.
        input: subset.shape
        return: ESRI shapefile (without .zip affix), it actualy produce 4 files which
        consist of ESRI shapefile
        """
        href = subsetshape.href
        filetype = subsetshape.type
        shapefile = download(href, dstdir, logger=self.logger, cfg=self.config,
                             access_token=self.message.accessToken)

        # get unit of the feature in the shapefile
        geometry, unit = get_coordinates_unit(shapefile)

        # convert into multi-polygon feature file
        fileprex = os.path.splitext(shapefile)[0]
        tmpfile_geojson = f'{fileprex}_tmp.geojson'

        # check BUFFER environment to get buf from message passed by harmony
        buffer_string = os.environ.get('BUFFER')
        if buffer_string:
            buffer_dic = eval(buffer_string)

            if unit and 'Polygon' not in geometry:
                if 'degree' in unit.lower():
                    buf = buffer_dic['degree']
                else:
                    buf = buffer_dic['meter']
            else:
                buf = None
        else:
            buf = None

        # convert to a new multipolygon file
        tmpfile_geojson = convert_to_multipolygon(shapefile, tmpfile_geojson,
                                                  buf=buf)

        # convert into ESRI shapefile
        outfile = f'{fileprex}.shp'
        command = ['ogr2ogr', '-f', 'ESRI Shapefile']
        command.extend([outfile, tmpfile_geojson])
        self.execute_gdal_command(*command)

        return outfile

    def resize(self, layerid, srcfile, dstdir):
        """Resizes the input layer

        Uses a call to gdal_translate using information in message.format.

        """
        interpolation = self.message.format.process('interpolation')
        if interpolation in resampling_methods:
            resample_method = interpolation
        else:
            resample_method = 'bilinear'

        command = ['gdal_translate']
        fmt = self.message.format
        normalized_layerid = layerid.replace('/', '_')
        dstfile = os.path.join(dstdir, f'{normalized_layerid}__resized.tif')

        if self.message.format.mime == 'image/png':
            # need to unscale values to color PNGs correctly
            command.extend(['-unscale', '-ot', 'Float64'])

        if fmt.width or fmt.height:
            width = fmt.process('width') or 0
            height = fmt.process('height') or 0
            command.extend(['-outsize', str(width), str(height)])
            command.extend(['-r', resample_method])
            command.extend([srcfile, dstfile])
            self.execute_gdal_command(*command)
            return dstfile
        else:
            return srcfile

    def reproject(self, layerid, srcfile, dstdir):
        crs = self.message.format.process('crs')
        interpolation = self.message.format.process('interpolation')
        if interpolation in resampling_methods:
            resample_method = interpolation
        else:
            resample_method = 'bilinear'
        if not crs:
            return srcfile

        normalized_layerid = layerid.replace('/', '_')
        dstfile = os.path.join(dstdir,
                               f'{normalized_layerid}__reprojected.tif')
        fmt = self.message.format
        if fmt.scaleSize:
            xres, yres = fmt.process('scaleSize').x, fmt.process('scaleSize').y
        else:
            xres, yres = None, None
        if fmt.scaleExtent:
            box = [fmt.process('scaleExtent').x.min,
                   fmt.process('scaleExtent').y.min,
                   fmt.process('scaleExtent').x.max,
                   fmt.process('scaleExtent').y.max]
        else:
            box = None

        def _regrid(infile, outfile, resampling_mode='bilinear', ref_crs=None,
                    ref_box=None, ref_xres=None, ref_yres=None):
            command = ['gdalwarp', '-of', 'GTiff',  '-overwrite',
                       '-r', resampling_mode]

            if ref_crs:  # proj, box, xres/yres
                command.extend(['-t_srs', ref_crs])
                # command.extend([ '-t_srs', "'{ref_crs}'".format(ref_crs=ref_crs)])
            if ref_box:
                box = [str(x) for x in ref_box]
                command.extend(['-te', box[0], box[1], box[2], box[3],
                                '-te_srs', ref_crs])
                # command.extend(['-te', box[0], box[1], box[2], box[3],
                #                 '-te_srs', "'{ref_crs`}'".format(ref_crs=ref_crs)])
            if ref_xres and ref_yres:
                command.extend(['-tr', str(ref_xres), str(ref_yres)])

            command.extend([infile, outfile])
            self.execute_gdal_command(*command)
            return outfile

        return _regrid(srcfile, dstfile, resampling_mode=resample_method,
                       ref_crs=crs, ref_box=box, ref_xres=xres, ref_yres=yres)

    def add_to_result(self, filelist, dstdir):
        dstfile = os.path.join(dstdir, 'result')
        output = None
        if 'png' in self.message.format.mime:
            dstfile += '.png'
            output = dstfile
        elif 'jpeg' in self.message.format.mime:
            dstfile += '.jpeg'
            output = dstfile
        else:
            dstfile += '.tif'
            if filelist and self.is_stackable(filelist):
                output = self.stack_multi_file_with_metadata(filelist, dstfile)
            else:
                raise IncompatibleVariablesError(
                    'Request cannot be completed: datasets are incompatible '
                    'and cannot be combined.'
                )
        return output

    def stack_multi_file_with_metadata(self, infilelist: List[str],
                                       outfile: str) -> str:
        """ The infilelist includes multiple files, each file does may not have
            the same number of bands, but they must have the same projection
            and geotransform.

        """
        collection = []
        count = 0
        ds_description = []

        with OpenGDAL(infilelist[0]) as first_input_file:
            proj = first_input_file.GetProjection()
            geot = first_input_file.GetGeoTransform()
            cols = first_input_file.RasterXSize
            rows = first_input_file.RasterYSize
            gtyp = first_input_file.GetRasterBand(1).DataType
            md = first_input_file.GetMetadata()

        for layer in infilelist:
            with OpenGDAL(layer) as ds:
                filename = ds.GetDescription()
                filestr = os.path.splitext(os.path.basename(filename))[0]

                bandnum = ds.RasterCount
                for band_index in range(1, bandnum + 1):
                    band = ds.GetRasterBand(band_index)
                    band_metadata = band.GetMetadata()
                    data = band.ReadAsArray()
                    nodata = band.GetNoDataValue()
                    mask = band.GetMaskBand().ReadAsArray()
                    count += 1
                    # update bandname, standard_name, and long_name
                    band_name = band_metadata.get('bandname')
                    if band_name is None:
                        if band.GetDescription():
                            band_name = band.GetDescription()
                        else:
                            band_name = f'{filestr}_band_{band_index}'

                    standard_name = band_metadata.get('standard_name', band_name)
                    long_name = band_metadata.get('long_name', band_name)

                    band_metadata.update({'bandname': band_name,
                                          'standard_name': standard_name,
                                          'long_name': long_name})

                    if band.GetDescription():
                        band_desc = band.GetDescription()
                    else:
                        band_desc = band_name

                    ds_description.append(f'Band{count}:{band_name}')
                    collection.append({'band_sn': count,
                                       'band_md': band_metadata,
                                       'band_desc': band_desc,
                                       'band_array': data,
                                       'mask_array': mask,
                                       'nodata': nodata})

        dst_ds = gdal.GetDriverByName('GTiff').Create(outfile, cols, rows,
                                                      count, gtyp)
        # maskband is read-only by default, you have to create the maskband
        # before you can write the maskband
        gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
        dst_ds.CreateMaskBand(GMF_PER_DATASET)
        dst_ds.SetProjection(proj)
        dst_ds.SetGeoTransform(geot)
        dst_ds.SetMetadata(md)
        dst_ds.SetDescription(' '.join(ds_description))

        for band_index, band in enumerate(collection, start=1):
            dst_band = dst_ds.GetRasterBand(band_index)
            dst_band.WriteArray(band['band_array'])
            dst_band.SetMetadata(band['band_md'])
            dst_band.SetDescription(band['band_desc'])
            dst_band.GetMaskBand().WriteArray(band['mask_array'])
            if band['nodata']:
                dst_band.SetNoDataValue(band['nodata'])

        dst_ds.FlushCache()                     # write to disk
        dst_ds = None
        return outfile

    def reformat(self, srcfile: str, dstdir: str) -> str:
        """ This method converts the transformed outputs produced by the
            service into a file with the requested MIME type. For example,
            NetCDF-4 requested outputs, will need to be converted and
            aggregated from transformed GeoTIFFs back to a NetCDF-4 file.

            The return value of this method is the path to the output file with
            the requested MIME type.

        """
        gdal_subsetter_version = f'gdal_subsetter_version={get_version()}'
        output_mime = self.message.format.process('mime')
        if not output_mime == 'image/png' or output_mime == 'image/jpeg':
            command = ['gdal_edit.py',  '-mo', gdal_subsetter_version, srcfile]
            self.execute_gdal_command(*command)
        if output_mime not in mime_to_gdal:
            raise Exception(f'Unrecognized output format: {output_mime}')
        if output_mime == 'image/tiff':
            return srcfile

        dstfile = os.path.join(
            dstdir, f'translated.{mime_to_extension[output_mime]}'
        )

        if output_mime in ('application/x-netcdf4', 'application/x-zarr'):
            dstfile = os.path.join(dstdir, 'translated.nc')
            dstfile = self.geotiff2netcdf_direct(srcfile, dstfile)
            return dstfile
        else:
            # png, jpeg, gif, etc.
            if os.path.exists(f'{os.path.splitext(srcfile)[0]}.wld'):
                # don't need to reformat files that have been processed earlier
                return srcfile

            command = ['gdal_translate', '-of', mime_to_gdal[output_mime],
                       '-scale', srcfile, dstfile]
            self.execute_gdal_command(*command)
            return dstfile

    def read_layer_format(self, collection, filename, layer_id):
        gdalinfo_lines = gdal.Info(filename).splitlines()

        layer_line = next((line for line in gdalinfo_lines
                           if re.search(f'SUBDATASET.*{layer_id}$', line)
                           is not None), None)

        if layer_line is None:
            print('Invalid Layer:', layer_id)

        layer = layer_line.split('=')[-1]

        return layer.replace(filename, '{}')

    def get_variables(self, filename: str):
        """ filename is either nc or tif. """
        gdalinfo_lines = gdal.Info(filename).splitlines()

        result = []
        if 'netCDF' in gdalinfo_lines[0] or 'HDF' in gdalinfo_lines[0]:
            # netCDF/Network Common Data Format, HDF5/Hierarchical Data Format
            # Release 5
            # Normal case of NetCDF / HDF, where variables are subdatasets
            for subdataset in filter((lambda line: re.match(r'^\s*SUBDATASET_\d+_NAME=', line)), gdalinfo_lines):
                result.append(HarmonyVariable({'name': re.split(r':', subdataset)[-1]}))
        elif is_geotiff(filename):
            #  GTiff/GeoTIFF
            # GeoTIFFs, directly use Band # as the variables.
            # for subdataset in filter((lambda line: re.match(r"^Band", line)), gdalinfo_lines):
            #     tmpline = re.split(r" ", subdataset)
            #     result.append(ObjectView({"name": tmpline[0].strip()+tmpline[1].strip()}))
            result = {}
            with OpenGDAL(filename) as dataset:
                for band_index in range(1, dataset.RasterCount + 1):
                    band = dataset.GetRasterBand(band_index)
                    bmd = band.GetMetadata()
                    if 'standard_name' in bmd:
                        result.update({f'Band{band_index}': bmd['standard_name']})
                    else:
                        result.update({f'Band{band_index}': f'Band{band_index}'})

        return result

    def perform_transforms(self, layer_id, filename, output_dir, band):
        """Push layer through the series of transforms. """
        filename = self.subset(layer_id, filename, output_dir, band)
        filename = self.reproject(layer_id, filename, output_dir)
        filename = self.resize(layer_id, filename, output_dir)

        return layer_id, filename, output_dir

    def get_bbox_lonlat(self, filename: str) -> List[float]:
        """ Get the bbox in longitude and latitude of the raster file, and
            update the bbox and extent for the file, and return bbox.

            input:
                filename: raster file name
            output: bbox of the raster file

        """
        output_mime = self.message.format.process('mime')

        if output_mime in ('image/png', 'image/jpeg'):
            gdal_mode = GA_ReadOnly
        else:
            gdal_mode = GA_Update

        with OpenGDAL(filename, gdal_mode) as dataset:
            geotransform = dataset.GetGeoTransform()
            cols = dataset.RasterXSize
            rows = dataset.RasterYSize
            if output_mime in ('image/png', 'image/jpeg'):
                proj_string = '+proj=longlat +datum=WGS84 +no_defs'
            else:
                dst = osr.SpatialReference(dataset.GetProjection())
                proj_string = dst.ExportToProj4()

            ct2 = Proj(proj_string)

            ul_x, ul_y = calc_ij_coord(geotransform, 0, 0)
            ur_x, ur_y = calc_ij_coord(geotransform, cols, 0)
            lr_x, lr_y = calc_ij_coord(geotransform, cols, rows)
            ll_x, ll_y = calc_ij_coord(geotransform, 0, rows)
            ul_x2, ul_y2 = ct2(ul_x, ul_y, inverse=True)
            ur_x2, ur_y2 = ct2(ur_x, ur_y, inverse=True)
            lr_x2, lr_y2 = ct2(lr_x, lr_y, inverse=True)
            ll_x2, ll_y2 = ct2(ll_x, ll_y, inverse=True)
            ul_x2 = float(f'{ul_x2:.7f}')
            ul_y2 = float(f'{ul_y2:.7f}')
            ur_x2 = float(f'{ur_x2:.7f}')
            ur_y2 = float(f'{ur_y2:.7f}')
            lr_x2 = float(f'{lr_x2:.7f}')
            lr_y2 = float(f'{lr_y2:.7f}')
            ll_x2 = float(f'{ll_x2:.7f}')
            ll_y2 = float(f'{ll_y2:.7f}')
            lon_left = min(ul_x2, ll_x2)
            lat_low = min(ll_y2, lr_y2)
            lon_right = max(lr_x2, ur_x2)
            lat_high = max(ul_y2, ur_y2)
            # write bbox and extent in lon/lat unit to the metadata of
            # the filename
            md = dataset.GetMetadata()
            bbox = [lon_left, lat_low, lon_right, lat_high]
            extent = {'ul': [ul_x2, ul_y2], 'll': [ll_x2, ll_y2],
                      'ur': [ur_x2, ur_y2], 'lr': [lr_x2, lr_y2]}
            md['bbox'] = str(bbox)
            md['extent'] = str(extent)
            dataset.SetMetadata(md)

        return bbox

    def is_stackable(self, files_to_stack: List[str]) -> bool:
        """ Returns a boolean value indicating if all files in the input list
            can be stacked. This is indicated by whether they have the same
            projection, geotransform and raster sizes. PNG files cannot be
            stacked.

            input:  list of GeoTIFF file names.
            return: boolean indicating that if the listed files can be stacked.

        """
        projections = set()
        geotransforms = set()
        dtypes = set()
        xsizes = set()
        ysizes = set()
        formats = set()

        for file_to_stack in files_to_stack:
            with OpenGDAL(file_to_stack) as dataset:
                projections.add(dataset.GetProjection())
                geotransforms.add(str(dataset.GetGeoTransform()))
                dtypes.add(dataset.GetRasterBand(1).DataType)
                xsizes.add(dataset.RasterXSize)
                ysizes.add(dataset.RasterYSize)

            formats.add(os.path.splitext(file_to_stack)[1])

        same_grid = all(
            len(grid_property_values) == 1
            for grid_property_values
            in (projections, geotransforms, dtypes, xsizes, ysizes)
        )
        stackable_format = '.png' not in formats

        return same_grid and stackable_format

    def unpack_zipfile(self, zipfilename: str, output_dir: str,
                       variable_names: List[str]):
        """ Extract the files contained within an input zip file. If the zip
            file contains GeoTIFF format files, this function assumes that the
            constituent files are separate single bands that represent the same
            variable. The function will attempt to stack these separate GeoTIFF
            files into a single multi-band file, the path to which will be
            returned from this function.

        """
        with ZipFile(zipfilename, 'r') as zip_ref:
            zip_ref.extractall(output_dir+'/unzip')

        tmptif = None
        filelist_tif = get_files_from_unzipfiles(f'{output_dir}/unzip',
                                                 'tif', variable_names)
        if len(filelist_tif) > 0:
            if self.is_stackable(filelist_tif):
                tmpfile = f'{output_dir}/tmpfile'
                tmptif = self.stack_multi_file_with_metadata(filelist_tif, tmpfile)
                msg_tif = 'OK'
            else:
                msg_tif = 'variables are not stackable, not process.'
        else:
            msg_tif = 'no available data for the variables in the granule, not process.'

        tmpnc = None
        filelist_nc = get_files_from_unzipfiles(f'{output_dir}/unzip', 'nc')
        if len(filelist_nc) > 0:
            tmpnc = filelist_nc
            msg_nc = 'OK'
        else:
            msg_nc = 'no available data for the variables, not process.'

        return tmptif, msg_tif, tmpnc, msg_nc

    def subset2(self, tiffile: str, outputfile: str, bbox=None,
                band=None, shapefile=None) -> str:
        """ Subset a GeoTIFF with a bounding box or shapefile. Bounding box and
            shapefile are exclusive.

            inputs: tiffile: GeoTIFF file path
                    outputfile: subsetted file name
                    bbox: [left,low,right,upper] in lon/lat coordinates
                    shapefile: a shapefile directory in which multiple files
                        exist
        """
        process_flags['subset'] = True
        # covert to absolute path
        tiffile = os.path.abspath(tiffile)
        outputfile = os.path.abspath(outputfile)
        # RasterFormat = 'GTiff'
        tmpfile = f'{os.path.splitext(outputfile)[0]}-tmp.tif'
        if bbox or shapefile:
            if bbox:
                shapefile_out = box_to_shapefile(tiffile, bbox)
                boxproj, _ = boxwrs84_boxproj(bbox, tiffile)
            else:
                shapefile_out = os.path.join(os.path.dirname(outputfile),
                                             'tmpshapefile')
                boxproj = shapefile_boxproj(shapefile, tiffile,
                                            shapefile_out)

            ul_x, ul_y, ul_i, ul_j, cols, rows = calc_subset_envelope_window(
                tiffile, boxproj
            )

            command = ['gdal_translate']

            if band:
                command.extend(['-b', str(band)])

            command.extend(['-srcwin', str(ul_i), str(ul_j), str(cols),
                            str(rows)])

            command.extend([tiffile, tmpfile])
            self.execute_gdal_command(*command)
            self.mask_via_combined(tmpfile, shapefile_out, outputfile)
        else:
            copyfile(tiffile, outputfile)

        return outputfile

    def mask_via_combined(self, inputfile, shapefile, outputfile):
        """ Calculates the maskbands and set databands with nodata value for
            each band according to the shape file.

            inputs:
                inputfile is geotiff file.
                shapefile int same coordinate reference as the inputfile.
            return:
                outputfile is masked geotiff file.
        """
        # define temporary file name
        tmpfile = f'{os.path.splitext(inputfile)[0]}-tmp.tif'
        copyfile(inputfile, tmpfile)
        copyfile(inputfile, outputfile)

        # read shapefile info
        shp = ogr.Open(shapefile)
        shape_file_layer = shp.GetLayerByIndex(0)

        # update the outputfile
        with OpenGDAL(outputfile, GA_Update) as dst_dataset:
            gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
            # this changes the maskband data in the dst_dataset
            dst_dataset.CreateMaskBand(GMF_PER_DATASET)

            # update tmpfile (used as a mask file)
            with OpenGDAL(tmpfile, GA_Update) as tmp_dataset:
                for band_index in range(1, tmp_dataset.RasterCount + 1):
                    self._mask_band(tmp_dataset, band_index, dst_dataset,
                                    shape_file_layer)

                tmp_dataset.FlushCache()

            dst_dataset.FlushCache()

        return outputfile

    def _mask_band(self, tmp_ds: gdal.Dataset, band_index: int,
                   dst_ds: gdal.Dataset, shape_file_layer: ogr.Layer):
        """ An internal method that is only called by `self.mask_via_combined`.
            This method uses an input shape file layer to determine which
            pixels in a raster array are within the shape file (or not). Those
            outside of the shape are masked.

        """
        tmp_band = tmp_ds.GetRasterBand(band_index)
        tmp_data = tmp_band.ReadAsArray()
        tmp_mskband_pre = tmp_band.GetMaskBand()
        tmp_msk_pre = tmp_mskband_pre.ReadAsArray()
        tmp_nodata_pre = tmp_band.GetNoDataValue()
        np_dt = gdal_array.GDALTypeCodeToNumericTypeCode(tmp_band.DataType)
        tmp_band.WriteArray(np.zeros(tmp_data.shape, np_dt))
        # this flushCache() changes the all values in the maskband data to 255.
        tmp_band.FlushCache()
        bands = [band_index]
        burn_value = 1
        burns = [burn_value]
        # following statement modify the maskband data to 255.
        # The original maskband data is stored in tmp_msk_pre.
        err = gdal.RasterizeLayer(tmp_ds, bands, shape_file_layer,
                                  burn_values=burns,
                                  options=['ALL_TOUCHED=TRUE'])
        tmp_ds.FlushCache()
        # combine original tmp mask band with tmp_data
        tmp_band = tmp_ds.GetRasterBand(band_index)
        # tmp_data includes 0 and 1 values, where 0 indicates no valid pixels, and 1 indicates valid pixels.
        tmp_data = tmp_band.ReadAsArray()
        out_band = dst_ds.GetRasterBand(band_index)
        out_data = out_band.ReadAsArray()
        # set out_band with nodata value if the tmp_band has nodata value
        if tmp_nodata_pre:
            msk = tmp_data == 0
            out_data[msk] = tmp_nodata_pre
            out_band.WriteArray(out_data)
            out_band.SetNoDataValue(tmp_nodata_pre)

        # modify out_mskband
        out_mskband = out_band.GetMaskBand()
        out_msk = tmp_data*tmp_msk_pre
        out_mskband.WriteArray(out_msk)
        out_mskband.FlushCache()
        out_band.FlushCache()

    def geotiff2netcdf_direct(self, infile: str, outfile: str):
        """ Convert GeoTIFF file to NetCDF-4 file by reading the GeoTIFF and
            writing to a new NetCDF-44 format file.

            input:
                infile - geotiff file name
            return:
                outfile - netcdf file name
        """
        def _process_projected(ds_in: gdal.Dataset, dst: Dataset):
            gt = ds_in.GetGeoTransform()
            crs = parse_crs_from_ogc_wkt(ds_in.GetProjectionRef())
            unitname = crs.unit.unitname.proj4
            # define dimensions
            dst.createDimension('x', ds_in.RasterXSize)
            dst.createDimension('y', ds_in.RasterYSize)
            # copy attributes, geotiff metadata is party of attributes in netcdf.
            # Conventions, GDAL, history
            for attribute_name, attribute_value in ds_in.GetMetadata().items():
                if attribute_name.find(r'#') > -1:
                    tmp = attribute_name.split(r'#')
                    if tmp[0] == 'NC_GLOBAL':
                        dst.setncattr(tmp[1], attribute_value.rstrip('\n'))
                else:
                    if attribute_name not in ('_FillValue', 'Conventions'):
                        dst.setncattr(attribute_name,
                                      attribute_value.rstrip('\n'))

            # create georeference variable
            crs_name = crs.proj.name.ogc_wkt.lower()
            geovar = dst.createVariable(crs_name, 'S1')
            geovar.grid_mapping_name = crs_name
            geovar.long_name = 'CRS definition'
            for item in crs.params:
                attr_name = str(item).split('.')[-1].split(' ')[0]
                attr_lst = re.findall('[A-Z][^A-Z]*', attr_name)
                name = '_'.join(attr_lst).lower()
                geovar.setncattr(name, item.value)

            geovar.longitude_of_prime_meridian = crs.geogcs.prime_mer.value
            if crs.geogcs.datum.ellips:
                geovar.semi_major_axis = crs.geogcs.datum.ellips.semimaj_ax.value
                geovar.inverse_flattening = crs.geogcs.datum.ellips.inv_flat.value

            geovar.spatial_ref = ds_in.GetProjectionRef()
            geovar.GeoTransform = ' '.join(map(str, list(gt)))

            # create 1D coordinate variables if the geotiff is not rotated image
            if gt[2] == 0.0 and gt[4] == 0.0:
                x_array = gt[0] + gt[1]*(np.arange(ds_in.RasterXSize) + 0.5)
                y_array = gt[3] + gt[5]*(np.arange(ds_in.RasterYSize) + 0.5)
                xvar = dst.createVariable('x', np.dtype('float64'), ('x'))
                xvar[:] = x_array
                xvar.setncattr('standard_name', 'projection_x_coordinate')
                xvar.setncattr('axis', 'X')
                xvar.setncattr('long_name',
                               'x-coordinate in projected coordinate system')
                xvar.setncattr('units', unitname)
                yvar = dst.createVariable('y', np.dtype('float64'), ('y'))
                yvar[:] = y_array
                yvar.setncattr('standard_name', 'projection_y_coordinate')
                xvar.setncattr('axis', 'Y')
                yvar.setncattr('long_name',
                               'y-coordinate in projected coordinate system')
                yvar.setncattr('units', unitname)
                lcc =Proj(ds_in.GetProjectionRef())

                # lon 1D
                tmp_y = np.zeros(x_array.shape, x_array.dtype)
                tmp_y[:] = y_array[0]
                lon, tmp_lat = lcc(x_array, tmp_y, inverse=True)

                # lat 1D
                tmp_x = np.zeros(y_array.shape, y_array.dtype)
                tmp_x[:] = x_array[0]
                tmp_lon, lat = lcc(tmp_x, y_array, inverse=True)

                lon_var = dst.createVariable('lon', np.float64, ('x'), zlib=True)
                lon_var[:] = lon
                lon_var.units = 'degrees_east'
                lon_var.standard_name = 'longitude'
                lon_var.long_name = 'longitude'

                lat_var = dst.createVariable('lat', np.float64, ('y'), zlib=True)
                lat_var[:] = lat
                lat_var.units = 'degrees_north'
                lat_var.standard_name = 'latitude'
                lat_var.long_name = 'latitude'

            # create data variables
            for band_index in range(1, ds_in.RasterCount + 1):
                band = ds_in.GetRasterBand(band_index)
                meta = band.GetMetadata()
                mask_band = band.GetMaskBand()
                data = band.ReadAsArray()
                mask = mask_band.ReadAsArray()
                mx = masked_array(data, mask=mask == 0)
                # get varname
                varnames = [item for item in meta if item == 'standard_name']
                if varnames:
                    varname =meta[varnames[0]].replace('-', '_')
                else:
                    varname = f'Band{band_index}'.replace('-', '_')

                vardatatype = mx.data.dtype
                fillvalue = band.GetNoDataValue()
                if fillvalue:
                    datavar = dst.createVariable(varname, vardatatype,
                                                 ('y', 'x'), zlib=True,
                                                 fill_value=fillvalue)
                else:
                    datavar = dst.createVariable(varname, vardatatype,
                                                 ('y', 'x'), zlib=True)

                datavar[:, :] = mx

                # write attrs of the variabale datavar
                for attribute_name, attribute_value in band.GetMetadata().items():
                    if attribute_name.find(r'#') > -1:
                        tmp = attribute_name.split(r'#')
                        if tmp[0] == 'NC_GLOBAL':
                            dst.setncattr(tmp[1], attribute_value.rstrip('\n'))

                    else:
                        if attribute_name != '_FillValue':
                            if (
                                attribute_name == 'units'
                                and attribute_value == 'unitless'
                            ):
                                datavar.setncattr(attribute_name, '1')
                            else:
                                datavar.setncattr(
                                    attribute_name,
                                    attribute_value.rstrip('\n').replace('-', '_')
                                )

                datavar.grid_mapping = crs_name

                # add standard_name no standard_name in datavar
                lst = [attr for attr in datavar.ncattrs()
                       if attr in ['standard_name', 'long_name']]

                if not lst:
                    datavar.standard_name = varname

                # add units attr
                if 'units' not in datavar.ncattrs():
                    datavar.setncattr('units', '1')

        def _process_geogcs(ds_in: gdal.Dataset, dst: Dataset):
            gt = ds_in.GetGeoTransform()
            crs = parse_crs_from_ogc_wkt(ds_in.GetProjectionRef())
            # define dimensions
            dst.createDimension('lon', ds_in.RasterXSize)
            dst.createDimension('lat', ds_in.RasterYSize)
            # copy attributes, geotiff metadata is party of attributes in netcdf.
            for attribute_name, attribute_value in ds_in.GetMetadata().items():
                if attribute_name.find(r'#') > -1:
                    tmp = attribute_name.split(r'#')
                    if tmp[0] == 'NC_GLOBAL':
                        dst.setncattr(tmp[1], attribute_value.rstrip('\n'))

                else:
                    if attribute_name not in ('_FillValue', 'Conventions'):
                        dst.setncattr(attribute_name,
                                      attribute_value.rstrip('\n'))

            # create georeference variable
            crs_name = 'latitude_longitude'
            geovar = dst.createVariable(crs_name, 'S1')
            geovar.grid_mapping_name = crs_name
            geovar.long_name = 'CRS definition'
            geovar.longitude_of_prime_meridian = crs.prime_mer.value
            if crs.datum.ellips:
                geovar.semi_major_axis = crs.datum.ellips.semimaj_ax.value
                geovar.inverse_flattening = crs.datum.ellips.inv_flat.value

            geovar.spatial_ref = ds_in.GetProjectionRef()
            geovar.GeoTransform = ' '.join(map(str, list(gt)))
            # create coordinate variables if the geotiff is a non-rotated image
            if gt[2] == 0.0 and gt[4] == 0.0:
                lon_array = gt[0] + gt[1] * (np.arange(ds_in.RasterXSize) + 0.5)
                lat_array = gt[3] + gt[5] * (np.arange(ds_in.RasterYSize) + 0.5)
                lonvar = dst.createVariable('lon', np.dtype('float64'), ('lon'))
                lonvar[:] = lon_array
                lonvar.setncattr('standard_name', 'longitude')
                lonvar.setncattr('long_name', 'longitude')
                lonvar.setncattr('units', 'degrees_east')
                latvar = dst.createVariable('lat', np.dtype('float64'), ('lat'))
                latvar[:] = lat_array
                latvar.setncattr('standard_name', 'latitude')
                latvar.setncattr('long_name', 'latitude')
                latvar.setncattr('units', 'degrees_north')
            # else:
            #  create auxilliary coordinates
            # lcc =Proj(ds_in.GetProjectionRef())
            # J, I = np.meshgrid(np.arange(dst.dimensions['lon'].size), np.arange(dst.dimensions['lat'].size) )
            # lon_array = gt[0] + gt[1]*(J + 0.5) + gt[2]*(I + 0.5)
            # lat_array = gt[3] + gt[4]*(J + 0.5) + gt[5]*(I + 0.5)
            # lon, lat = lcc(lon_array, lat_array,inverse=True )
            # lon_var = dst.createVariable('lon', np.float64, ('lat', 'lon'), zlib=True)
            # lon_var[:,:] = lon
            # lon_var.units = 'degrees_east'
            # lon_var.standard_name = 'longitude'
            # lon_var.long_name = 'longitude'
            # lat_var = dst.createVariable('lat', np.float64, ('lat', 'lon'), zlib=True)
            # lat_var[:,:] = lat
            # lat_var.units = 'degrees_north'
            # lat_var.standard_name = 'latitude'
            # lat_var.long_name = 'latitude'

            # create data variables
            for band_index in range(1, ds_in.RasterCount + 1):
                band = ds_in.GetRasterBand(band_index)
                meta = band.GetMetadata()
                mask_band = band.GetMaskBand()
                data = band.ReadAsArray()
                mask = mask_band.ReadAsArray()
                mx = masked_array(data, mask=mask == 0)
                # get varname
                varnames = [item for item in meta if item == 'standard_name']
                if varnames:
                    varname =meta[varnames[0]].replace('-', '_')
                else:
                    varname = f'Band{band_index}'.replace('-', '_')

                vardatatype = mx.data.dtype
                fillvalue = band.GetNoDataValue()
                if fillvalue:
                    datavar = dst.createVariable(varname, vardatatype,
                                                 ('lat', 'lon'), zlib=True,
                                                 fill_value=fillvalue)
                else:
                    datavar = dst.createVariable(varname, vardatatype,
                                                 ('lat', 'lon'), zlib=True)

                datavar[:, :] = mx
                # write attrs of the variable datavar
                for attribute_name, attribute_value in band.GetMetadata().items():
                    if attribute_name.find(r'#') > -1:
                        tmp = attribute_name.split(r'#')
                        if tmp[0] == 'NC_GLOBAL':
                            dst.setncattr(tmp[1], attribute_value.rstrip('\n'))

                    else:
                        if attribute_name != '_FillValue':
                            if (
                                attribute_name == 'units'
                                and attribute_value == 'unitless'
                            ):
                                datavar.setncattr(attribute_name, '1')
                            else:
                                datavar.setncattr(
                                    attribute_name,
                                    attribute_value.rstrip('\n').replace('-', '_')
                                )

                datavar.grid_mapping = crs_name

                # add standard_name if there is no standard_name or long_name
                # attributes associated with datavar
                lst = [attr for attr in datavar.ncattrs()
                       if attr in ['standard_name', 'long_name']]

                if not lst:
                    datavar.standard_name = varname

                # add units attribute
                if 'units' not in datavar.ncattrs():
                    datavar.setncattr('units', '1')

                if gt[2] == 0.0 and gt[4] == 0.0:
                    datavar.coordinates = 'lon lat'

        with (
            OpenGDAL(infile) as input_geotiff,
            Dataset(outfile, mode='w', format='NETCDF4') as output_netcdf4
        ):
            # define global attributes
            output_netcdf4.title = ''
            output_netcdf4.institution = 'Alaska Satellite Facility'
            output_netcdf4.source = ''
            output_netcdf4.references = ''
            output_netcdf4.comment = ''
            output_netcdf4.history = f'{datetime.utcnow():%d/%m/%Y %H:%M:%S} (UTC)'
            output_netcdf4.GDAL = f'Version {gdal.__version__}'

            # create dimensions
            crs = parse_crs_from_ogc_wkt(input_geotiff.GetProjectionRef())
            if crs.cs_type == 'Projected':
                _process_projected(input_geotiff, output_netcdf4)
            else:
                _process_geogcs(input_geotiff, output_netcdf4)

            output_netcdf4.Conventions = 'CF-1.7'

        return outfile
