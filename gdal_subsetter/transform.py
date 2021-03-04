"""
 CLI for adapting a Harmony operation to GDAL

If you have harmony in a peer folder with this repo, then you can run the following for an example
python3 -m harmony_gdal --harmony-action invoke --harmony-input "$(cat ../harmony/example/service-operation.json)"
"""

import os
import subprocess
import re
import zipfile
import math
from tempfile import mkdtemp
import shutil
from affine import Affine
import glob
import json

from harmony import BaseHarmonyAdapter, util
from harmony.util import stage, bbox_to_geometry, download, generate_output_filename

import shapely
from shapely.geometry import shape, mapping, Point, MultiPoint, LineString, Polygon
from shapely.ops import cascaded_union
import pyproj
from osgeo import gdal, osr, ogr, gdal_array
from osgeo.gdalconst import *
from netCDF4 import Dataset
from pystac import Asset
import numpy as np
import geopandas as gpd
import fiona


mime_to_gdal = {
    "image/tiff": "GTiff",
    "image/png": "PNG",
    "image/gif": "GIF",
    "application/x-netcdf4": "NETCDF"
}

mime_to_extension = {
    "image/tiff": "tif",
    "image/png": "png",
    "image/gif": "gif",
    "application/x-netcdf4": "nc"
}

mime_to_options = {
    "image/tiff": ["-co", "COMPRESS=LZW"]
}

process_flags = {
    "subset":False,
    "maskband":False,
}

resampling_methods = [
    "nearest","bilinear","cubic","cubicspline","lanczos","average","rms","mode"
]


class ObjectView(object):
    """
    Simple class to make a dict look like an object.

    Example
    --------
        >>> o = ObjectView({ "key": "value" })
        >>> o.key
        'value'
    """

    def __init__(self, d):
        """
        Allows accessing the keys of dictionary d as though they
        are properties on an object

        Parameters
        ----------
        d : dict
            a dictionary whose keys we want to access as object properties
        """
        self.__dict__ = d


class HarmonyAdapter(BaseHarmonyAdapter):
    """
    See https://git.earthdata.nasa.gov/projects/HARMONY/repos/harmony-service-lib-py/browse
    for documentation and examples.
    """

    def get_version(self):
        with open("version.txt") as file_version:
            version = ','.join(file_version.readlines())

        return version

    def process_item(self, item, source):
        """
        Converts an input STAC Item's data into Zarr, returning an output STAC item

        Parameters
        ----------
        item : pystac.Item
            the item that should be converted
        source : harmony.message.Source
            the input source defining the variables, if any, to subset from the item

        Returns
        -------
        pystac.Item
            a STAC item containing the Zarr output
        """
        logger = self.logger
        message = self.message

        if message.subset and message.subset.shape:
            logger.warn('Ignoring subset request for user shapefile %s' %
                        (message.subset.shape.href,))

        layernames = []

        operations = dict(
            variable_subset=source.variables,
            is_regridded=bool(message.format.crs),
            is_subsetted=bool(message.subset and message.subset.bbox)
        )
        
        stac_record = item.clone()
        stac_record.assets = {}

        output_dir = mkdtemp()
        try:
            # Get the data file
            asset = next(v for k, v in item.assets.items()
                         if 'data' in (v.roles or []))

            if self.config.fallback_authn_enabled:
                input_filename = download(
                    asset.href,
                    output_dir,
                    logger=self.logger,
                    access_token=None,
                    cfg=self.config)
            else:
                input_filename = download(
                    asset.href,
                    output_dir,
                    logger=self.logger,
                    access_token=self.message.accessToken,
                    cfg=self.config)

            basename = os.path.splitext(os.path.basename(
                generate_output_filename(asset.href, **operations))
            )[0]

            file_type = self.get_filetype(input_filename)

            if file_type is None:
                return item
            elif file_type == 'tif':
                layernames, filename = self.process_geotiff(
                    source, basename, input_filename, output_dir, layernames
                )
            elif file_type == 'nc':
                layernames, filename = self.process_netcdf(
                    source, basename, input_filename, output_dir, layernames
                )
            elif file_type == 'zip':
                layernames, filename = self.process_zip(
                    source, basename, input_filename, output_dir, layernames
                )
            else:
                self.completed_with_error(
                    'No recognized file foarmat, not process'
                )

            #self.update_layernames(filename, [v for v in layernames])

            # Update metadata with bbox and extent in lon/lat coordinates for the geotiff file
            # Also update the STAC record
            stac_record.bbox = self.get_bbox_lonlat(filename)
            stac_record.geometry = bbox_to_geometry(stac_record.bbox)

            # Filename may change into the format other than geotiff
            filename = self.reformat(filename, output_dir)
            output_filename = basename + os.path.splitext(filename)[-1]
            mime = message.format.mime
            url = stage(
                filename,
                output_filename,
                mime,
                location=message.stagingLocation,
                logger=logger,
                cfg=self.config)

            stac_record.assets['data'] = Asset(
                url, title=output_filename, media_type=mime, roles=['data']
            )

            return stac_record

        finally:
            shutil.rmtree(output_dir)

    def process_geotiff(self, source, basename, input_filename, output_dir, layernames):
        if not source.variables:
            # process geotiff and all bands

            filename = input_filename
            layer_id = basename + '__all'
            band = None
            layer_id, filename, output_dir = self.combin_transfer(
                layer_id, filename, output_dir, band)

            #result = self.rename_to_result(
            #    layer_id, filename, output_dir)
            result = self.add_to_result(
                    layer_id,
                    filename,
                    output_dir
                )
 
            layernames.append(layer_id)
        else:

            # all possible variables (band names) in the input_filenanme

            variables = self.get_variables(input_filename)
            
            # source.process('variables') is user's requested variables (bands)
            
            for variable in source.process('variables'):

                band = None

                index = next(i for i, v in enumerate(
                    variables) if v.name.lower() == variable.name.lower())
                if index is None:
                    return self.completed_with_error('band not found: ' + variable)
                band = index + 1

                filename = input_filename

                layer_id = basename + '__' + variable.name

                layer_id, filename, output_dir = self.combin_transfer(
                    layer_id, filename, output_dir, band)

                result = self.add_to_result(
                    layer_id,
                    filename,
                    output_dir
                )
                layernames.append(layer_id)

        return layernames, result

    def process_netcdf(self, source, basename, input_filename, output_dir, layernames):
        variables = source.process(
            'variables') or self.get_variables(input_filename)

        for variable in variables:
            band = None

            # For non-geotiffs, we reference variables by appending a file path
            layer_format = self.read_layer_format(
                source.collection,
                input_filename,
                variable.name
            )
            filename = layer_format.format(
                input_filename)

            layer_id = basename + '__' + variable.name

            # convert the subdataset in the nc file into the geotif file
            filename = self.nc2tiff(layer_id, filename, output_dir)

            layer_id, filename, output_dir = self.combin_transfer(
                layer_id, filename, output_dir, band)

            result = self.add_to_result(
                layer_id,
                filename,
                output_dir
            )
            layernames.append(layer_id)

        return layernames, result

    def process_zip(self, source, basename, input_filename, output_dir, layernames):
        [tiffile, ncfile] = self.pack_zipfile(input_filename, output_dir)

        if tiffile:
            layernames, result = self.process_geotiff(
                source, basename, tiffile, output_dir, layernames)

        if ncfile:
            layernames, result = self.process_netcdf(
                source, basename, ncfile, output_dir, layernames)

        return layernames, result

    def update_layernames(self, filename, layernames):
        """
        Updates the layers in the given file to match the list of layernames provided

        Parameters
        ----------
        filename : string
            The path to file whose layernames should be updated
        layernames : string[]
            An array of names, in order, to apply to the layers
        """
        ds = gdal.Open(filename)

        for count, layer_name in enumerate(layernames):
            ds.GetRasterBand(count + 1).SetDescription(layer_name)

        ds = None

    def prepare_output_dir(self, output_dir):
        """
        Deletes (if present) and recreates the given output_dir, ensuring it exists
        and is empty

        Parameters
        ----------
        output_dir : string
            the directory to delete and recreate
        """
        self.cmd2('rm', '-rf', output_dir)
        self.cmd2('mkdir', '-p', output_dir)

    def cmd2(self, *args):
        '''
        This is used to execute the OS commands, such as cp, mv, and rm.
        '''
        self.logger.info(
            args[0] + " " + " ".join(["'{}'".format(arg) for arg in args[1:]])
        )
        #result_str = subprocess.check_output(args).decode("utf-8")
        command = " ".join(args)
        exit_code=os.system(command)
        if exit_code == 0:
            exit_state = "successfully"
        else:
            exit_state = "failed"

        return "{command} executes {exit_state}.".format(command=command, exit_state=exit_state)

    def cmd(self, *args):
        '''
        This is used to execuate gdal* commands.
        '''
        self.logger.info(
            args[0] + " " + " ".join(["'{}'".format(arg) for arg in args[1:]])
        )
        result_str = subprocess.check_output(args).decode("utf-8")

        return result_str.split("\n")

    def is_rotated_geotransform(self, srcfile):
        dataset = gdal.Open(srcfile)
        gt = dataset.GetGeoTransform()
        check = False

        if gt[2] != 0.0 or gt[4] != 0:
            check = True

        return check

    def nc2tiff(self, layerid, filename, dstdir):

        def search(myDict, lookupkey):
            for key, value in myDict.items():
                if lookupkey in key:
                    return myDict[key]

        normalized_layerid = layerid.replace('/', '_')
        dstfile = "%s/%s" % (dstdir, normalized_layerid + '__nc2tiff.tif')
        ds = gdal.Open(filename)
        metadata = ds.GetMetadata()
        crs_wkt = search(metadata, "crs_wkt")

        if crs_wkt:
            command = ['gdal_translate', '-a_srs']
            command.extend([crs_wkt])
            command.extend([filename, dstfile])
            self.cmd(*command)
            return dstfile

        return filename

    def varsubset(self, srcfile, dstfile, band=None):
        if not band:
            return srcfile

        command = ['gdal_translate']
        command.extend(['-b', '%s' % (band)])
        command.extend([srcfile, dstfile])
        self.cmd(*command)

        return dstfile

    def subset(self, layerid, srcfile, dstdir, band=None):
        normalized_layerid = layerid.replace('/', '_')
        subset = self.message.subset

        if subset.bbox is None and subset.shape is None:
            dstfile = "%s/%s" % (dstdir, normalized_layerid +
                                 '__varsubsetted.tif')
            dstfile = self.varsubset(srcfile, dstfile, band=band)

            return dstfile

        if subset.bbox:
            [left, bottom, right, top] = self.get_bbox(srcfile)

            # subset.bbox is defined as [left/west,low/south,right/east,upper/north]
            subsetbbox = subset.process('bbox')

            [b0, b1], transform = self.lonlat2projcoord(
                srcfile, subsetbbox[0], subsetbbox[1])
            [b2, b3], transform = self.lonlat2projcoord(
                srcfile, subsetbbox[2], subsetbbox[3])

            if any(x is None for x in [b0, b1, b2, b3]):
                dstfile = "%s/%s" % (dstdir,
                                     normalized_layerid + '__varsubsetted.tif')
                dstfile = self.varsubset(srcfile, dstfile, band=band)
            elif b0 < left and b1 < bottom and b2 > right and b3 > top:
                # user's input subset totally covers the image bbox, do not do subset
                dstfile = "%s/%s" % (dstdir,
                                     normalized_layerid + '__varsubsetted.tif')
                dstfile = self.varsubset(srcfile, dstfile, band=band)
            else:
                dstfile = "%s/%s" % (dstdir,
                                     normalized_layerid + '__subsetted.tif')
                dstfile = self.subset2(
                    srcfile, dstfile, bbox=subsetbbox, band=band)

            return dstfile

        if subset.shape:
            # need know what the subset.shape is. Is it a file? do we need pre-process like we did for subset.bbox?
            dstfile = "%s/%s" % (dstdir,
                                 normalized_layerid + '__subsetted.tif')
            # get the shapefile
            shapefile = self.get_shapefile(subset.shape, dstdir)
            dstfile = self.subset2(
                srcfile, dstfile, shapefile=shapefile, band=band)

            return dstfile

    def convert2multipolygon(self, infile, outfile, buf=None):
        '''
        convert point or line feature geojson file to multi-polygon feature geojson file
        input:  infile - point or line feature geojson file name
        return: outfile - multi-polygon feature ESRI shapefile directory name
        '''
        if not buf:
            return infile

        fd_infile = fiona.open(infile)

        # get feature type of infile
        featype = fd_infile.schema.get('geometry')

        # prepare meta for polygon file
        meta = fd_infile.meta
        meta['schema']['geometry'] = 'Polygon'
        meta['schema']['properties'] = {'id': 'int'}
        meta['driver'] = 'GeoJSON'

        with fiona.open(outfile, 'w', **meta) as fd_outfile:
            poly_lst = []

            for index_point, point in enumerate(fd_infile):
                pt = shape(point['geometry'])
                polygon = pt.buffer(buf)
                poly_lst.append(polygon)

            polygons = cascaded_union(poly_lst)

            if polygons.geometryType() == 'Polygon':
                fd_outfile.write({
                    'geometry': mapping(polygons),
                    'properties': {'id': 0},
                })
            else:
                for index_polygon, polygon in enumerate(polygons):
                    fd_outfile.write({
                        'geometry': mapping(polygon),
                        'properties': {'id': index_polygon},
                    })

        return outfile

    def get_coord_unit(self, geojsonfile):
        try:
            # get unit of the feature in the shapefile
            fd_infile = fiona.open(geojsonfile)
            geometry = fd_infile.schema.get('geometry')
            proj = pyproj.crs.CRS(fd_infile.crs_wkt)
            proj_json = json.loads(proj.to_json())
            unit = proj_json['coordinate_system']['axis'][0]['unit']
        except Exception:
            unit = None

        return geometry, unit

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
        shapefile = util.download(href, dstdir)

        # get unit of the feature in the shapefile
        geometry, unit = self.get_coord_unit(shapefile)

        # convert into multi-polygon feature file
        fileprex = os.path.splitext(shapefile)[0]
        tmpfile_geojson = fileprex+"_tmp.geojson"

        # check BUFFER environment to get buf from message passed by harmony
        buffer_string = os.environ.get('BUFFER')
        if buffer_string:
            buffer_dic = eval(buffer_string)

            if unit and 'Polygon' not in geometry:
                if "degree" in unit.lower():
                    buf = buffer_dic["degree"]
                else:
                    buf = buffer_dic["meter"]
            else:
                buf = None
        else:
            buf = None

        # convert to a new multipolygon file
        tmpfile_geojson = self.convert2multipolygon(
            shapefile, tmpfile_geojson, buf=buf)

        # convert into ESRI shapefile
        outfile = fileprex+".shp"
        command = ['ogr2ogr', '-f', 'ESRI Shapefile']
        command.extend([outfile, tmpfile_geojson])
        self.cmd(*command)

        return outfile

    def reproject(self, layerid, srcfile, dstdir):
        crs = self.message.format.process('crs')
        interpolation = self.message.format.process('interpolation')
        if interpolation in resampling_methods:
            resample_method = interpolation
        else:
            resample_method = "bilinear"
        if not crs:
            return srcfile

        normalized_layerid = layerid.replace('/', '_')
        dstfile = "%s/%s" % (dstdir, normalized_layerid + '__reprojected.tif')
        self.cmd('gdalwarp', "-t_srs", crs, "-r", resample_method, "-overwrite", srcfile, dstfile)

        return dstfile

    def resize(self, layerid, srcfile, dstdir):
        interpolation = self.message.format.process('interpolation')
        if interpolation in resampling_methods:
            resample_method = interpolation
        else:
            resample_method = "bilinear"

        command = ['gdal_translate']
        fmt = self.message.format
        normalized_layerid = layerid.replace('/', '_')
        dstfile = "%s/%s__resized.tif" % (dstdir, normalized_layerid)

        if fmt.width or fmt.height:
            width = fmt.process('width') or 0
            height = fmt.process('height') or 0
            command.extend(["-outsize", str(width), str(height)])
        command.extend(['-r', resample_method])
        command.extend([srcfile, dstfile])
        self.cmd(*command)

        return dstfile

    def add_to_result(self, layerid, srcfile, dstdir):
        tmpfile = "%s/tmp-result.tif" % (dstdir)
        dstfile = "%s/result.tif" % (dstdir)
        if not os.path.exists(dstfile):
            filelist =[srcfile]
        else:
            filelist = [dstfile, srcfile]
        
        tmpfile = self.stack_multi_file_with_metadata(filelist, tmpfile)
        self.cmd2('cp', '-f', tmpfile, dstfile)
        return dstfile

    
    def stack_multi_file_with_metadata(self, infilelist, outfile):
        '''
        The infilelist includes multiple files, each file does may not have the same number of bands,
        but they must have the same peojection and geogransform.
        '''
        collection=[]
        count=0
        ds_description = []
        
        for id, layer in enumerate(infilelist, start=1):
            ds = gdal.Open(layer)
            #filestr = os.path.splitext(os.path.basename(layer))[0]
            filename = ds.GetDescription()
            filestr = os.path.splitext(os.path.basename(filename))[0]
            if id == 1:
                proj=ds.GetProjection()
                geot=ds.GetGeoTransform()
                cols=ds.RasterXSize
                rows=ds.RasterYSize
                gtyp = ds.GetRasterBand(1).DataType  

            bandnum=ds.RasterCount
            md=ds.GetMetadata()
            for i in range(1,bandnum+1):
                band=ds.GetRasterBand(i)
                bmd=band.GetMetadata()
                bmd.update(md)
                data=band.ReadAsArray()
                nodata = band.GetNoDataValue() 
                mask = band.GetMaskBand().ReadAsArray()
                count = count + 1
                if "standard_name" in bmd.keys():
                    band_name = "Band{count}:{standard_name}".format(
                    count=count, standard_name=bmd["standard_name"])
                    tmp_bmd = {"bandname": band_name}
                else:
                    band_name = "Band{count}:{filestr}-{i}".format(
                    count=count, filestr=filestr, i=i)
                    tmp_bmd ={"bandname": band_name,"standard_name": band_name, "long_name": band_name}

                bmd.update(tmp_bmd)
                ds_description.append(band_name)
                collection.append({"band_sn":count,"band_md":bmd, "band_array":data, "mask_array":mask, "nodata":nodata})
            ds = None

        dst_ds = gdal.GetDriverByName('GTiff').Create(outfile, cols, rows, count, gtyp)
        #maskband is readonly by default, you have to create the maskband before you can write the maskband
        gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
        dst_ds.CreateMaskBand(gdal.GMF_PER_DATASET)
        dst_ds.SetProjection(proj)
        dst_ds.SetGeoTransform(geot)
        for i, band in enumerate(collection):
            dst_ds.GetRasterBand(i+1).WriteArray(collection[i]["band_array"])
            dst_ds.GetRasterBand(i+1).SetMetadata(collection[i]["band_md"])
            dst_ds.GetRasterBand(i+1).SetDescription('')
            dst_ds.GetRasterBand(i+1).GetMaskBand().WriteArray(collection[i]["mask_array"])      
            if collection[i]["nodata"]:
                dst_ds.GetRasterBand(i+1).SetNoDataValue(collection[i]["nodata"])

        dst_ds.FlushCache()                     # write to disk
        dst_ds = None
        return outfile

    def stack_multi_file(self, infilelist, outfile):
        '''
        The infilelist includes multiple files, each file may not have the same number of bands,
        but they must have the same peojection and geogransform.
        '''
        collection=[]
        count=0
        ds_description = []
        
        for id, layer in enumerate(infilelist, start=1):
            ds = gdal.Open(layer)
            #filestr = os.path.splitext(os.path.basename(layer))[0]
            filename = ds.GetDescription()
            filestr = os.path.splitext(os.path.basename(filename))[0]
            if id == 1:
                proj=ds.GetProjection()
                geot=ds.GetGeoTransform()
                cols=ds.RasterXSize
                rows=ds.RasterYSize
                gtyp = ds.GetRasterBand(1).DataType
                
            bandnum=ds.RasterCount
            md=ds.GetMetadata()
            for i in range(1,bandnum+1):
                band=ds.GetRasterBand(i)
                bmd=band.GetMetadata()
                bmd.update(md)
                data=band.ReadAsArray()
                nodata = band.GetNoDataValue() 
                mask = band.GetMaskBand().ReadAsArray()
                count = count + 1
                band_name = "{filestr}-band{i}".format(count=count, filestr=filestr, i=i)
                long_name = band_name
                standard_name = band_name
                tmp_bmd = {"long_name": long_name, "standard_name":standard_name}
                bmd.update(tmp_bmd)
                ds_description.append(band_name)
                collection.append({"band_sn":count,"band_md":bmd, "band_array":data, "mask_array":mask, "nodata":nodata})
            ds = None

        dst_ds = gdal.GetDriverByName('GTiff').Create(outfile, cols, rows, count, gtyp)
        #maskband is readonly by default, you have to create the maskband before you can write the maskband
        gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
        dst_ds.CreateMaskBand(gdal.GMF_PER_DATASET)
        dst_ds.SetProjection(proj)
        dst_ds.SetGeoTransform(geot)
        for i, band in enumerate(collection):
            dst_ds.GetRasterBand(i+1).WriteArray(collection[i]["band_array"])
            dst_ds.GetRasterBand(i+1).SetMetadata(collection[i]["band_md"])
            dst_ds.GetRasterBand(i+1).GetMaskBand().WriteArray(collection[i]["mask_array"])      
            if collection[i]["nodata"]:
                dst_ds.GetRasterBand(i+1).SetNoDataValue(collection[i]["nodata"])

        dst_ds.FlushCache()                     # write to disk
        dst_ds = None
        return outfile


    def rename_to_result(self, layerid, srcfile, dstdir):
        dstfile = "%s/result.tif" % (dstdir)

        if not os.path.exists(dstfile):
            self.cmd2('mv', srcfile, dstfile)

        return dstfile

    def reformat(self, srcfile, dstdir):
        gdal_subsetter_version = "gdal_subsetter_version={gdal_subsetter_ver}".format(
            gdal_subsetter_ver=self.get_version())
        command = ['gdal_edit.py',  '-mo', gdal_subsetter_version, srcfile]
        self.cmd(*command)
        output_mime = self.message.format.process('mime')
        if output_mime not in mime_to_gdal:
            raise Exception('Unrecognized output format: ' + output_mime)
        if output_mime == "image/tiff":
            return srcfile
        dstfile = "%s/translated.%s" % (dstdir, mime_to_extension[output_mime])
        if output_mime == "application/x-netcdf4":
            ds = gdal.Open(srcfile)
            nodata_flag = ds.GetRasterBand(1).GetNoDataValue()
            ds = None
            dstfile = self.geotiff2netcdf_combined(srcfile, dstfile, nodata_flag)
        else:
            command = ['gdal_translate',
                       '-of', mime_to_gdal[output_mime],
                       '-scale',
                       srcfile, dstfile]
            self.cmd(*command)
        return dstfile

    def read_layer_format(self, collection, filename, layer_id):
        gdalinfo_lines = self.cmd("gdalinfo", filename)

        layer_line = next(
            filter(
                (lambda line: re.search(":*" + layer_id+"$", line)),
                gdalinfo_lines
            ),
            None
        )

        if layer_line is None:
            print('Invalid Layer:', layer_id)

        layer = layer_line.split("=")[-1]

        return layer.replace(filename, "{}")

    def get_variables(self, filename):
        gdalinfo_lines = self.cmd("gdalinfo", filename)
        result = []

        # Normal case of NetCDF / HDF, where variables are subdatasets
        for subdataset in filter((lambda line: re.match(r"^\s*SUBDATASET_\d+_NAME=", line)), gdalinfo_lines):
            result.append(ObjectView({"name": re.split(r":", subdataset)[-1]}))

        if result:
            return result

        # GeoTIFFs, where variables are bands, with descriptions set to their variable name
        for subdataset in filter((lambda line: re.match(r"^\s*Description = ", line)), gdalinfo_lines):
            result.append(ObjectView(
                {"name": re.split(r" = ", subdataset)[-1]}
            ))

        if result:
            return result

        # Some GeoTIFFdoes not have descriptions. directly use Band # as the variables
        for subdataset in filter((lambda line: re.match(r"^Band", line)), gdalinfo_lines):
            tmpline = re.split(r" ", subdataset)
            result.append(ObjectView(
                {"name": tmpline[0].strip()+tmpline[1].strip(), }))

        if result:
            return result

    def get_filetype(self, filename):
        if filename is None:
            return None

        if not os.path.exists(filename):
            return None

        file_basenamewithpath, file_extension = os.path.splitext(filename)

        if file_extension in ['.nc']:
            return 'nc'
        elif file_extension in ['.tif', '.tiff']:
            return 'tif'
        elif file_extension in ['.zip']:
            return 'zip'

        return 'others'

    def is_geotiff(self, filename):
        gdalinfo_lines = self.cmd("gdalinfo", filename)

        return gdalinfo_lines[0] == "Driver: GTiff/GeoTIFF"

    def combin_transfer(self, layer_id, filename, output_dir, band):
        filename = self.subset(
            layer_id,
            filename,
            output_dir,
            band
        )
        filename = self.reproject(
            layer_id,
            filename,
            output_dir
        )
        filename = self.resize(
            layer_id,
            filename,
            output_dir
        )
        return layer_id, filename, output_dir

    def get_bbox(self, filename):
        """
        input: the geotif file
        return: bbox[left,low,right,upper] of the file
        """
        ds = gdal.Open(filename)
        gt = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        ul_x, ul_y = self.calc_ij_coord(gt, 0, 0)
        ur_x, ur_y = self.calc_ij_coord(gt, cols, 0)
        lr_x, lr_y = self.calc_ij_coord(gt, cols, rows)
        ll_x, ll_y = self.calc_ij_coord(gt, 0, rows)

        return [min(ul_x, ll_x), min(ll_y, lr_y), max(lr_x, ur_x), max(ul_y, ur_y)]

    def get_bbox_lonlat(self, filename):
        """
        get the bbox in longitude and latitude of the raster file, and update the bbox and extent for the file, and return bbox.
        input:raster file name
        output:bbox of the raster file
        """
        ds = gdal.Open(filename, gdal.GA_Update)
        gt = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        ul_x, ul_y = self.calc_ij_coord(gt, 0, 0)
        ur_x, ur_y = self.calc_ij_coord(gt, cols, 0)
        lr_x, lr_y = self.calc_ij_coord(gt, cols, rows)
        ll_x, ll_y = self.calc_ij_coord(gt, 0, rows)

        projection = ds.GetProjection()
        dst = osr.SpatialReference(projection)
        dstproj4 = dst.ExportToProj4()
        ct2 = pyproj.Proj(dstproj4)

        ul_x2, ul_y2 = ct2(ul_x, ul_y, inverse=True)
        ur_x2, ur_y2 = ct2(ur_x, ur_y, inverse=True)
        lr_x2, lr_y2 = ct2(lr_x, lr_y, inverse=True)
        ll_x2, ll_y2 = ct2(ll_x, ll_y, inverse=True)

        ul_x2 = float("{:.7f}".format(ul_x2))
        ul_y2 = float("{:.7f}".format(ul_y2))

        ur_x2 = float("{:.7f}".format(ur_x2))
        ur_y2 = float("{:.7f}".format(ur_y2))

        lr_x2 = float("{:.7f}".format(lr_x2))
        lr_y2 = float("{:.7f}".format(lr_y2))

        ll_x2 = float("{:.7f}".format(ll_x2))
        ll_y2 = float("{:.7f}".format(ll_y2))

        lon_left = min(ul_x2, ll_x2)
        lat_low = min(ll_y2, lr_y2)
        lon_right = max(lr_x2, ur_x2)
        lat_high = max(ul_y2, ur_y2)

        # write bbox and extent in lon/lat unit to the metadata of the filename
        md = ds.GetMetadata()
        bbox = [lon_left, lat_low, lon_right, lat_high]
        extent = {'ul': [ul_x2, ul_y2], 'll': [ll_x2, ll_y2],
                  'ur': [ur_x2, ur_y2], 'lr': [lr_x2, lr_y2]}
        md['bbox'] = str(bbox)
        md['extent'] = str(extent)
        ds.SetMetadata(md)
        ds = None

        return bbox

    def pack_zipfile(self, zipfilename, output_dir, variables=None):
        with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
            zip_ref.extractall(output_dir+'/unzip')

        tmptif = None

        filelist_tif = self.get_file_from_unzipfiles(
            f'{output_dir}/unzip', 'tif', variables)

        if filelist_tif:
            tmpfile = f'{output_dir}/tmpfile'
            # stack the single-band files into a multiple-band file
            tmptif = self.stack_multi_file(filelist_tif, tmpfile)

        tmpnc = None
        filelist_nc = self.get_file_from_unzipfiles(f'{output_dir}/unzip', 'nc')

        if filelist_nc:
            tmpnc = filelist_nc

        return tmptif, tmpnc

    def get_file_from_unzipfiles(self, extract_dir, filetype, variables=None):
        tmpexp = f'{extract_dir}/*.{filetype}'

        filelist = sorted(glob.glob(tmpexp))

        if filelist:
            ch_filelist = []

            if variables:
                for variable in variables:
                    variable_raw = fr"{variable}"
                    m = re.search(variable, filelist)

                    if m:
                        ch_filelist.append(m.string)

                filelist = ch_filelist

        return filelist

    def lonlat2projcoord(self, srcfile, lon, lat):
        dataset = gdal.Open(srcfile)
        transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        dst = osr.SpatialReference(projection)
        dstproj4 = dst.ExportToProj4()
        ct2 = pyproj.Proj(dstproj4)
        xy = ct2(lon, lat)

        if math.isinf(xy[0]) or math.isinf(xy[1]):
            xy = [None, None]

        return [xy[0], xy[1]], transform

    def projcoord2lonlat(self, srcfile, x, y):
        """
        covert x,y of dataset's projected coord to longitude and latitude
        """
        dataset = gdal.Open(srcfile)
        transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        dst = osr.SpatialReference(projection)
        dstproj4 = dst.ExportToProj4()
        ct2 = pyproj.Proj(dstproj4)
        lon, lat = ct2(x, y, inverse=True)

        return [lon, lat], transform

    def subset2(self, tiffile, outputfile, bbox=None, band=None, shapefile=None):
        """
        subset tiffile with bbox or shapefile. bbox ans shapefile are exclusive.
        inputs: tiffile-geotif file
                outputfile-subsetted file name
                bbox - [left,low,right,upper] in lon/lat coordinates
                shapefile - a shapefile directory in which multiple files exist
        """
        process_flags["subset"] = True
        # covert to absolute path
        tiffile = os.path.abspath(tiffile)
        outputfile = os.path.abspath(outputfile)
        RasterFormat = 'GTiff'
        ref_ds = gdal.Open(tiffile)
        ref_band = ref_ds.GetRasterBand(1)
        ref_nodata = ref_band.GetNoDataValue()
        tmpfile = f'{os.path.splitext(outputfile)[0]}-tmp.tif'
        if bbox or shapefile:
            if bbox:
                shapefile_out = self.box2shapefile(tiffile, bbox)
                boxproj, proj = self.boxwrs84_boxproj(bbox, ref_ds)
            else:
                shapefile_out = f'{os.path.dirname(outputfile)}/tmpshapefile'
                boxproj, proj, shapefile_out, geometryname = self.shapefile_boxproj(
                shapefile, ref_ds, shapefile_out)

            ul_x, ul_y, ul_i, ul_j, cols, rows = self.calc_subset_envelopwindow(
                ref_ds, boxproj)
            command = ['gdal_translate']

            if band:
                command.extend(['-b', '%s' % (band)])

            command.extend(
                ['-srcwin', str(ul_i), str(ul_j), str(cols), str(rows)])
            command.extend([tiffile, tmpfile])
            self.cmd(*command)
            self.mask_via_combined(tmpfile, shapefile_out, outputfile)
        else:
            self.cmd2(*['cp', tiffile, outputfile])

        return outputfile

    def boxwrs84_boxproj(self, boxwrs84, ref_ds):
        """
        convert the box define in lon/lat to box in projection coordinates defined in red_ds
        inputs: boxwrs84, which is defined as [left,low,right,upper] in lon/lat,
                ref_ds is reference dataset
        returns:boxprj, which is also defined as {"llxy":llxy, "lrxy":lrxy, "urxy":urxy, "ulxy":ulxy},
                where llxy,lrxy, urxy, and ulxy are coordinate pairs in projection
                projection, which is the projection of ref_ds
        """
        projection = ref_ds.GetProjection()
        dst = osr.SpatialReference(projection)

        # get coordinates of four corners of the boxwrs84
        ll_lon, ll_lat = boxwrs84[0], boxwrs84[1]
        lr_lon, lr_lat = boxwrs84[2], boxwrs84[1]
        ur_lon, ur_lat = boxwrs84[2], boxwrs84[3]
        ul_lon, ul_lat = boxwrs84[0], boxwrs84[3]

        # convert all four corners
        dstproj4 = dst.ExportToProj4()
        ct = pyproj.Proj(dstproj4)
        llxy = ct(ll_lon, ll_lat)
        lrxy = ct(lr_lon, lr_lat)
        urxy = ct(ur_lon, ur_lat)
        ulxy = ct(ul_lon, ul_lat)

        boxproj = {"llxy": llxy, "lrxy": lrxy, "urxy": urxy, "ulxy": ulxy}

        return boxproj, projection

    def calc_coord_ij(self, gt, x, y):
        transform = Affine.from_gdal(*gt)
        rev_transform = ~transform
        cols, rows = rev_transform*(x, y)

        return int(cols), int(rows)

    def calc_subset_envelopwindow(self, ds, box, delt=0):
        """
        inputs: ds -- the reference dataset
                box -- which defined as {"llxy":llxy, "lrxy":lrxy, "urxy":urxy, "ulxy":ulxy},
                where llxy,lrxy, urxy, and ulxy are coordinate pairs in projection
                delt -- the number of deltax and dletay to extend the subsetting array which
                represets the box
        returns:ul_x,ul_y,ul_i,ul_j,cols,rows
        """

        # get 4 conners coordinate values in projection coorndinates
        ul = box.get("ulxy")
        ur = box.get("urxy")
        ll = box.get("llxy")
        lr = box.get("lrxy")

        # get i,j coordinates in the array of 4 conners of the box
        gt = ds.GetGeoTransform()
        ul_i, ul_j = self.calc_coord_ij(gt, ul[0], ul[1])
        ur_i, ur_j = self.calc_coord_ij(gt, ur[0], ur[1])
        ll_i, ll_j = self.calc_coord_ij(gt, ll[0], ll[1])
        lr_i, lr_j = self.calc_coord_ij(gt, lr[0], lr[1])

        # adjust box in array coordinates
        ul_i = ul_i - delt
        ul_j = ul_j - delt
        ur_i = ur_i + delt
        ur_j = ur_j - delt
        lr_i = lr_i + delt
        lr_j = lr_j + delt
        ll_i = ll_i - delt
        ll_j = ll_j + delt

        # get the envelop of the box in array coordinates
        ul_i = min(ul_i, ur_i, ll_i, lr_i)
        ul_j = min(ul_j, ur_j, ll_j, lr_j)
        lr_i = max(ul_i, ur_i, ll_i, lr_i)
        lr_j = max(ul_j, ur_j, ll_j, lr_j)

        # get the intersection between box and image in row, col coordinator
        cols_img = ds.RasterXSize
        rows_img = ds.RasterYSize
        ul_i = max(0, ul_i)
        ul_j = max(0, ul_j)
        lr_i = min(cols_img, lr_i)
        lr_j = min(rows_img, lr_j)
        cols = lr_i-ul_i
        rows = lr_j-ul_j
        ul_x, ul_y = self.calc_ij_coord(gt, ul_i, ul_j)

        return ul_x, ul_y, ul_i, ul_j, cols, rows

    def calc_ij_coord(self, gt, col, row):
        transform = Affine.from_gdal(*gt)
        x, y = transform * (col, row)

        return x, y

    def create_shapefile_with_box(self, box, projection, shapefile):
        """
         input: box {ll, lr, ur, ul} in projection cordinates,
         where ll=(ll_lon,ll_lat),lr=(lr_lon,lr_lat), ur=(ur_lon, ur_lat), ul=(ul_lon, ul_lat)
        """

        # output: polygon geometry
        llxy = box.get("llxy")
        lrxy = box.get("lrxy")
        urxy = box.get("urxy")
        ulxy = box.get("ulxy")
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(llxy[0], llxy[1])
        ring.AddPoint(lrxy[0], lrxy[1])
        ring.AddPoint(urxy[0], urxy[1])
        ring.AddPoint(ulxy[0], ulxy[1])
        ring.AddPoint(llxy[0], llxy[1])
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)

        # create output file
        outDriver = ogr.GetDriverByName('ESRI Shapefile')

        if os.path.exists(shapefile):
            if os.path.isfile(shapefile):
                os.remove(shapefile)
            else:
                shutil.rmtree(shapefile)

        outDataSource = outDriver.CreateDataSource(shapefile)
        outSpatialRef = osr.SpatialReference(projection)
        outLayer = outDataSource.CreateLayer(
            'boundingbox', outSpatialRef, geom_type=ogr.wkbPolygon)
        featureDefn = outLayer.GetLayerDefn()

        # add new geom to layer
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(polygon)
        outLayer.CreateFeature(outFeature)
        outFeature.Destroy()
        outDataSource.Destroy()

    def box2shapefile(self, inputfile, box):
        """
        inputs:inputfile is the geotiff file, box[minlon,minlat,maxlon,maxlat] is in lon/lat.
        return: shapefile.
        """

        in_ds = gdal.Open(inputfile)
        in_gt = in_ds.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(in_gt)
        boxproj, proj = self.boxwrs84_boxproj(box, in_ds)

        if inv_gt is None:
            raise RuntimeError('Inverse geotransform failed')

        inputdir = os.path.dirname(inputfile)
        basename = os.path.basename(inputfile).split(".")[0]
        shapefile = f'{inputdir}/{basename}-shapefile'

        if os.path.isfile(shapefile) or os.path.isdir(shapefile):
            self.cmd2(*['rm', '-rf', shapefile])

        self.create_shapefile_with_box(boxproj, proj, shapefile)

        return shapefile

    def rasterize(self, infile, shapefile, outfile):
        """
        inputs: infile- geotiff file
                shaepfile - mask the rasterfile with this shapefile
                outfile - output file name
        return: outfile
        """
        infile = os.path.abspath(infile)

        # copy infile to tmpfile
        self.cmd2(*['cp', '-f', infile, outfile])

        # get the infomation of outfile
        ds = gdal.Open(outfile, gdal.GA_Update)
        num = ds.RasterCount
        bands = list(range(1, num+1))
        burns = list(np.full(num, 0))

        # get the information of shapefile
        shp = ogr.Open(shapefile)
        ly = shp.GetLayerByIndex(0)
        lyname = ly.GetName()

        # rasterize the shapefile according to infile
        err = gdal.RasterizeLayer(
            ds,
            bands,
            ly,
            burn_values=burns
        )
        ds.FlushCache()
        ds = None

        # tested options initValues and Inverse, both are not work
        # options=["ATTRIBUTE={lyname}".format(lyname=lyname)]
        # need use following way to process the pixels outside the box
        # update the bands of ds
        in_ds = gdal.Open(infile, gdal.GA_ReadOnly)
        in_data = in_ds.ReadAsArray()
        ds = gdal.Open(outfile, gdal.GA_Update)
        out_data = ds.ReadAsArray()
        rst_data = in_data-out_data

        if num == 1:
            out_band = ds.GetRasterBand(1)
            out_band.WriteArray(rst_data)
        else:
            for i in range(1, num+1):
                out_band = ds.GetRasterBand(i)
                out_band.WriteArray(rst_data[i-1, :, :])

        ds.FlushCache()
        in_ds, ds = None, None

    def shapefile_boxproj(self, shapefile, ref_ds, outputfile):
        """
        convert shapefile and calculate the envelop box in the projection defined in ref_ds
        inputs:  shapefile - used to define the AOI
                 ref_ds - dataset associate with the reference geotiff file
                 outputfile - output shapefile anme
        returns: boxproj - extent of the outputfile
                 ref_proj - projection of the ref_ds
                 outputfile - output shapefile name
                 geometryname - geometry name of the features in the outputfile
        """

        ref_proj = ref_ds.GetProjection()
        tmp = gpd.GeoDataFrame.from_file(shapefile)
        tmpproj = tmp.to_crs(ref_proj)
        tmpproj.to_file(outputfile)
        shp = ogr.Open(outputfile)
        lyr = shp.GetLayer()
        lyrextent = lyr.GetExtent()
        feature = lyr.GetNextFeature()
        geometry = feature.GetGeometryRef()
        geometryname = geometry.GetGeometryName()

        # Extent[lon_min,lon_max,lat_min,lat_max]
        #boxproj={"llxy":llxy, "lrxy":lrxy, "urxy":urxy, "ulxy":ulxy}
        # where llxy,lrxy, urxy, and ulxy are coordinate pairs in projection
        llxy = (lyrextent[0], lyrextent[2])
        lrxy = (lyrextent[1], lyrextent[2])
        urxy = (lyrextent[1], lyrextent[3])
        ulxy = (lyrextent[0], lyrextent[3])

        boxproj = {"llxy": llxy, "lrxy": lrxy, "urxy": urxy, "ulxy": ulxy}

        return boxproj, ref_proj, outputfile, geometryname

    def transfer_metadata(self, infile, inband, outfile, outband):
        """
        readout the file metadata and band metadata from infile inband, write these metadata to outfile outband
        input: infile
               inband - band number in the infile, 1,2,3, etc.
               outfile
               outband -band nuber in outfile 1,2,3, etc.
        """
        in_ds = gdal.Open(infile, gdal.GA_ReadOnly)
        in_md = in_ds.GetMetadata()
        in_b = in_ds.GetRasterBand(inband)
        in_nodata = in_b.GetNoDataValue()
        in_bmd = in_b.GetMetadata()
        out_ds = gdal.Open(outfile, gdal.GA_Update)
        out_md = out_ds.GetMetadata()
        out_md.update(in_md)
        out_ds.SetMetadata(out_md)
        out_b = out_ds.GetRasterBand(outband)
        out_bmd = out_b.GetMetadata()
        out_bmd.update(in_bmd)
        out_b.SetMetadata(out_bmd)
        if in_nodata:
            out_b.SetNoDataValue(in_nodata)
        else:
            out_b.DeleteNoDataValue()

         # write to disk
        out_ds.FlushCache()
        out_ds = None

 
    def mask_via_combined(self, inputfile, shapefile, outputfile):
        """
        It calcualtes the maskbands and set databands with nodata value for each band according to the shapefile.
        inputs: inputfile is geotiff file.
                shapefile int same coordinate reference as the inputfile.
        return: outputfile is masked geotiff file.
        """
        
        # define temorary file name
        tmpfilepre = os.path.splitext(inputfile)[0]
        tmpfile = "{tmpfilepre}-tmp.tif".format(tmpfilepre=tmpfilepre)
        self.cmd2(*['cp', '-f', inputfile, tmpfile])
        #cmd = " ".join(['cp', '-f', inputfile, tmpfile])
        #os.system(cmd)

        self.cmd2(*['cp', '-f', inputfile, outputfile])
        #cmd = " ".join(['cp', '-f', inputfile, outputfile])
        #os.system(cmd)

        # read shapefile info
        shp = ogr.Open(shapefile)
        ly = shp.GetLayerByIndex(0)
        lyname = ly.GetName()
        # update the outputfile
        dst_ds = gdal.Open(outputfile, GA_Update)
        gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
        # this cahnges the maskband data in the dst_ds
        dst_ds.CreateMaskBand(gdal.GMF_PER_DATASET)

        # update tmpfile (used as a mask file)
        tmp_ds = gdal.Open(tmpfile, gdal.GA_Update)
        num = tmp_ds.RasterCount
        # define inner function
        def _mask_band(tmp_ds, band_sn, dst_ds):
            tmp_band = tmp_ds.GetRasterBand(band_sn)
            tmp_data = tmp_band.ReadAsArray()
            tmp_mskband_pre = tmp_band.GetMaskBand()                
            tmp_msk_pre = tmp_mskband_pre.ReadAsArray()
            tmp_nodata_pre = tmp_band.GetNoDataValue()
            np_dt = gdal_array.GDALTypeCodeToNumericTypeCode(tmp_band.DataType)
            tmp_band.WriteArray(np.zeros(tmp_data.shape, np_dt))
            #this flushCache() changes the all values in the maskband data to 255.
            tmp_band.FlushCache()
            
            bands = [band_sn]
            burn_value = 1
            burns = [burn_value] 
            # following statement modify the maskband data to 255.
            # The original maskband data is stored in tmp_msk_pre.
            
            err = gdal.RasterizeLayer(
                tmp_ds,
                bands,
                ly,
                burn_values=burns,
                options=["ALL_TOUCHED=TRUE"]
            )
            tmp_ds.FlushCache()
            #combine original tmp mask band with tmp_data
            tmp_band = tmp_ds.GetRasterBand(band_sn)
            # tmp_data includes 0 and 1 values, where 0 indicates no valid pixels, and 1 indicates valid pixels. 
            tmp_data = tmp_band.ReadAsArray()
            out_band = dst_ds.GetRasterBand(band_sn)
            out_data = out_band.ReadAsArray()

            # set data band with nodata value if out_nodata exist
            #tmp_nodata = tmp_band.GetNoDataValue()
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

            return

        for band_sn in range(1,num+1):
            _mask_band(tmp_ds, band_sn, dst_ds)
        
        dst_ds.FlushCache()
        tmp_ds.FlushCache()
        dst_ds = None
        tmp_ds = None
        return outputfile
 
    def geotiff2netcdf_combined(self, infile, outfile, nodata_flag):
        '''
        convert geotiff file to netcdf file by copy everything to a new netcdf4 format file.
        input: infile - geotiff file name
            nodata_flag indicates you filling nodata value (True) or use maskband to represnet the subset (False)
        return: outfile - netcdf file name
        '''
        # open infile
        ds_in = gdal.Open(infile)
        gt =ds_in.GetGeoTransform()  
        tmpfile = os.path.splitext(infile)[0]+"_tmp.nc"
        command = ['gdal_translate']
        command.extend(['-of', 'NETCDF','-co', 'FORMAT=NC4'])
        #command.extend(['-ot', datatypename])
        command.extend([infile, tmpfile])
        self.cmd(*command)
        #cmd = " ".join(command)
        #os.system(cmd)

        src = Dataset(tmpfile)
        # create a output nc file
        dst = Dataset(outfile, "w", format="NETCDF4")
        dimkeys = src.dimensions.keys()
        dimnames = [name for name in dimkeys] 
        # copy attributes 
        for name in src.ncattrs():
            dst.setncattr(name, src.getncattr(name))
        # copy dimensions
        for name, dimension in src.dimensions.items():
            dst.createDimension(
            name, (len(dimension) if not dimension.isunlimited() else None))
        # copy all file data for variables    
        for name, variable in src.variables.items():
            if variable.shape:
                if variable.ndim == 1:
                    if name in dimnames:
                        if gt[2] == 0.0 and gt[4] == 0.0:
                            x = dst.createVariable(name, variable.datatype, variable.dimensions) 
                            for t in variable.ncattrs():
                                if t !='_FillValue': 
                                    dst.variables[name].setncattr(t, variable.getncattr(t))
                            dst.variables[name][:]=variable[:]
                    else:
                        x = dst.createVariable(name, variable.datatype, variable.dimensions) 
                        for t in variable.ncattrs():
                            if t !='_FillValue': 
                                dst.variables[name].setncattr(t, variable.getncattr(t))
                        dst.variables[name][:]=variable[:]    

                elif variable.ndim == 2:
                    lst = [ attr for attr in variable.ncattrs() if attr == '_FillValue']

                    def find_band(ds_in, variable):
                        #find out what band is associate to the variable    
                        #if "Band" in name:
                        #    r1 = re.compile("([a-zA-Z]+)([0-9]+)")
                        #    m = r1.match(name)
                        #    band_sn = int(m.group(m.lastindex))
                        #    return band_sn
                        #else:
                        band_sn = None
                        for i in range(1, ds_in.RasterCount+1):
                            if ds_in.GetRasterBand(i).GetMetadata()["bandname"] == variable.getncattr("bandname"):
                                band_sn = i
                                break
                        return band_sn

                    band_sn = find_band(ds_in, variable)
                    if band_sn:
                        band_in = ds_in.GetRasterBand(band_sn)
                        nodata_in = band_in.GetNoDataValue() 
                        mask_in_v =band_in.GetMaskBand().ReadAsArray()
                        mask_in_b = mask_in_v == 0
                        #datatype_nc = variable.datatype
                  
                        if nodata_in:
                            x = dst.createVariable(name, variable.datatype, variable.dimensions, fill_value = variable._FillValue)
                        else:
                            x = dst.createVariable(name, variable.datatype, variable.dimensions)

                        for t in variable.ncattrs():
                            if t !='_FillValue': 
                                dst.variables[name].setncattr(t, variable.getncattr(t))

                        try: 
                            tmp = variable[:,:]
                        except Exception:
                            pass
                        tmp.mask = np.flipud(mask_in_b)
                        dst.variables[name][:,:] = tmp
                    
                    else:
                        if lst:
                            x = dst.createVariable(name, variable.datatype, variable.dimensions, fill_value = variable._FillValue)
                        else:
                            x = dst.createVariable(name, variable.datatype, variable.dimensions)

                        for t in variable.ncattrs():
                            if t !='_FillValue': 
                                dst.variables[name].setncattr(t, variable.getncattr(t))

                        dst.variables[name][:,:] = variable[:,:]
                                                    
                else:
                    print("error")
            else:
                x = dst.createVariable(name, variable.datatype, variable.dimensions)
                for t in variable.ncattrs():
                    if t !='_FillValue': 
                        dst.variables[name].setncattr(t, variable.getncattr(t))
            
        ds_in = None
        src.close
        dst.close
        return outfile
