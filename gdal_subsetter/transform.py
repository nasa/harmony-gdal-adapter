# CLI for adapting a Harmony operation to GDAL
#
# If you have harmony in a peer folder with this repo, then you can run the following for an example
#    python3 -m harmony_gdal --harmony-action invoke --harmony-input "$(cat ../harmony/example/service-operation.json)"

import os
import subprocess
import re
import zipfile
import math
import shutil
from tempfile import mkdtemp
from affine import Affine
from osgeo import gdal, osr, ogr
from osgeo.gdalconst import *
from pystac import Asset
from harmony import BaseHarmonyAdapter
from harmony.util import stage, bbox_to_geometry, download, generate_output_filename
import pyproj
import numpy as np
import glob

mime_to_gdal = {
    "image/tiff": "GTiff",
    "image/png": "PNG",
    "image/gif": "GIF"
}

mime_to_extension = {
    "image/tiff": "tif",
    "image/png": "png",
    "image/gif": "gif"
}

mime_to_options = {
    "image/tiff": ["-co", "COMPRESS=LZW"]
}


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
            logger.warn('Ignoring subset request for user shapefile %s' % (message.subset.shape.href,))

        layernames = []

        operations = dict(
            variable_subset=source.variables,
            is_regridded=bool(message.format.crs),
            is_subsetted=bool(message.subset and message.subset.bbox)
        )

        result = item.clone()
        result.assets = {}

        # Create a temporary dir for processing we may do
        output_dir = mkdtemp()
        try:
            # Get the data file
            asset = next(v for k, v in item.assets.items() if 'data' in (v.roles or []))
            input_filename = download(
                asset.href,
                output_dir,
                logger=self.logger,
                access_token=self.message.accessToken,
                cfg=self.config)

            basename = os.path.basename(generate_output_filename(asset.href, **operations))

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
                self.completed_with_error('No recognized file foarmat, not process')

            self.update_layernames(filename, [v.name for v in layernames])
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

            # Update the STAC record
            result.assets['data'] = Asset(url, title=output_filename, media_type=mime, roles=['data'])

            # update metadata with bbox and extent in lon/lat coordinates
            result.bbox = self.get_bbox_lonlat(filename)
            result.geometry = bbox_to_geometry(result.bbox)

            # Return the STAC record
            return result
        finally:
            # Clean up any intermediate resources
            shutil.rmtree(output_dir)

    def process_geotiff(self, source, basename, input_filename, output_dir, layernames):

        if not source.variables:
            # process geotiff and all bands

            filename = input_filename
            # file_base= os.path.basename(filename)
            layer_id = basename + '__all'
            band = None
            layer_id, filename, output_dir = self.combin_transfer(
                layer_id, filename, output_dir, band)
            result = self.rename_to_result(
                layer_id, filename, output_dir)
            layernames.append(layer_id)
        else:

            variables = self.get_variables(input_filename)

            for variable in source.process('variables'):

                band = None

                index = next(i for i, v in enumerate(
                        variables) if v.name.lower() == variable.name.lower())
                if index is None:
                    return self.completed_with_error('band not found: ' + variable)
                band = index + 1

                filename = input_filename
                # file_base= os.path.basename(filename)
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

        variables = source.process('variables') or self.get_variables(input_filename)
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
            layernames, result = self.process_geotiff(source, basename, tiffile, output_dir, layernames)

        if ncfile:
            layernames, result = self.process_netcdf(source, basename, ncfile, output_dir, layernames)

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
        for i in range(len(layernames)):
            ds.GetRasterBand(i + 1).SetDescription(layernames[i])
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
        self.cmd('rm', '-rf', output_dir)
        self.cmd('mkdir', '-p', output_dir)

    def cmd(self, *args):
        self.logger.info(
            args[0] + " " + " ".join(["'{}'".format(arg) for arg in args[1:]]))
        result_str = subprocess.check_output(args).decode("utf-8")
        return result_str.split("\n")


    def is_rotated_geotransform(self, srcfile):
        #check if the srcfile includes a rotated geotransform
        dataset=gdal.Open(srcfile)
        gt=dataset.GetGeoTransform()
        check=False
        if gt[2] != 0.0 or gt[4] != 0:
            check=True
        return check

    def nc2tiff(self, layerid, filename, dstdir):
        def search(myDict, lookupkey):
            for key, value in myDict.items():
                if lookupkey in key:
                    return myDict[key]
            return None

        normalized_layerid = layerid.replace('/', '_')
        dstfile = "%s/%s" % (dstdir, normalized_layerid + '__nc2tiff.tif')
        ds=gdal.Open(filename)
        metadata=ds.GetMetadata()
        crs_wkt=search(metadata, "crs_wkt")
        if crs_wkt:
            command=['gdal_translate','-a_srs']
            command.extend([crs_wkt])
            command.extend([filename, dstfile])
            self.cmd(*command)
            return dstfile
        else:
            return filename

    def varsubset(self, srcfile, dstfile, band=None):
        if band:
            command = ['gdal_translate']
            command.extend(['-b', '%s' % (band) ])
            command.extend([srcfile, dstfile])
            self.cmd(*command)
            return dstfile
        else:
            return srcfile

    def subset(self, layerid, srcfile, dstdir, band=None):
        normalized_layerid = layerid.replace('/', '_')
        subset = self.message.subset

        if subset.bbox==None and subset.shape==None:
            dstfile = "%s/%s" % (dstdir, normalized_layerid + '__varsubsetted.tif')
            dstfile=self.varsubset(srcfile, dstfile, band)
            return dstfile

        if subset.bbox:
            [left, bottom, right, top]=self.get_bbox(srcfile)
            #subset.bbox is defined as [left/west,low/south,right/east,upper/north]
            subsetbbox=subset.process('bbox')
            #bbox = [str(c) for c in subset.bbox]
            [b0,b1], transform = self.lonlat2projcoord(srcfile,subsetbbox[0],subsetbbox[1])
            [b2,b3], transform = self.lonlat2projcoord(srcfile,subsetbbox[2],subsetbbox[3])

            if any( x == None for x in [b0,b1,b2,b3] ):
                dstfile = "%s/%s" % (dstdir, normalized_layerid + '__varsubsetted.tif')
                dstfile=self.varsubset(srcfile, dstfile, band)
            elif b0<left and b1<bottom and b2>right and b3>top:
                #user's input subset totally covers the image bbox, do not do subset
                dstfile = "%s/%s" % (dstdir, normalized_layerid + '__varsubsetted.tif')
                dstfile=self.varsubset(srcfile, dstfile, band)
            else:
                dstfile = "%s/%s" % (dstdir, normalized_layerid + '__subsetted.tif')
                dstfile=self.subset2(srcfile, dstfile, subsetbbox, band)

            return dstfile

        if subset.shape:
            #need know what the subset.shape is. Is it a file? do we need pre-process like we did for subset.bbox?
            dstfile = "%s/%s" % (dstdir, normalized_layerid + '__subsetted.tif')
            dstfile=self.subset2(srcfile, dstfile, subset.shape, band)

            return dstfile

    def reproject(self, layerid, srcfile, dstdir):
        crs = self.message.format.process('crs')
        if not crs:
            return srcfile
        normalized_layerid = layerid.replace('/', '_')
        dstfile = "%s/%s" % (dstdir, normalized_layerid + '__reprojected.tif')
        self.cmd('gdalwarp',
                 "-t_srs",
                 crs,
                 srcfile,
                 dstfile)
        return dstfile

    def resize(self, layerid, srcfile, dstdir):
        command = ['gdal_translate']
        fmt = self.message.format
        normalized_layerid = layerid.replace('/', '_')
        dstfile = "%s/%s__resized.tif" % (dstdir, normalized_layerid)

        if fmt.width or fmt.height:
            width = fmt.process('width') or 0
            height = fmt.process('height') or 0
            command.extend(["-outsize", str(width), str(height)])

        command.extend([srcfile, dstfile])
        self.cmd(*command)
        return dstfile

    def add_to_result(self, layerid, srcfile, dstdir):
        tmpfile = "%s/tmp-result.tif" % (dstdir)
        dstfile = "%s/result.tif" % (dstdir)
        if not os.path.exists(dstfile):
            self.cmd('cp', srcfile, dstfile)
            return dstfile

        tmpfile=self.stack_multi_file_with_metadata([dstfile,srcfile],tmpfile)
        self.cmd('mv', tmpfile, dstfile)

        return dstfile

    def stack_multi_file_with_metadata(self,infilelist,outfile):
        #infilelist include multiple files, each file does not have the same number of bands, but they have the same projection and geotransform. This function works for both rotated and non-rotated image files.

        collection=[]
        count=0
        ds_description=[]

        for id, layer in enumerate(infilelist, start=1):
            ds=gdal.Open(layer)
            filestr=os.path.splitext(os.path.basename(layer))[0]
            filename=ds.GetDescription()
            filestr=os.path.splitext(os.path.basename(filename))[0]
            if id==1:
                proj=ds.GetProjection()
                geot=ds.GetGeoTransform()
                cols=ds.RasterXSize
                rows=ds.RasterYSize
                band=ds.GetRasterBand(1)
                gtyp=band.DataType

            bandnum=ds.RasterCount
            md=ds.GetMetadata()
            for i in range(1,bandnum+1):
                band=ds.GetRasterBand(i)
                bmd=band.GetMetadata()
                bmd.update(md)
                data=band.ReadAsArray()
                count=count+1
                band_name="Band{count}:{filestr}-{i}".format(count=count, filestr=filestr,i=i)
                tmp_bmd={"bandname":band_name}
                bmd.update(tmp_bmd)
                ds_description.append(band_name)
                collection.append({"band_sn":count,"band_md":bmd, "band_array":data})

        dst_ds = gdal.GetDriverByName('GTiff').Create( outfile, cols, rows, count, gtyp )
        dst_ds.SetProjection(proj)
        dst_ds.SetGeoTransform(geot)
        dst_ds.SetMetadata({"Band Info": " ".join(ds_description)})
        for i, band in enumerate(collection):
            dst_ds.GetRasterBand(i+1).WriteArray(collection[i]["band_array"])
            dst_ds.GetRasterBand(i+1).SetMetadata(collection[i]["band_md"])

        dst_ds.FlushCache()                     # write to disk
        dst_ds = None

        return outfile

    def rename_to_result(self, layerid, srcfile, dstdir):
        dstfile = "%s/result.tif" % (dstdir)
        if not os.path.exists(dstfile):
            self.cmd('mv', srcfile, dstfile)
        return dstfile

    def reformat(self, srcfile, dstdir):
        #add gdal_subsettter version info to the file
        gdal_subsetter_version="gdal_subsetter_version={gdal_subsetter_ver}".format(gdal_subsetter_ver=self.get_version() )
        command = ['gdal_edit.py',  '-mo', gdal_subsetter_version, srcfile ]
        self.cmd(*command)

        output_mime = self.message.format.process('mime')
        if output_mime not in mime_to_gdal:
            raise Exception('Unrecognized output format: ' + output_mime)
        if output_mime == "image/tiff":
            return srcfile

        dstfile = "%s/translated.%s" % (dstdir, mime_to_extension[output_mime])
        command = ['gdal_translate',
                   '-of', mime_to_gdal[output_mime],
                   '-scale',
                   srcfile, dstfile]
        self.cmd(*command)

        return dstfile

    def read_layer_format(self, collection, filename, layer_id):
        gdalinfo_lines = self.cmd("gdalinfo", filename)

        layer_line = next(
            filter( (lambda line: re.search(":*" + layer_id+"$", line)), gdalinfo_lines), None)

        if layer_line == None:
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
                {"name": re.split(r" = ", subdataset)[-1]}))
        if result:
            return result

        # Some GeoTIFFdoes not have descriptions. directly use Band # as the variables
        for subdataset in filter((lambda line: re.match(r"^Band", line)), gdalinfo_lines):
            #result.append(ObjectView({"name": re.split(r"=", subdataset)[-1]}))
            tmpline=re.split(r" ", subdataset)
            result.append(ObjectView({"name": tmpline[0].strip()+tmpline[1].strip(), }))
        if result:
            return result

    def get_filetype(self, filename):
        if filename == None:
            return None
        else:
            if not os.path.exists(filename):
                return None

        file_basenamewithpath, file_extension = os.path.splitext(filename)
        if file_extension in ['.nc']:
            return 'nc'
        elif file_extension in ['.tif','.tiff']:
            return 'tif'
        elif file_extension in [ '.zip']:
            return 'zip'
        else:
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
        ds=gdal.Open(filename)
        gt=ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        ul_x, ul_y=self.calc_ij_coord(gt, 0, 0)
        ur_x, ur_y=self.calc_ij_coord(gt, cols, 0)
        lr_x, lr_y=self.calc_ij_coord(gt, cols, rows)
        ll_x, ll_y=self.calc_ij_coord(gt, 0, rows)
        return [min(ul_x,ll_x), min(ll_y,lr_y), max(lr_x,ur_x), max(ul_y,ur_y)]

    def get_bbox_lonlat(self, filename):
        """
        get the bbox in longitude and latitude of the raster file, and update the bbox and extent for the file, and return bbox.
        input:raster file name
        output:bbox of the raster file
        """
        ds=gdal.Open(filename, gdal.GA_Update)
        gt=ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        ul_x, ul_y=self.calc_ij_coord(gt, 0, 0)
        ur_x, ur_y=self.calc_ij_coord(gt, cols, 0)
        lr_x, lr_y=self.calc_ij_coord(gt, cols, rows)
        ll_x, ll_y=self.calc_ij_coord(gt, 0, rows)

        projection = ds.GetProjection()
        dst = osr.SpatialReference(projection)
        dstproj4=dst.ExportToProj4()
        ct2 = pyproj.Proj(dstproj4)

        ul_x2,ul_y2 = ct2(ul_x,ul_y,inverse=True)
        ur_x2,ur_y2 = ct2(ur_x,ur_y,inverse=True)
        lr_x2,lr_y2 = ct2(lr_x,lr_y,inverse=True)
        ll_x2,ll_y2 = ct2(ll_x,ll_y,inverse=True)

        ul_x2=float( "{:.7f}".format(ul_x2) )
        ul_y2=float( "{:.7f}".format(ul_y2) )

        ur_x2=float( "{:.7f}".format(ur_x2) )
        ur_y2=float( "{:.7f}".format(ur_y2) )

        lr_x2=float( "{:.7f}".format(lr_x2) )
        lr_y2=float( "{:.7f}".format(lr_y2) )

        ll_x2=float( "{:.7f}".format(ll_x2) )
        ll_y2=float( "{:.7f}".format(ll_y2) )

        lon_left=min(ul_x2,ll_x2)
        lat_low=min(ll_y2,lr_y2)
        lon_right=max(lr_x2,ur_x2)
        lat_high=max(ul_y2,ur_y2)

        #write bbox and extent in lon/lat unit to the metadata of the filename
        md=ds.GetMetadata()
        bbox=[ lon_left, lat_low, lon_right, lat_high]
        extent={'ul':[ul_x2,ul_y2],'ll':[ll_x2,ll_y2], 'ur':[ur_x2,ur_y2],'lr':[lr_x2,lr_y2]}
        md['bbox']=str(bbox)
        md['extent']=str(extent)
        ds.SetMetadata(md)
        ds=None
        return bbox

    def pack_zipfile(self, zipfilename, output_dir, variables=None):

        #unzip the file

        with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
            zip_ref.extractall(output_dir+'/unzip')

        tmptif=None

        filelist_tif=self.get_file_from_unzipfiles(output_dir+'/unzip', 'tif',variables)

        if filelist_tif:
            tmpfile=output_dir+'/tmpfile'
            #stack the single-band files into a multiple-band file
            tmptif=self.stack_multi_file_with_metadata(filelist_tif,tmpfile)

        tmpnc=None

        filelist_nc=self.get_file_from_unzipfiles(output_dir+'/unzip', 'nc')

        if filelist_nc:

            tmpnc=filelist_nc

        return tmptif, tmpnc

    def get_file_from_unzipfiles(self, extract_dir, filetype,variables=None):

        #check if there are geotiff files

        tmpexp = extract_dir+'/*.'+filetype

        filelist=sorted(glob.glob(tmpexp))

        if filelist:

            ch_filelist=[]

            if variables:

                for variable in variables:

                    variable_raw=fr"{variable}"

                    m=re.search(variable,filelist)

                    if m:
                        ch_filelist.append(m.string)

                filelist=ch_filelist

        return filelist

    def lonlat2projcoord(self,srcfile,lon,lat):
        #covert lon and lat to dataset's coord
        dataset=gdal.Open(srcfile)
        transform=dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        dst = osr.SpatialReference(projection)
        dstproj4=dst.ExportToProj4()
        ct2 = pyproj.Proj(dstproj4)
        xy=ct2(lon, lat)
        if math.isinf(xy[0]) or math.isinf(xy[1]):
            xy=[None,None]

        return [ xy[0], xy[1] ], transform

    def projcoord2lonlat(self, srcfile,x,y):
        #covert x,y of dataset's projected coord to longitude and latitude
        dataset=gdal.Open(srcfile)
        transform=dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        dst = osr.SpatialReference(projection)
        dstproj4=dst.ExportToProj4()
        ct2 = pyproj.Proj(dstproj4)
        lon,lat=ct2(x,y, inverse=True)
        #if math.isinf(xy[0]) or math.isinf(xy[1]):
        #    xy=[None,None]
        return [ lon, lat ], transform

    def subset2(self, tiffile, outputfile, bbox=None, band=None, shapefile=None):
        """
        subset tiffile with bbox or shapefile. bbox ans shapefile are exclusive.
        inputs:tiffile-geotif file
               outputfile-subsetted file name
               bbox is defined [left,low,right,upper] in lon/lat coordinates
               shapefile is a shapefile directory in which multiple files exist
        """
        #covert to absolute path
        tiffile=os.path.abspath(tiffile)
        outputfile=os.path.abspath(outputfile)
        RasterFormat = 'GTiff'
        ref_ds=gdal.Open(tiffile)
        gt=ref_ds.GetGeoTransform()
        tmpfile=os.path.splitext(outputfile)[0]+"-tmp.tif"
        if bbox:
            boxproj, proj = self.boxwrs84_boxproj(bbox, ref_ds)
            ul_x, ul_y, ul_i, ul_j, cols, rows=self.calc_subset_envelopwindow(ref_ds, boxproj)
            command=['gdal_translate']
            if band:
                command.extend(['-b', '%s' % (band) ])
            command.extend( [ '-srcwin', str(ul_i), str(ul_j), str(cols), str(rows) ] )
            command.extend([tiffile, tmpfile])
            self.cmd(*command)
            if gt[2] !=0.0 or gt[4] != 0.0:
                #create shapefile with box
                shapefile=self.box2shapefile(tiffile, bbox)
                self.rasterize(tmpfile, shapefile, outputfile)
            else:
                self.cmd(*['cp', tmpfile, outputfile])
            return outputfile

        elif shapefile:
            shapefile_out=os.path.dirname(outputfile)+"/tmpshapefile"
            boxproj, proj, shapefile_out = self.shapefile_boxproj(shapefile, ref_ds, shapefile_out)
            ul_x, ul_y, ul_i, ul_j, cols, rows=self.calc_subset_envelopwindow(ref_ds, boxproj)
            command=['gdal_translate']
            if band:
                command.extend(['-b', '%s' % (band) ])
            command.extend( [ '-srcwin', str(ul_i), str(ul_j), str(cols), str(rows) ] )
            command.extend([tiffile, tmpfile])
            self.cmd(*command)
            self.rasterize(tmpfile, shapefile_out, outputfile)
            return outputfile
        else:
            self.cmd(*['cp', tiffile, outputfile])
            return outputfile


    def boxwrs84_boxproj(self, boxwrs84, ref_ds):
        """
        convert the box define in lon/lat to box in projection coordinates defined in red_ds
        inputs: boxwrs84, which is defined as [left,low,right,upper], ref_ds is reference dataset
        returns: boxprj, which is also defined as [left,low,right,upper] in reference projection
                 projection, which is the projection defined in ref_ds dataset
        """
        projection = ref_ds.GetProjection()
        dst = osr.SpatialReference(projection)
        ll_lon,ll_lat = boxwrs84[0],boxwrs84[1]
        ur_lon,ur_lat = boxwrs84[2],boxwrs84[3]
        dstproj4=dst.ExportToProj4()
        ct = pyproj.Proj(dstproj4)
        llxy=ct(ll_lon, ll_lat)
        urxy=ct(ur_lon, ur_lat)
        boxproj=[ llxy[0], llxy[1], urxy[0],urxy[1] ]
        return boxproj, projection

    def calc_coord_ij(self, gt, x,y ):
        transform = Affine.from_gdal(*gt)
        rev_transform=~transform
        cols, rows =rev_transform*(x,y)
        return int(cols), int(rows)

    def calc_subset_window(self,ds,box):
        #box is defined as [left,low,right,upper] in projection ccordinates
        gt=ds.GetGeoTransform()
        ul_x=box[0]
        ul_y=box[3]
        rl_x=box[2]
        rl_y=box[1]
        ul_i, ul_j=self.calc_coord_ij(gt, ul_x,ul_y)
        rl_i, rl_j=self.calc_coord_ij(gt, rl_x,rl_y)
        #get the intersection between box and image in row, col coordinator
        cols_img = ds.RasterXSize
        rows_img = ds.RasterYSize
        ul_i=max(0, ul_i)
        ul_j=max(0, ul_j)
        rl_i=min(cols_img, rl_i)
        rl_j=min(rows_img, rl_j)
        cols=rl_i-ul_i
        rows=rl_j-ul_j
        ul_x, ul_y = self.calc_ij_coord(gt, ul_i, ul_j)
        return ul_x,ul_y,ul_i,ul_j,cols,rows

    def calc_subset_envelopwindow(self,ds,box):
        #box is defined as [left,low,right,upper] in projection coordinates
        #get 4 conners coordinate valuesin projection coorndinates
        gt=ds.GetGeoTransform()
        ul=(box[0],box[3])
        ur=(box[2],box[3])
        ll=(box[0],box[1])
        lr=(box[2],box[1])
        #get i,j coordinates in the array of 4 conners of the box
        ul_i, ul_j=self.calc_coord_ij(gt, ul[0],ul[1])
        ur_i, ur_j=self.calc_coord_ij(gt, ur[0],ur[1])
        ll_i, ll_j=self.calc_coord_ij(gt, ll[0],ll[1])
        lr_i, lr_j=self.calc_coord_ij(gt, lr[0],lr[1])
        #get the envelop of the box in array coordinates
        ul_i = min(ul_i, ur_i, ll_i, lr_i)
        ul_j = min(ul_j, ur_j, ll_j, lr_j)
        lr_i = max(ul_i, ur_i, ll_i, lr_i)
        lr_j = max(ul_j, ur_j, ll_j, lr_j)
        #get the intersection between box and image in row, col coordinator
        cols_img = ds.RasterXSize
        rows_img = ds.RasterYSize
        ul_i=max(0, ul_i)
        ul_j=max(0, ul_j)
        lr_i=min(cols_img, lr_i)
        lr_j=min(rows_img, lr_j)
        cols=lr_i-ul_i
        rows=lr_j-ul_j
        ul_x, ul_y = self.calc_ij_coord(gt, ul_i, ul_j)
        return ul_x,ul_y,ul_i,ul_j,cols,rows

    def calc_ij_coord(self, gt, col, row):
        transform = Affine.from_gdal(*gt)
        x,y = transform * (col, row)
        return x,y

    def create_shapefile_with_box(self,box,projection,shapefile):
        """
        input: box=[min_lon,min_lat,max_lon,max_lat]
        projection: box values are based on the projection, lon and lat is EPSG4326
        output: polygon geometry
        """
        minX=box[0]
        minY=box[1]
        maxX=box[2]
        maxY=box[3]
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(minX, minY)
        ring.AddPoint(maxX, minY)
        ring.AddPoint(maxX, maxY)
        ring.AddPoint(minX, maxY)
        ring.AddPoint(minX, minY)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        # create output file
        outDriver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(shapefile):
            os.remove(shapefile)
        outDataSource = outDriver.CreateDataSource(shapefile)
        outSpatialRef = osr.SpatialReference(projection)
        outLayer = outDataSource.CreateLayer('box',outSpatialRef, geom_type=ogr.wkbPolygon )
        featureDefn = outLayer.GetLayerDefn()
        # add new geom to layer
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(polygon)
        outLayer.CreateFeature(outFeature)
        outFeature.Destroy
        outDataSource.Destroy()

    def box2shapefile(self, inputfile, box):
        """
        inputs:inputfile is the geotiff file, box[minlon,minlat,maxlon,maxlat] is in lon/lat.
        return: shapefile.
        """
        in_ds=gdal.Open(inputfile)
        in_gt=in_ds.GetGeoTransform()
        inv_gt=gdal.InvGeoTransform(in_gt)
        boxproj,proj = self.boxwrs84_boxproj(box, in_ds)
        if inv_gt is None:
            raise RuntimeError('Inverse geotransform failed')
        inputdir=os.path.dirname(inputfile)
        basename=os.path.basename(inputfile).split(".")[0]
        shapefile=inputdir+'/'+basename+"-shapefile"
        if os.path.isfile(shapefile) or os.path.isdir(shapefile):
            command=['rm']
            command.extend(['-rf',shapefile])
            self.cmd(*command)
        self.create_shapefile_with_box(boxproj, proj, shapefile)
        return shapefile

    def rasterize(self, infile, shapefile, outfile):
        """
        inputs: infile- geotiff file, shaepfile-mask the rasterfile with this shapefile         outfile - output file name
        return: outfile
        """
        infile=os.path.abspath(infile)
        #copy infile to outfile
        self.cmd(*['cp', '-f', infile, outfile])
        #get the infomation of outputfile
        ds=gdal.Open(outfile, GA_Update)
        num=ds.RasterCount
        bands = tuple(range(1,num+1))
        burns = tuple(np.full(num,0))
        #get the information of shapefile
        shp=ogr.Open(shapefile)
        ly=shp.GetLayerByIndex(0)
        lyname=ly.GetName()
        #rasterize the shapefile according to infile
        err = gdal.RasterizeLayer(
            ds,
            bands,
            ly,
            burn_values=burns
            )
        #tested options initValues and Inverse, both are not work
        #options=["ATTRIBUTE={lyname}".format(lyname=lyname)]
        #need use following way to process the pixels outside the box
        #update the bands of ds
        in_ds=gdal.Open(infile, GA_ReadOnly)
        in_data=in_ds.ReadAsArray()
        out_data=ds.ReadAsArray()
        rst_data=in_data-out_data
        if num == 1:
            out_band=ds.GetRasterBand(1)
            out_band.WriteArray(rst_data)
        else:
            for i in range(1,num+1):
                out_band=ds.GetRasterBand(i)
                out_band.WriteArray(rst_data[i-1,:,:])

        ds.FlushCache()
        ds, in_ds=None, None

    def shapefile_boxproj(self, shapefile, ref_ds, outputfile):
        """
        calculate the envelop box and projection in the coordinates defined in reffile/
        inputs: shapefile - used to define the AOI.
                reffile - the reference geotiff file.
        returns: boxproj, proj, shapefile
        """
        ref_proj=ref_ds.GetProjection()
        tmp = gpd.GeoDataFrame.from_file(shapefile)
        tmpproj = tmp.to_crs(ref_proj)
        tmpproj.to_file(outputfile)
        shp = ogr.Open(outputfile)
        lyr=shp.GetLayer()
        lyrextent=lyr.GetExtent()
        #Extent[lon_min,lon_max, lat_min,lat_max], boxproj is defined as [lon_min,lat_min, lon_max, lat_max]
        boxproj=[lyrextent[0], lyrextent[2], lyrextent[1], lyrextent[3]]
        return boxproj, ref_proj, outputfile


