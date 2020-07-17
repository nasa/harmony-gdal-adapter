# CLI for adapting a Harmony operation to GDAL
#
# If you have harmony in a peer folder with this repo, then you can run the following for an example
#    python3 -m harmony_gdal --harmony-action invoke --harmony-input "$(cat ../harmony/example/service-operation.json)"

import sys
import os
import subprocess
import os
import urllib.request
import urllib.parse
import re
import boto3
import rasterio
import zipfile
import math
from math import atan, tan, sqrt
from affine import Affine
from osgeo import gdal, osr, ogr
from harmony import BaseHarmonyAdapter
import pyproj
import numpy as np

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

    def invoke(self):
        """
        Run the service on the message contained in `self.message`.  Fetches data, runs the service,
        puts the result in a file, calls back to Harmony, and cleans up after itself.

        Note: When a synchronous request is made, this only operates on a single granule.  If multiple
            granules are requested, it subsets the first.  The subsetter is capable of combining
            multiple granules but will not do so until adequate performance of data access has
            been established
            For async requests, it subsets the granules individually and returns them as partial results
        """
        logger = self.logger
        message = self.message

        if message.subset and message.subset.shape:
            logger.warn('Ignoring subset request for user shapefile %s' %
                        (message.subset.shape.href,))

        try:
            # Limit to the first granule.  See note in method documentation
            granules = message.granules
            if message.isSynchronous:
                granules = granules[:1]

            output_dir = "tmp/data"
            self.prepare_output_dir(output_dir)

            layernames = []

            operations = dict(
                is_variable_subset=True,
                is_regridded=bool(message.format.crs),
                is_subsetted=bool(message.subset and message.subset.bbox)
            )

            result = None
            for i, granule in enumerate(granules):
                self.download_granules([granule])

                file_type = self.get_filetype(granule.local_filename)
                if file_type == 'tif':
                    layernames, result = self.process_geotiff(
                            granule,output_dir,layernames,operations,message.isSynchronous
                            )
                elif file_type == 'nc':
                    layernames, result = self.process_netcdf(
                            granule,output_dir,layernames,operations,message.isSynchronous
                            )
                elif file_type == 'zip':
                    layernames, result = self.process_zip(
                            granule,output_dir,layernames,operations,message.isSynchronous
                             )
                else:
                    logger.exception(e)
                    self.completed_with_error('No reconized file foarmat, not process')

                if not message.isSynchronous:
                    # Send a single file and reset
                    self.update_layernames(result, [v.name for v in granule.variables])
                    result = self.reformat(result, output_dir)
                    progress = int(100 * (i + 1) / len(granules))
                    self.async_add_local_file_partial_result(result, source_granule=granule, title=granule.id, progress=progress, **operations)
                    self.cleanup()
                    self.prepare_output_dir(output_dir)
                    layernames = []
                    result = None

######################################


            if message.isSynchronous:
                self.update_layernames(result, layernames)
                result = self.reformat(result, output_dir)
                self.completed_with_local_file(
                    result, source_granule=granules[-1], **operations)
            else:
                self.async_completed_successfully()

        except Exception as e:
            logger.exception(e)
            self.completed_with_error('An unexpected error occurred')

        finally:
            self.cleanup()


###########################################################
    def process_geotiff(self, granule, output_dir, layernames, operations,isSynchronous):

        if not granule.variables:
            # process geotiff and all bands
            
            filename = granule.local_filename
            #file_base= os.path.basename(filename)
            layer_id = granule.id + '__all'
            band = None
            layer_id, filename, output_dir = self.combin_transfer(
                layer_id, filename, output_dir, band)
            result = self.rename_to_result(
                layer_id, filename, output_dir)
            layernames.append(layer_id)
        else:

            variables = self.get_variables(granule.local_filename)

            for variable in granule.variables:

                band = None

                index = next( i for i, v in enumerate(
                        variables) if v.name.lower() == variable.name.lower() )
                if index is None:
                    return self.completed_with_error('band not found: ' + variable)
                band = index + 1

                filename = granule.local_filename
                #file_base= os.path.basename(filename)
                layer_id = granule.id + '__'  + variable.name

                layer_id, filename, output_dir = self.combin_transfer(
                    layer_id, filename, output_dir, band)

                result = self.add_to_result(
                                    layer_id,
                                    filename,
                                    output_dir
                                )
                layernames.append(layer_id)

        return layernames, result

#########################################################
    def process_netcdf(self, granule, output_dir, layernames, operations, isSynchronous):

        if not granule.variables:
            variables = self.get_variables(granule.local_filename)
            granule.variables = variables

        for variable in granule.variables:

            band = None
                                    # For non-geotiffs, we reference variables by appending a file path
            layer_format = self.read_layer_format(
                        granule.collection,
                        granule.local_filename,
                        variable.name
                        )
            filename = layer_format.format(
                        granule.local_filename)

            layer_id = granule.id + '__' + variable.name

            layer_id, filename, output_dir = self.combin_transfer(
                                    layer_id, filename, output_dir, band)

            result = self.add_to_result(
                                    layer_id,
                                    filename,
                                    output_dir
                                )
            layernames.append(layer_id)

        return layernames, result

####################################################

    def process_zip(self, granule, output_dir, layernames, operations, isSynchronous):
         
        [tiffile, ncfile]=self.pack_zipfile(granule.local_filename, output_dir)
        
        if tiffile:

            granule.local_filename=tiffile

            layernames, result = self.process_geotiff(granule, output_dir, layernames, operations,isSynchronous)


        if ncfile:

            granule.local_filename=ncfile

            layernames, result = self.process_netcdf(self, granule, output_dir, layernames, operations, isSynchronous)

        return layernames, result


##########################################################

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

    def subset(self, layerid, srcfile, dstdir, band=None):
        normalized_layerid = layerid.replace('/', '_')
        subset = self.message.subset

        if subset.bbox==None and subset.shape==None:
            return srcfile

        if subset.bbox:
            [left, bottom, right, top]=self.get_bbox(srcfile)
            #subset.bbox in srcfile is defined from ll to ur
            subsetbbox=subset.bbox
            bbox = [str(c) for c in subset.bbox]
            [b0,b1], transform = self.lonlat2projcoord(srcfile,subsetbbox[0],subsetbbox[1])
            [b2,b3], transform = self.lonlat2projcoord(srcfile,subsetbbox[2],subsetbbox[3])
            if any( x == None for x in [b0,b1,b2,b3] ):
                return srcfile
            #subset2 can do both rotated and non-rotated images
            dstfile = "%s/%s" % (dstdir, normalized_layerid + '__subsetted.tif')
            dstfile=self.subset2(srcfile, dstfile, subsetbbox, band)
            return dstfile
        if subset.shape:
            #to be done
            dstfile = "%s/%s" % (dstdir, normalized_layerid + '__subsetted.tif')            
            self.cmd('cp ', srcfile, dstfile) 
            return 
            
    def reproject(self, layerid, srcfile, dstdir):
        crs = self.message.format.crs
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
            width = fmt.width or 0
            height = fmt.height or 0
            command.extend(["-outsize", str(width), str(height)])

        command.extend([srcfile, dstfile])

        self.cmd(*command)
        return dstfile

    def add_to_result(self, layerid, srcfile, dstdir):
        tmpfile = "%s/tmp-result.tif" % (dstdir)
        dstfile = "%s/result.tif" % (dstdir)

        command = ['gdal_merge.py',
                   '-o', tmpfile,
                   '-of', "GTiff",
                   '-separate']
        command.extend(mime_to_options["image/tiff"])

        if not os.path.exists(dstfile):
            self.cmd('cp', srcfile, dstfile)
            return dstfile

        command.extend([dstfile, srcfile])

        self.cmd(*command)
        self.cmd('mv', tmpfile, dstfile)

        return dstfile

    def rename_to_result(self, layerid, srcfile, dstdir):
        dstfile = "%s/result.tif" % (dstdir)
        if not os.path.exists(dstfile):
            self.cmd('mv', srcfile, dstfile)
        return dstfile

    def reformat(self, srcfile, dstdir):
        output_mime = self.message.format.mime
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
            filter((lambda line: line.endswith(":" + layer_id)), gdalinfo_lines), None)
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
    
    def combin_transfer_rotated(self, layer_id, filenamelist, output_dir, band):
        #gdal_warp can only process single-band rotated image

        outfilelist=[]
        for filename in filenamelist:
            layer_id, filename, output_dir=self.combin_transfer(self, layer_id, filename, output_dir, band)
            outfilelist.append(filename)

        #comabine them into a tiff file
        outputfile=output_dir+'/rotated.tif'
        outputfile=self.stacking(outfilelist, outputfile)
    
        return layer_id, outputfile, output_dir

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
        ds_raster = rasterio.open(filename)
        bounds = ds_raster.bounds
        left= bounds.left
        bottom = bounds.bottom
        right = bounds.right
        top = bounds.top
        return [left, bottom, right, top]

    def pack_zipfile(self, zipfilename, output_dir, variables=None):

        #unzip the file
        
        with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
            zip_ref.extractall(output_dir+'/unzip')

        tmptif=None
           
        filelist_tif=self.get_file_from_unzipfiles(output_dir+'/unzip', 'tif',variables)
        
        if filelist_tif:
            tmpfile=output_dir+'/tmpfile'       
            #stack the single-band files into a multiple-band file
            tmptif=self.stacking(filelist_tif, tmpfile)

        tmpnc=None  

        filelist_nc=self.get_file_from_unzipfiles(output_dir+'/unzip', 'nc')

        if filelist_nc:

            tmpnc=filelist_nc

        return tmptif, tmpnc
    

    def get_file_from_unzipfiles(self, extract_dir, filetype,variables=None):
        
        #check if there are geotiff files
    
        lstfile=extract_dir +'/list.txt'

        command=['ls',
                extract_dir+'/*.'+filetype,
                '>',
                lstfile
                ] 

        cmdline=' '.join(map(str, command))

        os.system(cmdline)

        filelist=None

        if os.path.isfile(lstfile) and os.path.getsize(lstfile) > 0:

            with open(lstfile) as f:
                filelist = f.read().splitlines()
            
            ch_filelist=[]

            if variables:

                for variable in variables:

                    variable_raw=fr"{variable}"

                    m=re.search(variable,filelist)

                    if m:
                        ch_filelist.append(m.string)

                filelist=ch_filelist 

        return filelist
    
    def stacking(self, infilelist, outputfile):

        # Read metadata of first file
        with rasterio.open(infilelist[0]) as src0:
            meta = src0.meta

        # Update meta to reflect the number of layers
        meta.update(count = len(infilelist))

        # Read each layer and write it to stack
        with rasterio.open(outputfile, 'w', **meta) as dst:
            for id, layer in enumerate(infilelist, start=1):
                with rasterio.open(layer) as src1:
                    dst.write_band(id, src1.read(1))

        return outputfile
       

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

    def subset_rotated(self, tiffile, outputfile, bbox, band=None):
        RasterFormat = 'GTiff'
        raw_file_name = os.path.splitext(os.path.basename(tiffile))[0]
        [ll_lon, ll_lat,ur_lon, ur_lat]=bbox
        if band:
            command = ['gdal_translate']
            command.extend(['-b', '%s' % (band)])
            command.extend([tiffile, 'tmp/data/tmpbandfile'])
            self.cmd(*command)
            tiffile='tmp/data/tmpbandfile'

        command = ['gdalwarp']
        command.extend(['-te',ll_lon,ll_lat,ur_lon,ur_lat,'-te_srs', 'EPSG:4326'])
        command.extend([tiffile, outputfile])
        self.cmd(*command)
        return outputfile



    def subset2(self, tiffile, outputfile,bbox, band=None, shapefile=None):
        #bbox is defined from ll to ur
        RasterFormat = 'GTiff'
        ref_ds=gdal.Open(tiffile)
        gt=ref_ds.GetGeoTransform()
        boxproj, proj = self.boxwrs84_boxproj(bbox, ref_ds)
        ul_x, ul_y, ul_i, ul_j, cols, rows=self.calc_subset_window(ref_ds, boxproj)
        command=['gdal_translate']
        if band:
            command.extend(['-b', '%s' % (band) ])

        command.extend( [ '-srcwin', str(ul_i), str(ul_j), str(cols), str(rows) ] )
        command.extend([tiffile, outputfile])
        self.cmd(*command)
        return outputfile


    def subset4(self, tiffile, outputfile, bbox, band=None, shapefile=None):

        #bbox=[min_lon,min_lat, max_lon,max_lat]

        def fillnodata(data, i1,j1, i2,j2, nodata):
            data1=np.ma.array(data, mask=True, fill_value=nodata)
            #x direction (xres) corresponds to rows in data array
            #y direction (yres) corresponds to rows in data array
            data1.mask[j1:j2,i1:i2]=False
            data2=data1.filled()
            return data2

        RasterFormat = 'GTiff'
        ref_ds=gdal.Open(tiffile)
        gt=ref_ds.GetGeoTransform()
        boxproj, proj = boxwrs84_boxproj(bbox, ref_ds)
        ul_x, ul_y, ul_i, ul_j, subcols, subrows=calc_subset_window(ref_ds,boxproj)
        lr_i=ul_i+subcols
        lr_j=ul_j+subrows

        #create a copy of the tiffile
        driver = gdal.GetDriverByName('GTiff')
        driver = ref_ds.GetDriver()
        out_ds = driver.CreateCopy(outputfile, ref_ds, strict=0)
        cols=out_ds.RasterXSize
        rows=out_ds.RasterYSize
        nb=out_ds.RasterCount
        nodata=out_ds.GetRasterBand(1).GetNoDataValue()
        datatype=gdal.GetDataTypeName(out_ds.GetRasterBand(1).DataType)

        if datatype =='Byte':
            nodata=255
        else:
            nodata=-1
        
        if band:
            i=band
            in_band = ref_ds.GetRasterBand(i)
            in_data = in_band.ReadAsArray( 0, 0, cols, rows)
            out_data=np.copy(in_data)
            out_data1=fillnodata(out_data, ul_i, ul_j, lr_i, lr_j, nodata)
            out_band = out_ds.GetRasterBand(i)
            out_band.WriteArray(out_data1)
            out_band.SetNoDataValue(nodata)
        else:
            for i in range(1,nb+1):
                in_band = ref_ds.GetRasterBand(i)
                in_data = in_band.ReadAsArray( 0, 0, cols, rows)
                out_data=np.copy(in_data)
                out_data1=fillnodata(out_data, ul_i, ul_j, lr_i, lr_j, nodata)
                out_band = out_ds.GetRasterBand(i)
                out_band.WriteArray(out_data1)
                out_band.SetNoDataValue(nodata)
        out_ds.FlushCache() #saves to disk!!
        del out_ds
        return outputfile


    def boxwrs84_boxproj(self, boxwrs84, ref_ds):
        #boxwrs84 is defined from ll to ur, ref_ds is reference dataset
        #return boxprj is also defined as from ll to ur in reference projection
        projection = ref_ds.GetProjection()
        dst = osr.SpatialReference(projection)
        ll_lon,ll_lat = boxwrs84[0],boxwrs84[1]
        ur_lon,ur_lat = boxwrs84[2],boxwrs84[3]
        dstproj4=dst.ExportToProj4()
        ct = pyproj.Proj(dstproj4)
        llxy=ct(ll_lon, ll_lat)
        urxy=ct(ur_lon, ur_lat)
        boxproj=[ llxy[0],llxy[1], urxy[0],urxy[1] ]
        return boxproj, projection


    def calc_ij_coord(self, gt, col, row):
        transform = Affine.from_gdal(*gt)
        x,y = transform * (col, row)
        return x,y


    def calc_coord_ij(self, gt, x,y ):
        transform = Affine.from_gdal(*gt)
        rev_transform=~transform
        cols, rows =rev_transform*(x,y)
        return int(cols), int(rows)


    def calc_subset_window(self,ds,box):
        #box is defined from ll to ur
        gt=ds.GetGeoTransform()
        ul_x=box[0]
        ul_y=box[3]
        rl_x=box[2]
        rl_y=box[1]
        ul_i, ul_j=self.calc_coord_ij(gt, ul_x,ul_y)
        rl_i, rl_j=self.calc_coord_ij(gt, rl_x,rl_y)
        cols=rl_i-ul_i
        rows=rl_j-ul_j
        ul_x, ul_y = self.calc_ij_coord(gt, ul_i, ul_j)
        return ul_x,ul_y,ul_i,ul_j,cols,rows

