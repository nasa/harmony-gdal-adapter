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
            
            #convert the subdataset in the nc file into the geotif file
            filename=self.nc2tiff(layer_id,filename,output_dir)

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

    def varsubset(self, layerid, srcfile, dstfile, band=None):
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
            dstfile=self.varsubset(layerid, srcfile, dstfile, band)
            return dstfile

        if subset.bbox:
            [left, bottom, right, top]=self.get_bbox(srcfile)
            #subset.bbox in srcfile is defined from ll to ur
            subsetbbox=subset.bbox
            bbox = [str(c) for c in subset.bbox]
            [b0,b1], transform = self.lonlat2projcoord(srcfile,subsetbbox[0],subsetbbox[1])
            [b2,b3], transform = self.lonlat2projcoord(srcfile,subsetbbox[2],subsetbbox[3])

            if any( x == None for x in [b0,b1,b2,b3] ):
                dstfile = "%s/%s" % (dstdir, normalized_layerid + '__varsubsetted.tif')
                dstfile=self.varsubset(layerid, srcfile, dstfile, band)
            elif b0<left and b1<bottom and b2>right and b3>top:
                dstfile = "%s/%s" % (dstdir, normalized_layerid + '__varsubsetted.tif')
                dstfile=self.varsubset(layerid, srcfile, dstfile, band)
            else:
                dstfile = "%s/%s" % (dstdir, normalized_layerid + '__subsetted.tif')
                dstfile=self.subset2(srcfile, dstfile, subsetbbox, band)

            return dstfile

        if subset.shape:
            #to be done
            dstfile = "%s/%s" % (dstdir, normalized_layerid + '__subsetted.tif')            
            self.cmd('cp ', srcfile, dstfile) 
            return dstfile 
            
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
        if not os.path.exists(dstfile):
            self.cmd('cp', srcfile, dstfile)
            return dstfile

        #tmpfile=self.stacktwofileswithmetadata(dstfile,srcfile,tmpfile)
        tmpfile=self.stackwithmetadata2(dstfile,srcfile,tmpfile)

        self.cmd('mv', tmpfile, dstfile)
        return dstfile
    
    def stacktwofileswithmetadata(self,file1,file2,outfile):
        #file1 and file2 are geotiff files, there maybe multi-band files,
        #and two files ma hav differnt number of bands.

        def migrate_raster_metadata2band_metadata(file):
            ds=gdal.Open(file, gdal.GA_Update)
            md=ds.GetMetadata()
            bandnum=ds.RasterCount
            bmds=[]
            for i in range(bandnum):
                b=ds.GetRasterBand(i+1)
                bmd=b.GetMetadata()
                bmd.update(md)
                b.SetMetadata(bmd)
                bmds.append(bmd)

            ds=None
            return file, bmds

        file1,bmds1=migrate_raster_metadata2band_metadata(file1)
        file2,bmds2=migrate_raster_metadata2band_metadata(file2)
        flist=[file1,file2]
        mdlist=[*bmds1,*bmds2]
        command = ['gdal_merge.py',
                   '-o', outfile,
                   '-of', "GTiff",
                   '-separate']
        command.extend(mime_to_options["image/tiff"])
        command.extend([file1, file2])
        self.cmd(*command)

        #gdal_merge.py does not keep band metadata, update the band metadata explicity.

        outds=gdal.Open(outfile, gdal.GA_Update)

        for i, md in enumerate(mdlist):
            outds.GetRasterBand(i+1).SetMetadata(md)

        outds.FlushCache()
        outds=None

        return outfile

    def stackwithmetadata2(self,file1,file2,outfile):
        #file1 and file2 are geotiff files
        def migrate_raster_metadata2band_metadata(file):
            src_ds = gdal.Open(file)
            tmpfile=os.path.splitext(file)[0]+"_tmp.tif"

            driver = gdal.GetDriverByName('GTiff')
            ds = driver.CreateCopy(tmpfile, src_ds)
            md=ds.GetMetadata()
            bandnum=ds.RasterCount
            bmds=[]

            for i in range(bandnum):
                b=ds.GetRasterBand(i+1)
                bmd=b.GetMetadata()
                bmd.update(md)
                b.SetMetadata(bmd)
                bmds.append(bmd)

            ds.FlushCache()
            ds=None
            return tmpfile,bandnum,bmds

        file1,bn1,bmds1=migrate_raster_metadata2band_metadata(file1)
        file2,bn2,bmds2=migrate_raster_metadata2band_metadata(file2)
        flist=[file1,file2]
        mdlist=[bmds1,*bmds2]
        src_ds1 = gdal.Open(file1)
        src_ds2 = gdal.Open(file2)
        src_dss=[src_ds1,src_ds2]

        geotransform=src_ds1.GetGeoTransform()
        projection=src_ds1.GetProjection()
        cols=src_ds1.RasterXSize
        rows=src_ds1.RasterYSize
        bn=bn1+bn2
        dst_ds = gdal.GetDriverByName('GTiff').Create(outfile, cols, rows, bn, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)

        
        for i in rang(bn1):
            dst_ds.GetRasterBand(i+1).WriteArray(ds.ReadAsArray()[i]   # write file1 bands to the raster
            dst_ds.GetRasterBand(i+1).SetMetadata(mdlist[i])
            
        for i in rang(bn2):
            dst_ds.GetRasterBand(bn1+i+1).WriteArray(ds.ReadAsArray()[i]   # write file1 bands to the raster
            dst_ds.GetRasterBand(bn1+i+1).SetMetadata(mdlist[bn1+i])


        dst_ds.FlushCache()                     # write to disk
        dst_ds = None

        return outfile


    def stackmultiplefileswithmetadata(self,filelist,outfile):
        #filelist is a list of geotiff filenames, there maybe multi-band files, each file may includes differnt number of band.

        def migrate_raster_metadata2band_metadata(file):
            ds=gdal.Open(file, gdal.GA_Update)
            md=ds.GetMetadata()
            bandnum=ds.RasterCount
            bmds=[]
            for i in range(bandnum):
                b=ds.GetRasterBand(i+1)
                bmd=b.GetMetadata()
                bmd.update(md)
                b.SetMetadata(bmd)
                bmds.append(bmd)
            ds=None
            return file, bmds

        flist=[]
        mdlist=[]
        for filename in filelist:
            filename,bmds=migrate_raster_metadata2band_metadata(filename)
            flist.append(filename)
            mdlist.append(*bmds)

        command = ['gdal_merge.py',
                   '-o', outfile,
                   '-of', "GTiff",
                   '-separate']
        command.extend(mime_to_options["image/tiff"])
        command.extend(flist)
        self.cmd(*command)

        #gdal_merge.py does not keep band metadata, update the band metadata explicity.

        outds=gdal.Open(outfile, gdal.GA_Update)

        for i, md in enumerate(mdlist):
            outds.GetRasterBand(i+1).SetMetadata(md)

        outds.FlushCache()
        outds=None
        return outfile


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
        ds=gdal.Open(filename)
        gt=ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        ul_x, ul_y=self.calc_ij_coord(gt, 0, 0)
        ur_x, ur_y=self.calc_ij_coord(gt, cols, 0)
        lr_x, lr_y=self.calc_ij_coord(gt, cols, rows)
        ll_x, ll_y=self.calc_ij_coord(gt, 0, rows)
        return [min(ul_x,ll_x), min(ll_y,lr_y), max(lr_x,ur_x), max(ul_y,ur_y)] 

    def pack_zipfile(self, zipfilename, output_dir, variables=None):

        #unzip the file
        
        with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
            zip_ref.extractall(output_dir+'/unzip')

        tmptif=None
           
        filelist_tif=self.get_file_from_unzipfiles(output_dir+'/unzip', 'tif',variables)
        
        if filelist_tif:
            tmpfile=output_dir+'/tmpfile'       
            #stack the single-band files into a multiple-band file
            #tmptif=self.stackwithmetadata(filelist_tif,tmpfile)
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

    def calc_ij_coord(self, gt, col, row):
        transform = Affine.from_gdal(*gt)
        x,y = transform * (col, row)
        return x,y
