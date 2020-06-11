# CLI for adapting a Harmony operation to GDAL
#
# If you have harmony in a peer folder with this repo, then you can run the following for an example
#    python3 -m harmony_gdal --harmony-action invoke --harmony-input "$(cat ../harmony/example/service-operation.json)"

import sys
import subprocess
import os
import urllib.request
import urllib.parse
import re
import boto3

from harmony import BaseHarmonyAdapter

from osgeo import gdal

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
            logger.warn('Ignoring subset request for user shapefile %s' % (message.subset.shape.href,))

        try:
            # Limit to the first granule.  See note in method documentation
            granules = message.granules
            if message.isSynchronous:
                granules = granules[:1]

            output_dir = "tmp/data"
            self.prepare_output_dir(output_dir)

            layernames = []

            operations = dict(
                is_variable_subset = True,
                is_regridded = bool(message.format.crs),
                is_subsetted = bool(message.subset and message.subset.bbox)
            )

            result = None
            for i, granule in enumerate(granules):
                self.download_granules([granule])

                variables = self.get_variables(granule.local_filename)
                is_geotiff = self.is_geotiff(granule.local_filename)
                if not granule.variables:
                    granule.variables = variables

                for variable in granule.variables:
                    band = None
                    if is_geotiff:
                        # For geotiffs, we can't reference variables by name but need to reference
                        # by raster band instead
                        index = next(i for i, v in enumerate(variables) if v.name == variable.name)
                        if index is None:
                            return self.completed_with_error('band not found: ' + variable)
                        band = index + 1
                        filename = granule.local_filename
                    else:
                        # For non-geotiffs, we reference variables by appending a file path
                        layer_format = self.read_layer_format(
                            granule.collection,
                            granule.local_filename,
                            variable.name
                        )
                        filename = layer_format.format(granule.local_filename)

                    layer_id = granule.id + '__' + variable.name
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
                    result = self.add_to_result(
                        layer_id,
                        filename,
                        output_dir
                    )
                    layernames.append(layer_id)


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

            if message.isSynchronous:
                self.update_layernames(result, layernames)
                result = self.reformat(result, output_dir)
                self.completed_with_local_file(result, source_granule=granules[-1], **operations)
            else:
                self.async_completed_successfully()

        except Exception as e:
            logger.exception(e)
            self.completed_with_error('An unexpected error occurred')

        finally:
            self.cleanup()

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
        self.logger.info(args[0] + " " + " ".join(["'{}'".format(arg) for arg in args[1:]]))
        result_str = subprocess.check_output(args).decode("utf-8")
        return result_str.split("\n")

    def subset(self, layerid, srcfile, dstdir, band=None):
        normalized_layerid = layerid.replace('/', '_')
        subset = self.message.subset
        if not subset:
            return srcfile

        command = ['gdal_translate', '-of', 'GTiff']
        if band is not None:
            command.extend(['-b', '%s' % (band)])
        if subset.bbox:
            bbox = [str(c) for c in subset.bbox]
            if float(bbox[2]) < float(bbox[0]):
                # If the bounding box crosses the antimeridian, subset into the east half and west half and merge
                # the result
                west_dstfile = "%s/%s" % (dstdir, normalized_layerid + '__west_subsetted.tif')
                east_dstfile = "%s/%s" % (dstdir, normalized_layerid + '__east_subsetted.tif')
                dstfile = "%s/%s" % (dstdir, normalized_layerid + '__subsetted.tif')
                west = command + ["-projwin", '-180', bbox[3], bbox[2], bbox[1], srcfile, west_dstfile]
                east = command + ["-projwin", bbox[0], bbox[3], '180', bbox[1], srcfile, east_dstfile]
                self.cmd(*west)
                self.cmd(*east)
                self.cmd('gdal_merge.py',
                    '-o', dstfile,
                    '-of', "GTiff",
                    east_dstfile,
                    west_dstfile)
                return dstfile

            command.extend(["-projwin", bbox[0], bbox[3], bbox[2], bbox[1]])

        dstfile = "%s/%s" % (dstdir, normalized_layerid + '__subsetted.tif')
        command.extend([srcfile, dstfile])
        self.cmd(*command)
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
            result.append(ObjectView({ "name": re.split(r":", subdataset)[-1] }))
        if len(result) > 0:
            return result

        # GeoTIFFs, where variables are bands, with descriptions set to their variable name
        for subdataset in filter((lambda line: re.match(r"^\s*Description = ", line)), gdalinfo_lines):
            result.append(ObjectView({ "name": re.split(r" = ", subdataset)[-1] }))
        return result

    def is_geotiff(self, filename):
        gdalinfo_lines = self.cmd("gdalinfo", filename)
        return gdalinfo_lines[0] == "Driver: GTiff/GeoTIFF"
