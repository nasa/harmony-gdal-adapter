#This script does unit test for transform.py
#Input is the meassage


import pytest
import os

########################
#in debug mode
import pdb
pdb.set_trace()
########################


#define the input as fixture

messagestr='{"sources":[{"collection":"C1225776654-ASF","variables":[{"id":"V1234600145-ASF","name":"/science/grids/data/connectedComponents","fullPath":"/science/grids/data/connectedComponents"}],"granules":[{"id":"G1234646236-ASF","name":"S1-GUNW-A-R-166-tops-20200313_20200206-014119-34455N_32574N-PP-1749-v2_0_2","bbox":[-116.8219161,32.465775,-113.7630671,34.492821],"temporal":{"start":"2020-03-13T01:41:06.357Z","end":"2020-03-13T01:41:33.312Z"},"url":"https://grfn-test.asf.alaska.edu/door/download/S1-GUNW-A-R-166-tops-20200313_20200206-014119-34455N_32574N-PP-1749-v2_0_2.nc"}]}],"format":{"mime":"image/tiff"},"subset":{},"requestId":"61164a96-b7ef-4099-b8fb-70e24e1989c0","user":"cirrusasf","client":"harmony-local","isSynchronous":true,"stagingLocation":"s3://local-staging-bucket/public/harmony/gdal-subsetter/b1ff5919-0850-4619-99d3-927eddfa2f72/","callback":"http://host.docker.internal:3001/service/61164a96-b7ef-4099-b8fb-70e24e1989c0","version":"0.8.0"}'

#test functions in adapter




def test_gfrn(adapter):
    #messagestring='{"sources":[{"collection":"C1225776654-ASF","variables":[{"id":"V1234600145-ASF","name":"/science/grids/data/connectedComponents","fullPath":"/science/grids/data/connectedComponents"}],"granules":[{"id":"G1234646236-ASF","name":"S1-GUNW-A-R-166-tops-20200313_20200206-014119-34455N_32574N-PP-1749-v2_0_2","bbox":[-116.8219161,32.465775,-113.7630671,34.492821],"temporal":{"start":"2020-03-13T01:41:06.357Z","end":"2020-03-13T01:41:33.312Z"},"url":"https://grfn-test.asf.alaska.edu/door/download/S1-GUNW-A-R-166-tops-20200313_20200206-014119-34455N_32574N-PP-1749-v2_0_2.nc"}]}],"format":{"mime":"image/tiff"},"subset":{},"requestId":"61164a96-b7ef-4099-b8fb-70e24e1989c0","user":"cirrusasf","client":"harmony-local","isSynchronous":true,"stagingLocation":"s3://local-staging-bucket/public/harmony/gdal-subsetter/b1ff5919-0850-4619-99d3-927eddfa2f72/","callback":"http://host.docker.internal:3001/service/61164a96-b7ef-4099-b8fb-70e24e1989c0","version":"0.8.0"}'
    adapter = adapter(messagestr)

    message = adapter.message

    logger = adapter.logger

    if message.subset and message.subset.shape:
        logger.warn('Ignoring subset request for user shapefile %s' %
                        (message.subset.shape.href,))

    try:
        # Limit to the first granule.  See note in method documentation
        granules = message.granules
        if message.isSynchronous:
            granules = granules[:1]

        output_dir = "tmp/data"
        adapter.prepare_output_dir(output_dir)

        layernames = []

        operations = dict(
            is_variable_subset=True,
            is_regridded=bool(message.format.crs),
            is_subsetted=bool(message.subset and message.subset.bbox)
            )

        result = None
        for i, granule in enumerate(granules):
            adapter.download_granules([granule])
            
            assert os.path.exists(granule.local_filename)

            file_type = adapter.get_filetype(granule.local_filename)
            if file_type == 'tif':
                layernames, result = adapter.process_geotiff(
                            granule,output_dir,layernames,operations,message.isSynchronous
                            )
            elif file_type == 'nc':
                layernames, result = adapter.process_netcdf(
                            granule,output_dir,layernames,operations,message.isSynchronous
                            )
            elif file_type == 'zip':
                layernames, result = adapter.process_zip(
                            granule,output_dir,layernames,operations,message.isSynchronous
                             )
            else:
                logger.exception(e)
                adapter.completed_with_error('No reconized file foarmat, not process')

            #test the result
            assert result

            if not message.isSynchronous:
                # Send a single file and reset
                adapter.update_layernames(result, [v.name for v in granule.variables])
                result = adapter.reformat(result, output_dir)
                progress = int(100 * (i + 1) / len(granules))
                adapter.async_add_local_file_partial_result(result, source_granule=granule, title=granule.id, progress=progress, **operations)
                adapter.cleanup()
                adapter.prepare_output_dir(output_dir)
                layernames = []
                result = None



        if message.isSynchronous:
            adapter.update_layernames(result, layernames)
            result = adapter.reformat(result, output_dir)

            #adapter.completed_with_local_file(
            #        result, source_granule=granules[-1], **operations)

            print("adapter.completed_with_local_file()")

        else:
            #adapter.async_completed_successfully()
            print("adapter.async_completed_successfully()")

    except Exception as e:
        logger.exception(e)
        adapter.completed_with_error('An unexpected error occurred')

    finally:
        adapter.cleanup()

