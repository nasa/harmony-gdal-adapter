import os
import fiona
import rasterio
import rasterio.mask
#from shapely.geometry import box
from osgeo import gdal, ogr, osr


def boxwrs84_boxproj(boxwrs84, ref_ds):
    #boxwrs84[min_lon,min_lat, max_lon, max_lat], ref_ds is reference dataset
    #return boxprj, box in reference projection

    src = osr.SpatialReference()

    src.SetWellKnownGeogCS("WGS84")


    transform=ref_ds.GetGeoTransform()

    xRes=transform[1]

    yRes=transform[5]

    projection = ref_ds.GetProjection()

    dst = osr.SpatialReference(projection)

    ct = osr.CoordinateTransformation(src, dst)

    ll_lon,ll_lat = boxwrs84[0],boxwrs84[1]
    
    ur_lon,ur_lat = boxwrs84[2],boxwrs84[3]

    llxy = ct.TransformPoint(ll_lon, ll_lat)

    urxy = ct.TransformPoint(ur_lon, ur_lat)
 
    boxproj=[ llxy[0],llxy[1], urxy[0],urxy[1] ]

    return boxproj, projection


def box2shapefile(box,projection,shapefile):

    #input: box=[min_lon,min_lat,max_lon,max_lat]
    #output: polygon geometry

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

    outLayer = outDataSource.CreateLayer('boundingbox',outSpatialRef, geom_type=ogr.wkbPolygon )

    featureDefn = outLayer.GetLayerDefn()

    # add new geom to layer
    outFeature = ogr.Feature(featureDefn)

    outFeature.SetGeometry(polygon)
    
    outLayer.CreateFeature(outFeature)
    
    outFeature.Destroy
    
    outDataSource.Destroy()



def subset2(tiffile, bbox, shapefile=None):

    #bbox=[min_lon,min_lat, max_lon,max_lat]

    RasterFormat = 'GTiff'

    raw_file_name = os.path.splitext(os.path.basename(tiffile))[0]
    
    output_dir= os.path.dirname(tiffile)

    output_file_base = raw_file_name + "_" + "subsection" + ".tif"

    output_file = os.path.join(output_dir, output_file_base)

    
    ref_ds=gdal.Open(tiffile)

    boxproj, proj = boxwrs84_boxproj(bbox, ref_ds)

    shapes=box2shapefile(boxproj,proj,'box.shp')

    
    #options=gdal.WarpOptions(transformerOptions=transform, format='GTiff',copyMetadata=True)
    # Create raster
    
    #out_dataset = gdal.Warp(output_file, dataset, format=RasterFormat, outputBounds=shapes, xRes=xRes, yRes=yRes, dstSRS=projection, resampleAlg=gdal.GRA_NearestNeighbour, options=['COMPRESS=DEFLATE'])
    #out_dataset = gdal.Warp(output_file, dataset, format=RasterFormat, outputBounds=shapes, options=options)


    #dataset = None # Close dataset

    with fiona.open("box.shp", "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(tiffile) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    
    out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(out_image)

if __name__=="__main__":

    filename="/home/jzhu4/projects/work/harmony-curr/sampledata/avnir/IMG-01-ALAV2A279143000-OORIRFU_000.tif"
    bbox=[-96.898, 29.738, -96.636, 29.876]
    subset2(filename,bbox)
    print("complete...")


