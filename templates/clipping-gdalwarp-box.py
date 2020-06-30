import os
#import fiona
#import rasterio
#import rasterio.mask
#from shapely.geometry import box
from osgeo import gdal, osr
import math
from affine import Affine

def subset2(tiffile, bbox, shapefile=None):

    RasterFormat = 'GTiff'

    raw_file_name = os.path.splitext(os.path.basename(tiffile))[0]
    
    #output_dir= os.path.dirname(tiffile)

    output_file_base = raw_file_name + "_" + "subset_gdalwarp" + ".tif"

    output_file_base2 = raw_file_name + "_" + "subset_dalwarp_shift" + ".tif"

    output_file = os.path.join(output_dir, output_file_base)

    output_file2 = os.path.join(output_dir, output_file_base2)
    

    [ll_lon, ll_lat,ur_lon, ur_lat]=bbox

    #covert bbox to tiffile's coordinator
    src = osr.SpatialReference()

    src.SetWellKnownGeogCS("WGS84")

    dataset = gdal.Open(tiffile)
    
    transform=dataset.GetGeoTransform()
  
    xRes=transform[1]

    yRes=transform[5]

    projection = dataset.GetProjection()

    dst = osr.SpatialReference(projection)
    
    ct = osr.CoordinateTransformation(src, dst)

    llxy = ct.TransformPoint(ll_lon, ll_lat)

    urxy = ct.TransformPoint(ur_lon, ur_lat)

    shapes=[ llxy[0],llxy[1], urxy[0],urxy[1] ]
   
    #calculate the extent

    extent,dltx,dlty = calcualte_extent(dataset)
    
    #ll,ul,ur,lr=extent

    #minlon=min(ll[0],ul[0],ur[0],lr[0])
    #maxlon=max(ll[0],ul[0],ur[0],lr[0])

    #minlat=min(ll[1],ul[1],ur[1],lr[1])
    #maxlat=max(ll[1],ul[1],ur[1],lr[1])

    #shapes=[ minlon, minlat, maxlon, maxlat]

    #options=gdal.WarpOptions(transformerOptions=transform, format='GTiff',copyMetadata=True)
    # Create raster
    
    gdal.Warp(output_file, dataset, format=RasterFormat, outputBounds=shapes, xRes=xRes, yRes=yRes, dstSRS=projection, resampleAlg=gdal.GRA_NearestNeighbour, options=['COMPRESS=DEFLATE'])
    #out_dataset = gdal.Warp(output_file, dataset, format=RasterFormat, outputBounds=shapes, options=options)

    #out_dataset = gdal.Warp(output_file, dataset, format=RasterFormat, outputBounds=shapes)

    dataset=None
   
    #read the outout_file back

    out_dataset =gdal.Open(output_file)

    driver = gdal.GetDriverByName('GTiff')

    dst_ds=driver.CreateCopy(output_file2, out_dataset, 0)

    out_gt=dst_ds.GetGeoTransform()
    
    out_ncol=dst_ds.RasterXSize

    out_nrow=dst_ds.RasterXSize

    #shift the ul 

    out_gt_lst2=list(out_gt)

    out_gt_lst2[0]=out_gt_lst2[0]+dltx

    out_gt_lst2[3]=out_gt_lst2[3]+dlty

    out_gt2=tuple(out_gt_lst2)

    dst_ds.SetGeoTransform(out_gt2)

    driver = gdal.GetDriverByName('GTiff')

    dst_ds=driver.CreateCopy(output_file2, dst_ds,0)

    dataset = None # Close dataset

    dst_ds=None

    #with fiona.open("tests/data/box.shp", "r") as shapefile:
    #    shapes = [feature["geometry"] for feature in shapefile]

    #with rasterio.open(tiffile) as src:
    #    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    
    #out_meta = src.meta

    #out_meta.update({"driver": "GTiff",
    #             "height": out_image.shape[1],
    #             "width": out_image.shape[2],
    #             "transform": out_transform})

    #with rasterio.open(output_file, "w", **out_meta) as dest:
    #    dest.write(out_image)

def calcualte_extent(ds):
    
    gt=ds.GetGeoTransform()
    
    ncol = ds.RasterXSize

    nrow = ds.RasterYSize

    transform = Affine.from_gdal(*gt)

    ul=transform.c, transform.f #upperleft

    ll=transform * (0, nrow)  # lowerleft

    lr=transform * (ncol, nrow)    # lower right

    ur=transform * (ncol, 0)     # upper right

    extent=[ll, ul, ur, lr]


    center=((ur[0]+ll[0])/2.0, (ur[1]+ll[1])/2.0)

    
    ref_angle_ulur=math.atan( (ur[1]-ul[1])/(ur[0]-ul[0]) )

    dlty=gt[1]*math.sin(ref_angle_ulur)   

    dltx=gt[1]*math.cos(ref_angle_ulur)

    #ul2=( center[0]-dltx, center[1]-dlty )

    #lr2=( center[0]+dltx, center[1]+dlty )


    #ur2=( center[0]+dlty, center[1]+dlty)

    #ll2=( center[0]-dltx, center[1]-dlty)

        
    #xsize=(ur2[0]-ul2[0])/ncol

    #ysize=(lr2[1]-ur2[1])/nrow



    return extent, dltx,dlty



if __name__=="__main__":

    filename="/home/jzhu4/projects/work/harmony-curr/sampledata/avnir/IMG-01-ALAV2A279143000-OORIRFU_000.tif"
    bbox=[-96.898, 29.738, -96.636, 29.876]
    subset2(filename,bbox)
    print("complete...")


