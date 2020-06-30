import os
import math
from affine import Affine
#import fiona
#import rasterio
#import rasterio.mask
#from shapely.geometry import box
from osgeo import gdal, osr

def subset2(tiffile, bbox, shapefile=None):

    RasterFormat = 'GTiff'

    raw_file_name = os.path.splitext(os.path.basename(tiffile))[0]
    
    output_dir= os.path.dirname(tiffile)

    output_file_base = raw_file_name + "_" + "subsetted" + ".tif"

    output_file = os.path.join(output_dir, output_file_base)

    #[ll_lon, ll_lat,ur_lon, ur_lat]=bbox

    dataset = gdal.Open(tiffile)
    src = osr.SpatialReference()
    src.SetWellKnownGeogCS("WGS84")

    #covert bbox to tiffile's coordinator
    def convert_coord(src, dataset, bbox):

        #src = osr.SpatialReference()
        #src.SetWellKnownGeogCS("WGS84")

        #dataset = gdal.Open(tiffile)
    
        gt=dataset.GetGeoTransform()

        projection = dataset.GetProjection()

        dst = osr.SpatialReference(projection)

        ct = osr.CoordinateTransformation(src, dst)

        [ll_lon, ll_lat,ur_lon, ur_lat]=bbox

        llxyz = ct.TransformPoint(ll_lon, ll_lat)

        urxyz = ct.TransformPoint(ur_lon, ur_lat)

        box_dst=[llxyz[0],urxyz[1], urxyz[0], llxyz[1]]
    
        return box_dst

    #rotate the rotation gt to no-rotation-gt
    def pixel2geocoord(GT,Xpixel,Yline):
        Xgeo = GT[0] + Xpixel*GT[1] + Yline*GT[2]
        Ygeo = GT[3] + Xpixel*GT[4] + Yline*GT[5]
        return (Xgeo,Ygeo)

    def rotate2horiz(ds):
        #rotate the rotation geotransform to non-geotransform
        ncol = ds.RasterXSize
        nrow = ds.RasterYSize
        gt=ds.GetGeoTransform()

        transform = Affine.from_gdal(*gt)

        ul=transform.c, transform.f #upperleft

        ll=transform * (0, nrow)  # lowerleft

        lr=transform * (ncol, nrow)    # lower right

        ur=transform * (ncol, 0)     # upper right

        center=((ur[0]+ll[0])/2.0, (ur[1]+ll[1])/2.0)

        #rotate the geotransform along the center
    
        xleng=math.dist(ul,ur)

        yleng=math.dist(ul,ll)

        ul2=( center[0]-xleng/2.0, center[1]+yleng/2.0)

        ur2=( center[0]+xleng/2.0, center[1]+yleng/2.0)

        ll2=( center[0]-xleng/2.0, center[1]-yleng/2.0)

        lr2=( center[0]+xleng/2.0, center[1]-yleng/2.0)

        xsize=xleng/ncol

        ysize=-yleng/nrow
        
        #gt2

        gt2=(ul2[0], xsize, 0, ul2[1], 0, ysize) 

        return gt2
    

    def horiz2rotate(ds, ref_gt):

        #rotate the ds according to ref_gt

        ncol = ds.RasterXSize

        nrow = ds.RasterYSize
    
        gt=ds.GetGeoTransform()

        transform = Affine.from_gdal(*ref_gt)

        ref_ul=transform.c, transform.f #upperleft

        ref_ll=transform * (0, nrow)  # lowerleft

        ref_lr=transform * (ncol, nrow)    # lower right

        ref_ur=transform * (ncol, 0)     # upper right

        ref_center=((ref_ur[0]+ref_ll[0])/2.0, (ref_ur[1]+ref_ll[1])/2.0)

        ref_diagleng=math.dist(ref_lr,ref_ul)

        ref_angle=math.atan( (ref_lr[1]-ref_ul[1])/(ref_lr[0]-ref_ul[0]) )
    
        ref_angle_ullr=math.atan( (ref_lr[1]-ref_ul[1])/(ref_lr[0]-ref_ul[0]) )

        ref_angle_llur=math.atan( (ref_ur[1]-ref_ll[1])/(ref_ur[0]-ref_ll[0]) )


        transform = Affine.from_gdal(*gt)

        ul=transform.c, transform.f #upperleft

        ll=transform * (0, nrow)  # lowerleft

        lr=transform * (ncol, nrow)    # lower right

        ur=transform * (ncol, 0)     # upper right

        center=((ur[0]+ll[0])/2.0, (ur[1]+ll[1])/2.0)

        diagleng=math.dist(lr,ul)

        dlty=0.5*diagleng*math.sin(ref_angle_ullr)   

        dltx=0.5*diagleng*math.cos(ref_angle_ullr)

        ul2=( center[0]-dltx, center[1]-dlty )

        lr2=( center[0]+dltx, center[1]+dlty )


        dlty=0.5*diagleng*math.sin(ref_angle_llur)

        dltx=0.5*diagleng*math.cos(ref_angle_ullr)


        ur2=( center[0]+dlty, center[1]+dlty)

        ll2=( center[0]-dltx, center[1]-dlty)

        
        xsize=(ur2[0]-ul2[0])/ncol

        ysize=(lr2[1]-ur2[1])/nrow

        #gt2

        gt2=(ul2[0], ref_gt[1], ref_gt[2], ul2[1], ref_gt[4], ref_gt[5])

        return gt2

    no_r_gt = rotate2horiz(dataset)

    gt=dataset.GetGeoTransform()

    no_r_gt=(gt[0],gt[1],0.0, gt[3],0.0, gt[5])


    #copy to a temoarary dataset

    driver = gdal.GetDriverByName('GTiff')

    tmp_ds=driver.CreateCopy('MEM', dataset, 0)

    tmp_ds.SetGeoTransform(no_r_gt)
    

    #output tmp_change_gt.tif

    tmp3_ds=driver.CreateCopy('tmp_change_gt.tif', tmp_ds, 0)    

    #call translate to subset

    tmp2_ds=gdal.Open('tmp_change_gt.tif')

    box_dst=convert_coord(src, tmp2_ds, bbox)

    #box_dst=[llxyz[0],urxyz[1], urxyz[0], llxyz[1]]
    #shapes=[ llxy[0],llxy[1], urxy[0],urxy[1] ]

    shapes=[ box_dst[0],box_dst[3], box_dst[2],box_dst[1] ]

    #out_ds=gdal.Translate('tmp_change_gt_clip.tif', tmp2_ds, format=RasterFormat, projWin=box_dst)

    out_ds = gdal.Warp('tmp_change_gt_clip.tif', tmp2_ds, format=RasterFormat, outputBounds=box_dst)

    #change back to rotation gt
    
    ref_gt=dataset.GetGeoTransform()

    out_gt=horiz2rotate(out_ds, ref_gt)

    out_ds.SetGeoTransform(out_gt)

    #output the out_ds
    
    driver = gdal.GetDriverByName('GTiff')

    dst_ds=driver.CreateCopy(output_file, out_ds,0)

    return output_file
    
    #options=gdal.WarpOptions(transformerOptions=transform, format='GTiff',copyMetadata=True)
    # Create raster
    
    #out_dataset = gdal.Warp(output_file, dataset, format=RasterFormat, outputBounds=shapes, xRes=xRes, yRes=yRes, dstSRS=projection, resampleAlg=gdal.GRA_NearestNeighbour, options=['COMPRESS=DEFLATE'])
    #out_dataset = gdal.Warp(output_file, dataset, format=RasterFormat, outputBounds=shapes, options=options)





if __name__=="__main__":

    filename="/home/jzhu4/projects/work/harmony-curr/sampledata/avnir/IMG-01-ALAV2A279143000-OORIRFU_000.tif"
    bbox=[-96.898, 29.738, -96.636, 29.876]
    subset2(filename,bbox)
    print("complete...")


