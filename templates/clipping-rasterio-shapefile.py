import os
import fiona
import rasterio
import rasterio.mask
from shapely.geometry import box
from osgeo import gdal, osr

def subset2(tiffile, bbox, shapefile=None):

    raw_file_name = os.path.splitext(os.path.basename(tiffile))[0]

    output_file_base = raw_file_name + "_" + "subsection" + ".tif"

    output_file = os.path.join("data", raw_file_name, output_file_base)

    [ll_lon, ll_lat,ur_lon, ur_lat]=bbox

    #covert bbox to tiffile's coordinator
    src = osr.SpatialReference()

    src.SetWellKnownGeogCS("WGS84")

    dataset = gdal.Open(tiffile)
    projection = dataset.GetProjection()
    dst = osr.SpatialReference(projection)
    
    ct = osr.CoordinateTransformation(src, dst)

    llxy = ct.TransformPoint(ll_lon, ll_lat)

    urxy = ct.TransformPoint(ur_lon, ur_lat)

    shapes=box( llxy[0],llxy[1], urxy[0],urxy[1] )

    dataset.close()
    dst.close()
    src.close()

    #with fiona.open("tests/data/box.shp", "r") as shapefile:
    #    shapes = [feature["geometry"] for feature in shapefile]

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


