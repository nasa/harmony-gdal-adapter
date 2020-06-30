import os

from osgeo import gdal,osr


def subsetbbox(file_name,bbox):
    
    [ll_lon, ll_lat,ur_lon, ur_lat]=bbox

    raw_file_name = os.path.splitext(os.path.basename(file_name))[0]

    output_dir=os.path.dirname(file_name)

    output_file_base = raw_file_name + "_" + "subsection" + ".tif"

    output_file=os.path.join(output_dir, output_file_base)

    driver = gdal.GetDriverByName('GTiff')
    dataset = gdal.Open(file_name)
    band = dataset.GetRasterBand(1)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    
    #change bbox from WRG84 to geotiff file's coordinators

    src = osr.SpatialReference()
    src.SetWellKnownGeogCS("WGS84")
    projection = dataset.GetProjection();
    dst = osr.SpatialReference(projection);
    
    ct = osr.CoordinateTransformation(src, dst)
        
    transform=dataset.GetGeoTransform()

    ulxy = ct.TransformPoint(ll_lon, ur_lat)

    lrxy = ct.TransformPoint(ur_lon, ll_lat)

    minx = transform[0]
    maxx = transform[0] + cols * transform[1] + rows * transform[2]
    miny = transform[3] + cols * transform[4] + rows * transform[5]
    maxy = transform[3]
    width = maxx - minx
    height = maxy - miny

    #tiles = create_tiles(minx, miny, maxx, maxy, n)

    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    print(xOrigin, yOrigin)

    # Subsitute with your new subsection values

    #newminx = llxy[0] 
    #newmaxx = urxy[0]
    #newminy = llxy[1]
    #newmaxy = urxy[1]

    #p1 = (newminx, newmaxy)  #upperleft, ulxy
    #p2 = (newmaxx, newminy)  #lowerright, lrxy

    i1 = int((ulxy[0] - xOrigin) / pixelWidth)
    j1 = int((yOrigin - ulxy[1])  / pixelHeight)
    i2 = int((lrxy[0] - xOrigin) / pixelWidth)
    j2 = int((yOrigin - lrxy[1]) / pixelHeight)

    print(i1, j1)
    print(i2, j2)

    new_cols = i2-i1
    new_rows = j2-j1

    data = band.ReadAsArray(i1, j1, new_cols, new_rows)

    #print data

    new_x = xOrigin + i1*pixelWidth
    new_y = yOrigin - j1*pixelHeight

    print(new_x, new_y)

    new_transform = (new_x, transform[1], transform[2], new_y, transform[4], transform[5])

    dst_ds = driver.Create(output_file,
                           new_cols,
                           new_rows,
                           1,
                           gdal.GDT_Float32)

    #writting output raster
    dst_ds.GetRasterBand(1).WriteArray( data )

    #setting extension of output raster
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform(new_transform)

    wkt = dataset.GetProjection()

    # setting spatial reference of output raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection( srs.ExportToWkt() )

    #Close output raster dataset
    dst_ds = None

    dataset = None


    return output_file

if __name__ == "__main__":

    filename="/home/jzhu4/projects/work/harmony-curr/sampledata/avnir/IMG-01-ALAV2A279143000-OORIRFU_000.tif"
    bbox=[-96.898, 29.738, -96.636, 29.87]
    subsetbbox(filename, bbox) 
