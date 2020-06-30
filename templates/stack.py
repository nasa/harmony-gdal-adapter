from osgeo import gdal
outvrt = 'stacked.vrt' #/vsimem is special in-memory virtual "directory"
outtif = 'stacked.tif'
tifs = ['a.tif', 'b.tif', 'c.tif', 'd.tif'] 
#or for all tifs in a dir
#import glob
#tifs = glob.glob('dir/*.tif')

outds = gdal.BuildVRT(outvrt, tifs, separate=True)
outds = gdal.Translate(outtif, outds)
