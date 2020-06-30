InputImage = 'XXX.tif'
Shapefile = 'XXX.shp'
RasterFormat = 'GTiff'
PixelRes = 0.5
VectorFormat = 'ESRI Shapefile'

# Open datasets
Raster = gdal.Open(InputImage, gdal.GA_ReadOnly)
Projection = Raster.GetProjectionRef()

VectorDriver = ogr.GetDriverByName(VectorFormat)
VectorDataset = VectorDriver.Open(Shapefile, 0) # 0=Read-only, 1=Read-Write
layer = VectorDataset.GetLayer()
FeatureCount = layer.GetFeatureCount()
print("Feature Count:",FeatureCount)

# Iterate through the shapefile features
Count = 0
for feature in layer:
    Count += 1
    print("Processing feature "+str(Count)+" of "+str(FeatureCount)+"...")

    geom = feature.GetGeometryRef() 
    minX, maxX, minY, maxY = geom.GetEnvelope() # Get bounding box of the shapefile feature

    # Create raster
    OutTileName = str(Count)+'.SomeTileName.tif'
    OutTile = gdal.Warp(OutTileName, Raster, format=RasterFormat, outputBounds=[minX, minY, maxX, maxY], xRes=PixelRes, yRes=PixelRes, dstSRS=Projection, resampleAlg=gdal.GRA_NearestNeighbour, options=['COMPRESS=DEFLATE'])
    OutTile = None # Close dataset

# Close datasets
Raster = None
VectorDataset.Destroy()
print("Done.")
