import rasterio

from rasterio.plot import show

from rasterio.plot import show_hist

from rasterio.mask import mask

from shapely.geometry import box

import geopandas as gpd

from fiona.crs import from_epsg

import pycrs

fp = r"C:\HY-DATA\HENTENKA\CSC\Data\p188r018_7t20020529_z34__LV-FIN.tif"

out_tif = r"C:\HY-DATA\HENTENKA\CSC\Data\Helsinki_masked_p188r018_7t20020529_z34__LV-FIN.tif"

data = rasterio.open(fp)

 # WGS84 coordinates

minx, miny = 24.60, 60.00

maxx, maxy = 25.22, 60.35

bbox = box(minx, miny, maxx, maxy)


geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))


geo = geo.to_crs(crs=data.crs.data)


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

coords = getFeatures(geo)

out_img, out_transform = mask(raster=data, shapes=coords, crop=True)

out_meta = data.meta.copy()

epsg_code = int(data.crs.data['init'][5:])

out_meta.update({"driver": "GTiff",
                  "height": out_img.shape[1],
                  "width": out_img.shape[2],
                  "transform": out_transform,
                  "crs": pycrs.parser.from_epsg_code(epsg_code).to_proj4()}
                          )


with rasterio.open(out_tif, "w", **out_meta) as dest:
     dest.write(out_img)






