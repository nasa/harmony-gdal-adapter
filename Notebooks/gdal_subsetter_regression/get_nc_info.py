import netCDF4
import numpy as np

def get_nc_info(collection, outfile):

    def findnth(string, substring, n):
        parts = string.split(substring, n + 1)
        if len(parts) <= n + 1:
            return -1
        return len(string) - len(parts[-1]) - len(substring)

    f = netCDF4.Dataset(outfile)
    extent = f.extent #old metadata
    #extent=f.GDAL_extent #new metadata
    ul_start=findnth(extent, '[', 0)
    ul_end=findnth(extent, ']', 0)
    lr_start=findnth(extent, '[', 3)
    lr_end=findnth(extent, ']', 3)
    ul=extent[ul_start+1:ul_end].split(', ')
    lr=extent[lr_start+1:lr_end].split(', ')
    min_lon=round(float(ul[0]),2)
    max_lon=round(float(lr[0]),2)
    min_lat=round(float(lr[1]),2)
    max_lat=round(float(ul[1]),2)
    var_keys = f.variables.keys()
    keys = list(var_keys)
    band_keys=list(keys[1:])
    n_bands = len(band_keys)
    var_list = ['NA','NA','NA','NA','NA','NA','NA']
    band_index=0
    for band in band_keys:
        var_list[band_index]=band
        band_index += 1

    if keys[0] == 'crs':
        cs = 'Geographic'
        crs = f.variables['crs']
        sr = crs.spatial_ref
        authority_idx = sr.find('AUTHORITY')
        authority = sr[authority_idx+11:authority_idx+15]
        projcs = 'NA'
        gcs_idx = sr.find('GEOGCS')
        gcs = sr[gcs_idx+8:gcs_idx+14]
        proj_epsg = 'NA'
        gcs_epsg_idx = findnth(sr, 'EPSG', 3)
        gcs_epsg = sr[gcs_epsg_idx+7:gcs_epsg_idx+11]
        xy_size=[0,0]

    if keys[0] != 'crs':
        cs = 'Projected'
        proj = f.variables[keys[0]]
        sr = proj.spatial_ref
        authority_idx = sr.find('AUTHORITY')
        authority = sr[authority_idx+11:authority_idx+15]
        projcs_idx = sr.find('PROJCS')
        projcs = sr[projcs_idx+8:projcs_idx+29]
        gcs_idx = sr.find('GEOGCS')
        gcs = sr[gcs_idx+8:gcs_idx+14]
        proj_epsg_idx = sr.rfind('EPSG')
        proj_epsg = sr[proj_epsg_idx+7:proj_epsg_idx+12]
        gcs_epsg_idx = findnth(sr, 'EPSG', 4)
        gcs_epsg = sr[gcs_epsg_idx+7:gcs_epsg_idx+11]
        xy_size=[0,0]
        xy_dim_keys = f.dimensions.keys()
        xy_size[0] = f.dimensions['x'].size
        xy_size[1] = f.dimensions['y'].size

    extent =[min_lat, max_lat, min_lon, max_lon]

    information = {'gdal_cs': cs,
                   'gdal_proj_cs': projcs,
                   'gdal_gcs': gcs,
                   'gdal_authority': authority,
                   'gdal_proj_epsg': proj_epsg,
                   'gdal_gcs_epsg': gcs_epsg,
                   'gdal_spatial_extent': extent,
                   'gdal_n_bands' : n_bands,
                   'gdal_xy_size' : xy_size,
                   'gdal_variables' : var_list}
    
    return(information)
