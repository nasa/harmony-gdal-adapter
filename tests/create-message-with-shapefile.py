#!/home/jzhu4/apps/anaconda3/envs/har-gdal-env/bin/python
# coding: utf-8

# # Harmony API Introduction
# 
# This notebook provides an overview of the capabilities offered through the Harmony API, which supports the [OpenGIS Web Map Service](https://www.ogc.org/standards/wms#overview) and the [OGC API - Coverages](https://github.com/opengeospatial/ogc_api_coverage) specification. The examples below demonstrate synchronous and asynchronous access of several subsetting and reprojection services available from the Harmony/gdal demo service, native data access for data without transformation services, and the WMS map image service. 
# 
# Authors: Amy Steiker, Patrick Quinn

# ## Import packages
# 
# Most packages below should be included natively with the Anaconda Python distribution, for example, but some may need to install packages like `rasterio` manually using the following example:

# In[1]:


# Install prerequisite packages
import sys

#get_ipython().system('{sys.executable} -m pip install rasterio OWSLib # Install a pip package in the current Jupyter kernel')


# In[2]:

from pathlib import Path
from urllib import request, parse
from http.cookiejar import CookieJar
import getpass
import netrc
import tempfile
import requests
import json
import pprint
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rasterio
from rasterio.plot import show
import numpy as np
import os
import time
from netCDF4 import Dataset
from owslib.wms import WebMapService

#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Local directory setup 
# 
# Specify a local directory where the following Harmony outputs will be saved:

# In[3]:


# ---- Change this to save to a directory where you have write permissions
local_dir = tempfile.mkdtemp()



Path('local_dir' ).mkdir( parents=True, exist_ok=True )

# ## Earthdata Login Authentication
# 
# An Earthdata Login account is required to access data from NASA EOSDIS. In order to access data from the Harmony API, you will need to create an account in the Earthdata Login UAT environment. Please visit https://uat.urs.earthdata.nasa.gov to set up an account in this test environment. These accounts, as all Earthdata Login accounts, are free to create and only take a moment to set up.
# 
# 

# We need some boilerplate up front to log in to Earthdata Login.  The function below will allow Python scripts to log into any Earthdata Login application programmatically.  To avoid being prompted for
# credentials every time you run and also allow clients such as curl to log in, you can add the following
# to a `.netrc` (`_netrc` on Windows) file in your home directory:
# 
# ```
# machine uat.urs.earthdata.nasa.gov
#     login <your username>
#     password <your password>
# ```
# 
# Make sure that this file is only readable by the current user or you will receive an error stating
# "netrc access too permissive."
# 
# `$ chmod 0600 ~/.netrc` 
# 

# In[4]:


def setup_earthdata_login_auth(endpoint):
    """
    Set up the request library so that it authenticates against the given Earthdata Login
    endpoint and is able to track cookies between requests.  This looks in the .netrc file 
    first and if no credentials are found, it prompts for them.

    Valid endpoints include:
        uat.urs.earthdata.nasa.gov - Earthdata Login UAT (Harmony's current default)
        urs.earthdata.nasa.gov - Earthdata Login production
    """
    try:
        username, _, password = netrc.netrc().authenticators(endpoint)
    except (FileNotFoundError, TypeError):
        # FileNotFound = There's no .netrc file
        # TypeError = The endpoint isn't in the netrc file, causing the above to try unpacking None
        print('Please provide your Earthdata Login credentials to allow data access')
        print('Your credentials will only be passed to %s and will not be exposed in Jupyter' % (endpoint))
        username = input('Username:')
        password = getpass.getpass()

    manager = request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, endpoint, username, password)
    auth = request.HTTPBasicAuthHandler(manager)

    jar = CookieJar()
    processor = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(auth, processor)
    request.install_opener(opener)


# Now call the above function to set up Earthdata Login for subsequent requests

# In[5]:


#setup_earthdata_login_auth('uat.urs.earthdata.nasa.gov')
setup_earthdata_login_auth('urs.earthdata.nasa.gov')

# ## Identify a data collection of interest
# 
# A CMR collection ID is needed to request services through Harmony. The collection ID can be determined using the [CMR API](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html). We will query the corresponding ID of a known collection short name, `harmony_example`, which is a Level 3 test collection with transformation services available through Harmony.

# In[6]:

#'short_name': 'SENTINEL-1_INTERFEROGRAMS'

params = {
    'short_name': 'ALOS_AVNIR_OBS_ORI'
} # parameter dictionary with known CMR short_name

#cmr_collections_url = 'https://cmr.uat.earthdata.nasa.gov/search/collections.json'
cmr_collections_url = 'https://cmr.earthdata.nasa.gov/search/collections.json'
cmr_response = requests.get(cmr_collections_url, params=params)
cmr_results = json.loads(cmr_response.content) # Get json response from CMR collection metadata

collectionlist = [el['id'] for el in cmr_results['feed']['entry']]
harmony_collection_id = collectionlist[0]
print(harmony_collection_id)


# We can also view the `harmony_example` collection metadata to glean more information about the collection:

# In[7]:


pprint.pprint(cmr_results)


# ## Determine service availability
# 
# We will determine what services are available for the `harmony_example` collection based on the services.yml file available in the Harmony repository. 

# In[8]:


#os.chdir('..') # Move up a directory in harmony repo. Modify if you are not currently in this notebook directory within the Harmony repo. 
#yml_path = str(os.getcwd() + '/config/services.yml')

#with open(yml_path, 'r') as yml:
#    data = yml.read()
#    print(data)


# According to the services.yml, our `C1233800302-EEDTEST` collection is associated with the `harmony/gdal` service with bounding box and variable subsetting, reprojection, and reformatting. We will request these services below. 

# ## Explore the Harmony Root URL
# 
# Harmony conforms to the OGC API - Coverages specification: https://github.com/opengeospatial/ogc_api_coverages.
# 
# The basic Harmony URL convention is as follows:
# 
# `<harmony_root>/<collection_id>/ogc-api-coverages/1.0.0/`
# 
# We will set the Harmony root path with our chosen collection id:

# In[9]:


#harmony_root = 'https://harmony.uat.earthdata.nasa.gov'

harmony_root = 'http://localhost:3000'

config = {
    'collection_id': harmony_collection_id,
    'ogc-api-coverages_version': '1.0.0'
}
coverages_root = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/'.format(**config)
print('Request URL', coverages_root)


# This root URL of the coverages endpoint provides links to its child resources:

# In[10]:


root_response = request.urlopen(coverages_root)
root_results = root_response.read()
root_json = json.loads(root_results.decode('utf-8'))
pprint.pprint(root_json)


# The `service_desc` endpoint contains OpenAPI documentation, including information on all supported request parameters: 

# In[11]:


service_desc = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/api/'.format(**config)
service_response = request.urlopen(service_desc)
service_results = service_response.read()
service_txt = service_results.decode('utf-8') 
print(service_txt)


# The `conformance` endpoint provides the specifications this API conforms to:

# In[12]:


conform_desc = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/conformance/'.format(**config)
conform_response = request.urlopen(conform_desc)
conform_results = conform_response.read()
conform_json = json.loads(conform_results.decode('utf-8'))
print(conform_json)


# The `collections` endpoint provides metadata on the resource collections, which include variable metadata from CMR's [UMM-Var schema](https://git.earthdata.nasa.gov/projects/EMFD/repos/unified-metadata-model/browse/variable) in this example:

# In[13]:


collections_desc = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/'.format(**config)
collections_response = request.urlopen(collections_desc)
collections_results = collections_response.read()
collections_json = json.loads(collections_results.decode('utf-8'))
pprint.pprint(collections_json)

"""

# ## Access native data without transformation services

# For EOSDIS collections without associated Harmony transformation services, the Harmony API can still be utilized to access data through the provided data access links. Before we request services for `harmony_example`, We will use `ATL08`, or the "ATLAS/ICESat-2 L3A Land and Vegetation Height" data product, as an example of this "no processing" request. Note that this collection has restricted access and therefore cannot be queried for collection ID like above without passing Earthdata Login credentials. For convenience the collection ID was predetermined for this example: `C1229246405-NSIDC_TS1`.
# 
# The URL for requesting `ATLO8` is printed below. In this simple case, the entire data product is requested. The request response is also printed below, which includes information such as JobID, data access links, associated granule IDs, request messages, and status:

# In[14]:


noProcConfig = {
    'is2collection_id': 'C1229246405-NSIDC_TS1',
    'ogc-api-coverages_version': '1.0.0'
}

no_proc_url = harmony_root+'/{is2collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/all/coverage/rangeset'.format(**noProcConfig)
print('Request URL', no_proc_url)

no_proc_response = request.urlopen(no_proc_url)
no_proc_results = no_proc_response.read()
no_proc_json = json.loads(no_proc_results)
pprint.pprint(no_proc_json)


# Note that the request located all granules available in that collection.  We can pull those data access links from the request response:

# In[ ]:





# In[15]:


links = no_proc_json['links'] #list of links from response

for i in range(len(links)):
    link_dict = links[i] 
    print(link_dict['href'])


# Instead of requesting the entire `ATL08` collection, we can also specify a single granule ID to download. For convenience, the first granule returned in the request above is queried separately to demonstrate this single granule request:

# In[16]:


# Determine first granule ID in list
first_link_dict = links[0] 
granuleID = first_link_dict['title']
print(granuleID)

# Update noProcConfig
noProcConfig = {
    'is2collection_id': 'C1229246405-NSIDC_TS1',
    'ogc-api-coverages_version': '1.0.0',
    'variable': 'all',
    'granuleID': granuleID
}

no_proc_single_url = harmony_root+'/{is2collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?granuleID={granuleID}'.format(**noProcConfig)
print(no_proc_single_url)
no_proc_single_response = request.urlopen(no_proc_single_url)
no_proc_single_results = no_proc_single_response.read()
no_proc_single_json = json.loads(no_proc_single_results)
pprint.pprint(no_proc_single_json)


# The single file output is downloaded to a directory with write permissions:

# In[17]:


single_link = no_proc_single_json['links']
single_dict = single_link[0]
file_url = single_dict['href']

file_response = request.urlopen(file_url)
file_results = file_response.read()

# Write data to file 
file_name = 'harmonyNoProc.h5'
filepath = str(local_dir+file_name)
file_ = open(filepath, 'wb')
file_.write(file_results)
file_.close()


"""


# ## Access data subsetted by variable
# 
# Now we'll move into some subsetting examples with the `harmony_example` collection, beginning with a basic variable subset of a single pre-determined granule with global coverage. The variable request is included in the URL below as a /collections path. As stated in the API documentation, "This API interprets OGC 'collections' to be equivalent to CMR 'variables'". Unlike the no processing requests above, this result will be returned synchronously to us. By default, any single granule request that has associated Harmony services will be returned synchronously. 

# In[23]:


#varSubsetConfig = {
#    'collection_id': harmony_collection_id,
#    'ogc-api-coverages_version': '1.0.0',
#    'variable': 'blue_var',
#    'granuleid': 'G1233800343-EEDTEST'
#}
#var_url = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?granuleid={granuleid}'.format(**varSubsetConfig)
#print('Request URL', var_url)
#var_response = request.urlopen(var_url)
#var_results = var_response.read()


# This single subsetted file output is downloaded to the Harmony outputs directory:

# In[24]:


#file_name = 'harmonyvarsubset.tif'
#var_filepath = str(local_dir+file_name)
#file_ = open(var_filepath, 'wb')
#file_.write(var_results)
#file_.close()


# We can plot the TIF output of the single `blue_var` band to verify our output: 

# In[25]:


#var_raster = rasterio.open(var_filepath)
#blue = var_raster.read(1) # read first band, in this case blue_var
#plt.imshow(blue, cmap='Blues');


# ## Access data subsetted by geographic bounding box
# 
# Using the same global coverage granule as above, we will request a bounding box subset over Australia. Harmony supports spatial subset requests within the `rangeset` query in the following structure: 
# 
# `subset=lat(South:North)&subset=lon(West:East)`
# 
# More details included in the Harmony documentation:
# 
# Harmony supports the axes "lat" and "lon" for spatial subsetting, regardless of the names of those axes in the data files.  Examples:
# - Subset to the lat/lon bounding box with southwest corner (-10, -10) and northeast corner (10, 10)
#             subset=lat(-10:10)&subset=lon(-10:10)
# - Subset to all latitudes north of -10 degrees and all longitudes west of 10 degrees
#             subset=lat(-10:*)&subset=lon(*:10)
# - Subset to only points with latitudes from -10 to 10 degrees, disregarding longitude
#             subset=lat(-10:10)

# In[26]:


#http://localhost:3000/C1595422627-ASF/ogc-api-coverages/1.0.0/collections/science%2Fgrids%2Fdata%2Famplitude/coverage/rangeset?format=image%2Fgif&subset=lon(-70%3A-69)&subset=lat(-38%3A-37)&maxResults=1

#http://localhost:3000/C1808440897-ASF/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?granuleID=G1809522146-ASF&subset=lat(13.2:13.4)&subset=lon(32.9:33.1)&format=image%2Ftiff

#bboxSubsetConfig = {
#    'collection_id': harmony_collection_id,
#    'ogc-api-coverages_version': '1.0.0',
#    'variable': 'Band2',
#    'granuleid': 'G1809522146-ASF',
#    'lat': '(13.2:13.4)',
#    'lon': '(32.9:33.1)'
#}
#bbox_url = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?granuleid={granuleid}&subset=lat{lat}&subset=lon{lon}'.format(**bboxSubsetConfig)
#print('Request URL', bbox_url)
#bbox_response = request.urlopen(bbox_url)
#bbox_results = bbox_response.read()


# This spatially subsetted file output is downloaded to the Harmony outputs directory:

# In[27]:


#bbox_file_name = 'harmonybboxsubset.tif'
#bbox_filepath = str(local_dir+bbox_file_name)
#file_ = open(bbox_filepath, 'wb')
#file_.write(bbox_results)
#file_.close()


# We can plot the TIF output of the subsetted file to verify our output. All bands are overlaid to plot the color composite, with this code example modified from the following source:
# 
# https://automating-gis-processes.github.io/CSC/notebooks/L5/plotting-raster.html \
# © Copyright 2018, Henrikki Tenkanen \
# [License](https://creativecommons.org/licenses/by-sa/4.0/)

# In[28]:


# Open the file:
#bbox_raster = rasterio.open(bbox_filepath)

# Read the grid values into numpy arrays
#red = bbox_raster.read(3)
#green = bbox_raster.read(2)
#blue = bbox_raster.read(1)

# Function to normalize the grid values
#def normalize(array):
#    """Normalizes numpy arrays into scale 0.0 - 1.0"""
#    import numpy as np
#    np.seterr(divide='ignore', invalid='ignore') #ignore divide by 0 error
#    array_min, array_max = array.min(), array.max()
#    return ((array - array_min)/(array_max - array_min))

# Normalize the bands
#redn = normalize(red)
#greenn = normalize(green)
#bluen = normalize(blue)

# Create RGB natural color composite
#rgb = np.dstack((redn, greenn, bluen))

# Let's see how our color composite looks like
#plt.imshow(rgb);


# ## Access data filtered by temporal range
# 
# Filering data results by temporal range is also available on this test collection. According to the Harmony API documentation, the `time` keyword within the `rangeset` query supports the following:
# 
# Either a date-time or a period string that adheres to RFC 3339. Examples:
#         * A date-time: "2018-02-12T23:20:50Z" * A period: "2018-02-12T00:00:00Z/2018-03-18T12:31:12Z" or "2018-02-12T00:00:00Z/P1M6DT12H31M12S"
# Only collections that have a temporal property that intersects the value of `time` are selected. If a collection has multiple temporal properties, it is the decision of the server whether only a single temporal property is used to determine the extent or all relevant temporal properties.
# 
# We will search for the following time range:
# 
# Start time: 2020-01-16 02:00:00 \
# End time: 2020-01-16 03:00:00
# 
# According to [Earthdata Search](https://search.uat.earthdata.nasa.gov/search/granules?p=C1233800302-EEDTEST&g=G1233800479-EEDTEST&q=harmony_example&m=-66.80545903827544!19.313018908611213!1!1!0!0%2C2&qt=2020-01-16T02%3A00%3A00.000Z%2C2020-01-16T03%3A00%3A00.000Z&tl=1567098728!4!!), a single granule `016_01_ff0000_africa` is returned over this time.

# In[29]:


#timeSubsetConfig = {
#    'collection_id': harmony_collection_id,
#    'ogc-api-coverages_version': '1.0.0',
#    'variable': 'all',
#    'time': '("2020-01-16T02:00:00.000Z":"2020-01-16T03:00:00.000Z")'
#}

#time_url = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?&subset=time{time}'.format(**timeSubsetConfig)
#print('Request URL', time_url)
#time_response = request.urlopen(time_url)
#time_results = time_response.read()


# This file returned over the time range of interest is downloaded to the Harmony outputs directory:

# In[30]:


#time_file_name = 'harmonytimesubset.tif'
#time_filepath = str(local_dir+time_file_name)
#file_ = open(time_filepath, 'wb')
#file_.write(time_results)
#file_.close()


# We can plot the TIF output of this file to verify the coverage over Africa (for simplicity, plotting the first band):

# In[31]:


#time_raster = rasterio.open(time_filepath)
#time_band = time_raster.read(1)
#plt.imshow(time_band, cmap='Blues');


# ## Access data subsetted by geographic shapefile
# We will request data overlapping South America by uploading a shapefile with that boundary.
# 
# This requires the use of a multipart/form-data POST request. Supported shapefile formats include ESRI, GeoJSON, and KML. The associated mime-types are as follows:
# 
# | Shapefile Format | mime-type |
# |:-----------------|----------:|
# | ESRI | application/shapefile+zip |
# | GeoJSON | application/geo+json |
# | KML | application/vnd.google-earth.kml+xml |
# 
# 
# ESRI shapefiles must be uploaded as a single .zip file.
# See the Harmony documenation for more details.

shapefileSubsetConfig = {
    'collection_id': harmony_collection_id,
    'ogc-api-coverages_version': '1.0.0',
    'granuleid': 'G1809522146-ASF',
    'variable': 'Band1',
}
shapefile_url = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?granuleid={granuleid}'.format(**shapefileSubsetConfig)


#shapefileSubsetConfig = {
#    'collection_id': harmony_collection_id,
#    'ogc-api-coverages_version': '1.0.0',
#    'variable': 'Band2',
#}
#shapefile_url = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?'.format(**shapefileSubsetConfig)


#shapefile_path = str(os.getcwd() + '/docs/shapefiles/south_america.geojson')
shapefile_path = "/home/jzhu4/projects/work/harmony-curr/gdal-subsetter-dev/cases/shapefile/polygons/anvir2.zip"

print(shapefile_path)

with open(shapefile_path, 'rb') as fd1:
    # the form must have a 'shapefile' key which must include the mime-type as shown. Additional parameters
    # such as temporal subsetting can be included in the form.

    #KML file
    # multipart_form_data = {
    #     'shapefile': ('shape.geojson', fd1, 'application/vnd.google-earth.kml+xml'),
    #     'subset': (None, 'time("2020-01-01T20:00:00.000Z":"2020-01-01T22:00:00.000Z")')
    # }
    #start: "2020-03-13T01:41:06.357Z",
    #end: "2020-03-13T01:41:33.312Z"

    #ESRI shapefile
    #multipart_form_data = {
    #    'granuleid': ('G1809522146-ASF'),
    #    'shapefile': ('anvir2.zip', fd1, 'application/shapefile+zip'),
    #    'subset': (None, None)
    #}

    #ESRI shapefile
    multipart_form_data = {
        'shapefile': ('anvir2.zip', fd1, 'application/shapefile+zip'),
        'subset': ( None, None )
    }


    #KML file
    #multipart_form_data = {
    #    'shapefile': ('anvir2_anti.kml', fd1, 'application/vnd.google-earth.kml+xml'),
    #    'subset': (None, None)
    #}

    #geojson
    #multipart_form_data = {
    #    'shapefile': ('avnir-n3-anti.geojson', fd1, 'application/geo+json'),
    #    'subset': (None, None)
    #}

    print(multipart_form_data)
    # submit the form using a POST request and prepare to stream the result
    shapefile_response = requests.post(shapefile_url, files=multipart_form_data, stream=True)

print(shapefile_url)

sys.exit(0) 

# We can stream the result back and write it out to a file:

shapefile_output_filepath = str(local_dir + 'shapefile_output.tif')
with open(shapefile_output_filepath, 'wb') as fd:
    for chunk in shapefile_response.iter_content(chunk_size=128):
        fd.write(chunk)


# We plot the file as before:

# In[38]:


# Open the file:
shapefile_raster = rasterio.open(shapefile_output_filepath)

# Read the grid values into numpy arrays
red = shapefile_raster.read(3)
green = shapefile_raster.read(2)
blue = shapefile_raster.read(1)

# Function to normalize the grid values
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore') #ignore divide by 0 error
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

# Normalize the bands
redn = normalize(red)
greenn = normalize(green)
bluen = normalize(blue)

# Create RGB natural color composite
rgb = np.dstack((redn, greenn, bluen))

# Let's see how our color composite looks like
plt.imshow(rgb);


# ## Access reprojected data
# 
# The Harmony API accepts reprojection requests with a given coordinate reference system using the `outputCrs` keyword. According to the Harmony API documentation, this keyword "recognizes CRS types that can be inferred by gdal, including EPSG codes, Proj4 strings, and OGC URLs (http://www.opengis.net/def/crs/...) ". Two examples below demonstrate inputting an EPSG code and Proj4 string using the global test granule from previous examples. First, let's view the projection information of the granule in the native projection, using the variable subset example:

# In[ ]:


native_proj = gdal.Open(var_filepath, gdal.GA_ReadOnly)
native_proj.GetProjection()


# Request reprojection to EPSG 6933 ("WGS 84 / NSIDC EASE-Grid 2.0 Global"):

# In[ ]:


epsgConfig = {
    'collection_id': harmony_collection_id,
    'ogc-api-coverages_version': '1.0.0',
    'variable': 'all',
    'granuleid': 'G1233800343-EEDTEST',
    'outputCrs': 'EPSG:6933',
}

epsg_url = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?&granuleid={granuleid}&outputCrs={outputCrs}'.format(**epsgConfig)
print('Request URL', epsg_url)
epsg_response = request.urlopen(epsg_url)
epsg_results = epsg_response.read()


# This reprojected output is downloaded to the Harmony outputs directory and the projection information can be viewed using GDAL:

# In[ ]:


epsg_file_name = 'harmonyepsg.tif'
epsg_filepath = str(local_dir+epsg_file_name)
file_ = open(epsg_filepath, 'wb')
file_.write(epsg_results)
file_.close()

# get projection information
epsg = gdal.Open(epsg_filepath, gdal.GA_ReadOnly)
epsg.GetProjection()


# We can see that the output was reprojected to Cylindrical Equal Area as expected. We can do a visual check of this as well:

# In[ ]:


epsg_raster = rasterio.open(epsg_filepath)
epsg_band = epsg_raster.read(1)
plt.imshow(epsg_band, cmap='Blues');


# Reprojection can also be requested using a proj4 string. You must ensure that the proper URL encoding is included in the request so that proj4 string spaces and special characters are handled without error:

# In[ ]:


# URL encode string using urllib parse package
proj_string = '+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs' # proj4 of WGS 84 / NSIDC EASE-Grid 2.0 Global projection
proj_encode = parse.quote(proj_string)

projConfig = {
    'collection_id': harmony_collection_id,
    'ogc-api-coverages_version': '1.0.0',
    'variable': 'all',
    'granuleid': 'G1233800343-EEDTEST',
    'outputCrs': proj_encode
}

proj_url = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?&granuleid={granuleid}&outputCrs={outputCrs}'.format(**projConfig)
print('Request URL', proj_url)
proj_response = request.urlopen(proj_url)
proj_results = proj_response.read()


# This reprojected output is downloaded to a directory with write permissions and the projection information can be viewed using GDAL. The projection is equivalent to the specified EPSG request above as expected:

# In[ ]:


proj_file_name = 'harmonyproj4.tif'
proj_filepath = str(local_dir+proj_file_name)
file_ = open(proj_filepath, 'wb')
file_.write(proj_results)
file_.close()

# get projection information
proj = gdal.Open(proj_filepath, gdal.GA_ReadOnly)
proj.GetProjection()


# ## Access Level 2 swath regridded data
# 
# Moving outside of the `harmony/gdal` service, we will now request regridding from the `sds/swot-reproject` service using the `C1233860183-EEDTEST`, or Harmony L2 swath example, collection provided in NetCDF format. 
# 
# 
# The Harmony API accepts several query parameters related to regridding and interpolation in addition to the reprojection parameters above: 
# 
# `interpolation=<String>` - Both `near` and `bilinear` are valid options
# 
# `scaleSize=x,y` - 2 comma separated numbers as floats
# 
# `scaleExtent=xmin,ymin,xmax,ymax` - 4 comma separated numbers as floats
# 
# `width=<Float>`  
# 
# `height=<Float>` 
# 
# An error is returned if both `scaleSize` and `width`/`height` parameters are both provided (only one or the other can be used).

# Request reprojection to [Europe Lambert Conformal Conic](https://epsg.io/102014) with a new scale extent and nearest neighbor interpolation:

# In[ ]:


# URL encode string using urllib parse package
l2proj_string = '+proj=lcc +lat_1=43 +lat_2=62 +lat_0=30 +lon_0=10 +x_0=0 +y_0=0 +ellps=intl +units=m +no_defs'
l2proj_encode = parse.quote(proj_string)


regridConfig = {
    'l2collection_id': 'C1233860183-EEDTEST',
    'ogc-api-coverages_version': '1.0.0',
    'variable': 'all',
    'granuleid': 'G1233860486-EEDTEST',
    'outputCrs': l2proj_encode,
    'interpolation': 'near',
    'scaleExtent': '-7000000,1000000,8000000,8000000'
}

regrid_url = harmony_root+'/{l2collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?&granuleid={granuleid}&outputCrs={outputCrs}&interpolation={interpolation}&scaleExtent={scaleExtent}'.format(**regridConfig)
print('Request URL', regrid_url)
regrid_response = request.urlopen(regrid_url)
regrid_results = regrid_response.read()


# This reprojected and regridded output is downloaded to the Harmony outputs directory and we can inspect a variable to check for projection and grid dimension:

# In[ ]:


regrid_file_name = 'regrid.nc'
regrid_filepath = str(local_dir+regrid_file_name)
file_ = open(regrid_filepath, 'wb')
file_.write(regrid_results)
file_.close()

# Inspect dimensions of the blue_var:
regrid_nc = Dataset(regrid_filepath)
print(regrid_nc.variables.keys())
blue_var = regrid_nc.variables['blue_var'] 
print(blue_var) 


# Print the x and y dimensions to confirm that the output matches the requested scale extent in meters:

# In[ ]:


x = regrid_nc.variables['x'] 
y = regrid_nc.variables['y'] 
print('min x', min(x), 'max x', max(x))
print('min y', min(y), 'max y', max(y))


# ## Access multiple files from an asynchronous request
# 
# By default, a request resulting in more than one file will be returned asynchronously via a Job URL. The initial request submission is automatically redirected to this URL, and output links are appended to the response as they complete. The following query should return three granules based on the following temporal range:

# In[ ]:


asyncConfig = {
    'collection_id': harmony_collection_id,
    'ogc-api-coverages_version': '1.0.0',
    'variable': 'all',
    'time': '("2020-01-16T02:00:00.000Z":"2020-01-16T07:00:00.000Z")'
}

async_url = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?&subset=time{time}'.format(**asyncConfig)
print('Request URL', async_url)
async_response = request.urlopen(async_url)
async_results = async_response.read()
async_json = json.loads(async_results)
pprint.pprint(async_json)


# The async response initially shows 0% progress. The initial request URL will automatically redirect to a job URL, which we can manually determine using the jobID:

# In[ ]:


jobConfig = {
    'jobID': async_json['jobID']
}

job_url = harmony_root+'/jobs/{jobID}'.format(**jobConfig)
print('Job URL', job_url)


# The `links` list in the job response will continue to be updated as outputs are produced:

# In[ ]:


job_response = request.urlopen(job_url)
job_results = job_response.read()
job_json = json.loads(job_results)

print('Job response:')
print()
pprint.pprint(job_json)


# A loop can be set up to query the job status and download outputs once the job is complete:

# In[ ]:


#Continue loop while request is still processing
while job_json['status'] == 'running' and job_json['progress'] < 100: 
    print('Job status is running. Progress is ', job_json['progress'], '%. Trying again.')
    time.sleep(10)
    loop_response = request.urlopen(job_url)
    loop_results = loop_response.read()
    job_json = json.loads(loop_results)
    if job_json['status'] == 'running':
        continue

if job_json['status'] == 'successful' and job_json['progress'] == 100:
    print('Job progress is 100%. Output links printed below:')
    links = [link for link in job_json['links'] if link.get('rel', 'data') == 'data'] #list of data links from response
    for i in range(len(links)):
        link_dict = links[i] 
        print(link_dict['href'])
        output_file_name = str(link_dict['title']+'.tif')
        proj_filepath = str(local_dir+output_file_name)
        file_ = open(proj_filepath, 'wb')
        file_.write(proj_results)
        file_.close()


# ## Previewing a small number of results
# 
# By default, a request will return as many results as match the spatial and temporal query parameters, although this is subject to system limitations to prevent users from inadvertently overwhelming the system.  If a user wishes to further limit the number of results returned in order to preview a small number of results before requesting a larger transformation, they can simply supply a parameter called 'maxResults'.

# In[ ]:


asyncConfig = {
    'collection_id': harmony_collection_id,
    'ogc-api-coverages_version': '1.0.0',
    'variable': 'all',
    'maxResults': '2'
}

async_url = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?&maxResults={maxResults}'.format(**asyncConfig)
print('Request URL', async_url)
async_response = request.urlopen(async_url)
async_results = async_response.read()
async_json = json.loads(async_results)
pprint.pprint(async_json)


# ## Access WMS Map Image
# 
# Harmony supports WMS requests, producing geo-registered map images, for all collections associated to a given Harmony transformation service. The following steps were adapted from the [ORNL DAAC help document](https://webmap.ornl.gov/ogc/help/wms_script_python.html) on interacting with a WMS service in Python, using a Sentinel-1 ASF collection.

# In[ ]:


wmsConfig = {
    'asf_collection_id': 'C1225776654-ASF',
    'service': 'WMS',
    'version': '1.3.0',
    'request': 'GetCapabilities'
}

wms_url = harmony_root+'/{asf_collection_id}/wms?service={service}&version={version}&request={request}'.format(**wmsConfig)
print('Request URL', wms_url)


# Information on the WMS service contents and titles of each variable layer:

# In[ ]:


wms = WebMapService(wms_url)
print(wms.identification.title)

[op.name for op in wms.operations]


# In[ ]:


contents = list(wms.contents)
print ('Variable contents:')
contents


# In[ ]:


print('Variable titles:')
[wms[contents[i]].title for i in range(len(contents))]


# Select Coherence layer and send request:

# In[ ]:


coh = contents[1] # Coherence layer

# send the request
img = wms.getmap(
    layers=[coh],
    version='1.3.0',
    CRS='CRS:84',
    styles=['default'],
    bbox=(-180, -90, 180, 90), # Return full extent 
#     bbox=(-121.6,37.2,-120.57,38.0), # Example of a subset over California 
    size=(600, 300),
    format='image/png',
    transparent=True)


# The image file is downloaded to the Harmony outputs directory:

# In[ ]:


# save image in a local file
img_name = '/coh.png'
img_path = str(local_dir+img_name)
out = open(img_path, 'wb')
out.write(img.read())
out.close()


# Read and plot the Coherence layer image:

# In[ ]:


# read png image file 
im = mpimg.imread(img_path) 

# show image 
plt.imshow(im) 
plt.colorbar();

