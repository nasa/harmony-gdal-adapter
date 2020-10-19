import requests
import os
import glob

# Run the queries to subset granules and output product geotiffs
def harmony_requests(harmony_url, path_flag, outfile):

    # Get rid of file from last run
    global path
    if path_flag == 'grfn':
        path = './grfn/grfn_products/'
    elif path_flag == 'uavsar':
        path = './uavsar/uavsar_products/'
    elif path_flag == 'avnir':
        path = './avnir/avnir_products/'

    files = glob.glob(path + '*.tiff')

    for f in files:
        if f == path + outfile:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

# Make the request to harmony
    r = requests.get(harmony_url)
    status_code = r.status_code
    with open(path + outfile, 'wb') as f:
        f.write(r.content)

    return(status_code)
