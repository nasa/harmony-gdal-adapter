{
    "accessToken": "tokin",
    "callback": "http://172.17.0.1:3001/service/1330e84b-8959-4291-87ef-315366855359",
    "client": "harmony-local",
    "format": {
        "isTransparent": true,
        "mime": "image/png",
        "width": 40960,
        "height": 20480
    },
    "isSynchronous": true,
    "requestId": "1330e84b-beef-4291-87ef-315366855359",
    "sources": [
        {
            "collection": "C1996881146-POCLOUD",
            "granules": [
                {
                    "bbox": [
                        -180,
                        -90,
                        180,
                        90
                    ],
                    "id": "G2145874703-POCLOUD",
                    "name": "20211017090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1",
                    "temporal": {
                        "end": "2021-10-17T21:00:00.000Z",
                        "start": "2021-10-16T21:00:00.000Z"
                    },
                    "url": "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/MUR-JPL-L4-GLOB-v4.1/20211017090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"
                }
            ],
            "variables": [
                {
                    "fullPath": "analysed_sst",
                    "id": "V2028632042-POCLOUD",
                    "name": "analysed_sst"
                }
            ]
        }
    ],
    "stagingLocation": "s3://local-staging-bucket/public/test/gdal-subsetter/8dfd8964-1513-4b5d-bd2d-08b29f14c4f6/",
    "subset": {},
    "user": "greguska",
    "version": "0.9.0"
}