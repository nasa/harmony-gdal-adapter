import rasterio

file_list = ['IMG-01-ALAV2A279143000-OORIRFU_000.tif', 'IMG-02-ALAV2A279143000-OORIRFU_000.tif', 'IMG-03-ALAV2A279143000-OORIRFU_000.tif','IMG-04-ALAV2A279143000-OORIRFU_000.tif']

# Read metadata of first file
with rasterio.open(file_list[0]) as src0:
    meta = src0.meta

# Update meta to reflect the number of layers
meta.update(count = len(file_list))

# Read each layer and write it to stack
with rasterio.open('stack.tif', 'w', **meta) as dst:
    for id, layer in enumerate(file_list, start=1):
        with rasterio.open(layer) as src1:
            dst.write_band(id, src1.read(1))
