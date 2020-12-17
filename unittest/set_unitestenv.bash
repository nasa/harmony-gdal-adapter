export $(grep -v '^#' .env.unittest | xargs)
export STAGING_PATH=public/asfdataservices/gdal-subsetter
