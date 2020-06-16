#enter into har-gdal-env virtual env
#conda activate har-gdal-env
#create env variables based on key/value pairs in .env file
export $(grep -v '^#' ../harmony/.env | xargs)
echo "check..."

#set ENV=dev for development
export ENV=dev
#unset the env variables based on key/value pairs in .env file
#unset $(grep -v '^#' ../harmony/.env | xargs -d "=" | cut -d " " -f 1)
