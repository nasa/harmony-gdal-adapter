#!/bin/bash

#This script accept users input: message_file, output_dir

#execute this script in unittest directory

if [ "$#" -ne 2 ]; then
       	echo "you must enter message_file and output_dir"
	exit 1
else
	message_file=$1
	output_dir=$2	
fi

cwd=$PWD

x=$(env | grep EDL)

if [ -z "$x" ]; then
	cd ..
	source bin/localenv.bash
	x=$(env | grep EDL)
	if [ -z "$x" ]; then
		echo "you need define local environmet"
		exit 1
	fi
	cd $cwd
fi
	
#This script does the auto test

export TEST_MESSAGE_FILE=${message_file}

export TEST_OUTPUT_DIR=${output_dir}

#get into har-gdal-env environment

#conda activate har-gdal-env
#pytest -v test_grfn.py
#deactivate
pytest -v test_transform.py

exit 0

