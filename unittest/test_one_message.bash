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

cd /home/unittest

#message_file=data/messages/message1

#output_dir=data/results/grfn-results

cwd=$PWD

echo $cwd

x=$(env | grep EDL)

if [ -z "$x" ]; then

	source .env_unittest

	x=$(env | grep EDL)
	if [ -z "$x" ]; then
		echo "you need define local environmet"
		exit 1
	fi

fi

#This script does the auto test

export TEST_MESSAGE_FILE=${message_file}

export TEST_OUTPUT_DIR=${output_dir}

echo `env|grep TEST`

#pytest -v test_transform.py

coverage run -m pytest -v test_transform.py

exit 0

