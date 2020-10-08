The unittest do test according to the message json file.

requires:

cd ${home directory of unittest}

copy env_unittest .env_unittest file, and edit the file, replace edl_username and edl_password in the file
export EDL_USERNAME=XXXXXXXXXXX
export EDL_PASSWORD=XXXXXXXXXXX


Two ways to run the test

1. run the unittest insider the docker

from host 

./run-interactive.bash

This brings you in to the docker container.

Insdier the container,

cd /home/unittest

./test_all_message.bash


2. run the script in the host

./run-unittest.bash





