The unittest does test according to the message json file.

requires:
1.if you run the container version unittest in local environment, you need connect to ASF VPN-Full.

2. you need define the EDL_USERNAME and EDL_PASSWORD in the unittest directory
cd ${home directory of unittest}

copy env_unittest .env_unittest file, and edit the file, replace edl_username and edl_password in the file
export EDL_USERNAME=XXXXXXXXXXX
export EDL_PASSWORD=XXXXXXXXXXX


Two ways to run the test:

1. run the unittest insider the docker

from host 

./run-interactive.bash

This brings you in to the docker container.

Insdier the container,

cd /home/unittest

./test_all_message.bash


2. run the script in the host

./run-unittest.bash





