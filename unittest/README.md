The unittest does test according to the message json file.

requires:
1.if you run the container version unittest in a local environment, you need connect to ASF VPN-Full.

2. you need define environment variables in the unittest directory

the environment variables are those in the harmony .env file. We provided the env.unittest.example as a template in the ${home directory of unittest}.

cd ${home directory of unittest}

cp env.unittest.example .env.unittest

edit .env.unittest with your EDL_username and EDL_password

source set_unittestenv.bash

3. build the docker iamge asfdataservices/gdal-subsetter on the host

cd gdal-subsetter

bin/build_iamge

4. Two ways to run the test:

a. run the unittest insider the docker

from host 

./run-interactive.bash

This brings you in to the docker container.

Insdier the container,

cd /home/unittest

./test_whole_message.bash

b. run the script in the host

./run-unittest.bash

 


