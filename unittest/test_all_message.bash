#!/bin/bash

#run test_one_message.py multiple times

cd /home/unittest

coverage run -m pytest -v test_all_message.py

#coverage report
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
echo "Report on $DATE_WITH_TIME" >coverage_report_for_module_transform.txt
coverage report -m transform.py >>coverage_report_for_module_transform.txt
