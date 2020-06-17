#!/bin/bash

rst=$(docker network ls |grep harmony | awk '{print $2}')

if [ "$rst" != "harmony" ]

then

   docker network create harmony

fi
