#!/bin/bash

rst=$(docker network ls |grep harmony | awk '{print $2}')

if [ "$rst" != "harmony" ]

then

   docker network create harmony --subnet 172.24.24.0/24

fi

#docker run --network harmony makes docker get the host/run/systemd/resolve/resolv.conf, while host/etc/resolv.conf -> host/run//run/systemd/resolve/stub-resolv.conf, we need make docker /etc/resolv.conf be /host/etc/resolv.conf

cd /run/systemd/resolve

sudo mv resolv.conf resolv.conf.orig

sudo cp stub-resolv.conf resolv.conf


