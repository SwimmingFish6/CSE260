#!/usr/bin/env bash
git pull origin master
make
mpirun -np 2 ./apf -n 400 -i 200 -x 1 -y 2
