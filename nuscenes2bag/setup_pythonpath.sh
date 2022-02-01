#!/bin/bash

COMPUTE_LOCATION=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[ 0]}" )" && pwd )"

export PKG_PATH=$DIR
export PYTHONPATH=$PKG_PATH:$PYTHONPATH
# for ray in satori
export LD_LIBRARY_PATH=/nobackup/users/xiaoli/anaconda3/envs/nuscenes/lib:$LD_LIBRARY_PATH 

export COMPUTE_LOCATION=$COMPUTE_LOCATION
echo "Current compute location:"
echo $COMPUTE_LOCATION
export PYTHONIOENCODING=utf8 # for printing temporal operators
