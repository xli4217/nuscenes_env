#!/bin/bash

COMPUTE_LOCATION=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[ 0]}" )" && pwd )"

export PKG_PATH=$DIR
#export PYTHONPATH=$DIR:$PKG_PATH/external_libs/:$DIR/external_libs/highway_env/:$PYTHONPATH
export PYTHONPATH=$PKG_PATH:$PKG_PATH/external_libs/:$PKG_PATH/external_libs/nuscenes_env:$PKG_PATH/external_libs/facets/facets_overview/:$PYTHONPATH
# for ray in satori
export LD_LIBRARY_PATH=/nobackup/users/xiaoli/anaconda3/envs/nuscenes/lib:$LD_LIBRARY_PATH 

#CPU_NAME=$(cat /proc/cpuinfo | grep 'model name' | uniq)
#echo $CPU_NAME

export COMPUTE_LOCATION=$COMPUTE_LOCATION
echo "Current compute location:"
echo $COMPUTE_LOCATION
export PYTHONIOENCODING=utf8 # for printing temporal operators
