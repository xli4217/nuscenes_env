#!/bin/bash

COMPUTE_LOCATION=$1

echo "Current compute location: $COMPUTE_LOCATION"

if [[ $COMPUTE_LOCATION == "mac" ]]; then
    v="/Users/xiaoli/Xiao/TRI/nuscenes_env/nuscenes2bag"
fi

if [[ $COMPUTE_LOCATION == "local" ]]; then
    v="/home/xli4217/Xiao/postdoc/TRI/nuscenes_env/nuscenes2bag"
    v_data="/home/xli4217/Xiao/datasets/nuscenes/"
fi

if [[ $COMPUTE_LOCATION == "supercloud" ]]; then
    v="/home/xli4217/Xiao/postdoc/TRI/nuscenes_env/nuscenes2bag"
fi


DIR="$( cd "$( dirname "${BASH_SOURCE[ 0]}" )" && pwd )"

docker run -it --rm \
           --env="DISPLAY"  \
           --env="QT_X11_NO_MITSHM=1"  \
           --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
           --workdir="/home/$USER" \
           --volume="$v:/home/$USER" \
           --volume="$v_data:/home/$USER/data" \
           --name to_bag nuscenes2bag:0.0.1