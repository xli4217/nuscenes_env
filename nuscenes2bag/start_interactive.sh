#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[ 0]}" )" && pwd )"

docker run -it --rm \
           --env="DISPLAY"  \
           --env="QT_X11_NO_MITSHM=1"  \
           --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
           --workdir="/home/$USER" \
           --volume="/home/xli4217/Xiao/postdoc/TRI/nuscenes_env/nuscenes2bag:/home/$USER" \
           --name to_bag nuscenes2bag:0.0.1