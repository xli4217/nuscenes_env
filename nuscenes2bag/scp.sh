#!/bin/bash

PKG_NAME=CnnLstmAgn
CKPT_NAME=epoch=499-step=1999
SCENE_NAME=scene-0061

mkdir -p ./data/supercloud_data/$PKG_NAME/$CKPT_NAME/$SCENE_NAME
scp -r xiaoli@txe1-login.mit.edu:/home/gridsan/xiaoli/TRI/ppc/experiments/next/$PKG_NAME/$CKPT_NAME/$SCENE_NAME ./data/supercloud_data/$PKG_NAME/$CKPT_NAME/$SCENE_NAME

