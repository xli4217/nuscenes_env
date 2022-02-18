#!/bin/bash

# PKG_NAME=CnnLstmAgn_dmp
# CKPT_NAME=epoch=0-step=62
# SCENE_NAME=scene-0061-pure-dmp

# mkdir -p ./data/supercloud_data/$PKG_NAME/$CKPT_NAME/$SCENE_NAME
# #scp -r xiaoli@txe1-login.mit.edu:/home/gridsan/xiaoli/TRI/ppc/experiments/next/$PKG_NAME/$CKPT_NAME/$SCENE_NAME ./data/supercloud_data/$PKG_NAME/$CKPT_NAME/$SCENE_NAME
# scp -r xiaoli@satori-login-001.mit.edu:/nobackup/users/xiaoli/TRI/ppc/experiments/testing/$PKG_NAME/$CKPT_NAME/$SCENE_NAME ./data/supercloud_data/$PKG_NAME/$CKPT_NAME/


mkdir -p ./data/supercloud_data/risk_logic_net

scp -r xiaoli@txe1-login.mit.edu:/home/gridsan/xiaoli/TRI/stl_risk_aware_ioc/experiments/test/test/final ./data/supercloud_data/risk_logic_net/