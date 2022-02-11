#!/bin/bash


PKG_NAME=CnnLstmAgn_dmp_trainable
mkdir ./data/supercloud_data/$PKG_NAME
scp -r xiaoli@txe1-login.mit.edu:/home/gridsan/xiaoli/TRI/ppc/experiments/next/$PKG_NAME/epoch=499-step=1999/scene-0061 ./data/supercloud_data/$PKG_NAME
scp -r xiaoli@txe1-login.mit.edu:/home/gridsan/xiaoli/TRI/ppc/experiments/next/$PKG_NAME/epoch=499-step=1999/scene-0103 ./data/supercloud_data/$PKG_NAME
scp -r xiaoli@txe1-login.mit.edu:/home/gridsan/xiaoli/TRI/ppc/experiments/next/$PKG_NAME/eval_data ./data/supercloud_data/$PKG_NAME
