#!/bin/bash

scp -r xiaoli@txe1-login.mit.edu:/home/gridsan/xiaoli/TRI/ppc/experiments/next/CnnLstmAgn/epoch=17-step=71/scene-0061 ./data/supercloud_data/
mkdir ./data/supercloud_data/CnnLstmAgn_eval_data
scp -r xiaoli@txe1-login.mit.edu:/home/gridsan/xiaoli/TRI/ppc/experiments/next/CnnLstmAgn/epoch=17-step=71/eval_data ./data/supercloud_data/CnnLstmAgn_eval_data/