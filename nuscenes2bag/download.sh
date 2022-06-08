#!/bin/bash

declare -a Scenes=(
    "scene-1000" \
    "scene-0429" \
    "scene-0068" \
    "scene-0517"
)

declare -a Models=(
    #"AgnPlanner_agn_dmp_dmp_param_learn" \
    #"AgnPlanner_ctrl_layer_gaussian_action_lstm_goal_conditioned"
    "AgnPlanner_agn_dmp_dmp_param_learn_silu"
)

declare -a ModelNames=(
    "Ours" \ 
    #"RvS"
)

declare -a CKPT=(
    #"epoch=6-step=1861" \
    #"epoch=7-step=2127"
    "epoch=169-step=45219"
)

#experiment_root_dir="/home/gridsan/xiaoli/TRI/ppc/experiments/abalation"
experiment_root_dir="/home/gridsan/xiaoli/TRI/ppc/experiments/testing"

# Iterate the string array using for loop
for i in ${!Models[@]}; do
    echo ${Models[i]}
    model_dir=/home/xli4217/Xiao/datasets/supercloud_data/${ModelNames[i]}
    mkdir $model_dir
    for scene in ${Scenes[@]}; do
        echo $scene
        mkdir $model_dir/$scene
        scp -r xiaoli@txe1-login.mit.edu:$experiment_root_dir/${Models[i]}/${CKPT[i]}/$scene/* $model_dir/$scene
        sudo cp -r $model_dir /home/xli4217/Xiao/datasets/nuscenes/supercloud_data/
    done
done