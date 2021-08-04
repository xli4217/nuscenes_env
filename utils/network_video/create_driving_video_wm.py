from texinmotion import TexAnimation
import numpy as np
import time
import os
import argparse
from parula_color import get_rgb_value

def distort_w(w):
    return w

def activity_to_opacity(v_pre,mu,sigma):
    # return 1.0
    x = sigma*(v_pre - mu)
    return 1 / (1 + np.exp(-x))


def v_to_color(v,center,low,high):
    v = np.clip(v,low,high)
    if(v < center):
        p = (v-low)/(center-low+0.001)
        c_value = 0.5*p
    else:
        p = (v-center)/(high-center+0.001)
        c_value = 0.5*p+0.5
    return get_rgb_value(c_value)


# Parse arugments
parser = argparse.ArgumentParser()
parser.add_argument('--trace',default='None')
args = parser.parse_args()

# python3 create_driving_video_wm.py --trace wm_test1_var0_2019-07-31-10-35-56
# python3 create_driving_video_wm.py --trace wm_test1_var01_2019-07-31-11-21-42
# trace = "wm_test1_var0_2019-07-31-10-35-56"
# # trace = "wm_test1_var01_2019-07-31-11-21-42"
trace = args.trace

title = ""
if("var01" in trace):
    title += "under $\\sigma^2$=0.1 pertubation"
elif("var02" in trace):
    title += "under $\\sigma^2$=0.2 pertubation"
elif("var03" in trace):
    title += "under $\\sigma^2$=0.3 pertubation"

frame = TexAnimation("driving/template.tex",output_dir=os.path.join("sequences","seq_{}".format(trace)))
frame.update_newcommands("titletext","{}".format(title))

automode_path = os.path.join("/home/mathias/dev/ros_analysis/automode_export/",trace+".csv")
base_path = os.path.join("/home/mathias/dev/autonomous_driving/exported_replays/",trace)
# gleak;vleak;cm
neuron_params = np.loadtxt("/home/mathias/dev/autonomous_driving/final_models/wm_ep_55/wm_dump/neuron_parameters.csv",skiprows=1,delimiter=";")

# src;dest;sigma;mu,w;erev;erev_init
inter_synapse_param = np.loadtxt("/home/mathias/dev/autonomous_driving/final_models/wm_ep_55/wm_dump/inter_synapses.csv",skiprows=1,delimiter=";")
sensory_synapse_param = np.loadtxt("/home/mathias/dev/autonomous_driving/final_models/wm_ep_55/wm_dump/sensory_synapses.csv",skiprows=1,delimiter=";")

# 1st column: Id, then values
activation_neuron = np.loadtxt(os.path.join(base_path,"neurons.csv"),delimiter=";",skiprows=0)
activation_output = np.loadtxt(os.path.join(base_path,"output.csv"),delimiter=";",skiprows=0)
activation_sensory = np.loadtxt(os.path.join(base_path,"sensory.csv"),delimiter=";",skiprows=0)

automode = np.loadtxt(automode_path)[:,1]



neuron_center = np.mean(activation_neuron,axis=0)
neuron_high = neuron_center + np.std(activation_neuron,axis=0)
neuron_low = neuron_center - np.std(activation_neuron,axis=0)
sensory_center = np.mean(activation_sensory,axis=0)
sensory_high = sensory_center + np.std(activation_sensory,axis=0)
sensory_low = sensory_center - np.std(activation_sensory,axis=0)


# Set polarity and width of synapses
for i in range(inter_synapse_param.shape[0]):
    attr_name = "exsyn"
    if(inter_synapse_param[i,5]<0.0):
        attr_name = "inhsyn"
    frame.update_tikz_style("syn{:d}x{:d}".format(int(inter_synapse_param[i,0]),int(inter_synapse_param[i,1])),attr_name,None)

    # frame.update_tikz_style("syn{:d}x{:d}".format(int(inter_synapse_param[i,0]),int(inter_synapse_param[i,1])),"line width","{:0.2f}".format(distort_w(inter_synapse_param[i,4])))

for i in range(sensory_synapse_param.shape[0]):
    attr_name = "exsynsensory"
    if(sensory_synapse_param[i,5]<0.0):
        attr_name = "inhsynsensory"
    frame.update_tikz_style("sensyn{:d}x{:d}".format(int(sensory_synapse_param[i,0]),int(sensory_synapse_param[i,1])),attr_name,None)

    # frame.update_tikz_style("sensyn{:d}x{:d}".format(int(sensory_synapse_param[i,0]),int(sensory_synapse_param[i,1])),"line width","{:0.2f}".format(distort_w(sensory_synapse_param[i,4])))


# Loop over each frame
# for t in range(activation_neuron.shape[0]):
N = activation_neuron.shape[0]
start_time = time.time()
skip_at_start = 0 #int(np.argmax(automode))
for t in range(N):
    print("Generating frame {:05d}/{:05d} ... ".format(t,N),end="")
    if((frame.already_exist()) or (t < skip_at_start)):
        frame.skip()
        print(" [skip]")
        continue

    # Create file with for dynamic variables
    frame.update_newcommands("framecamera","\\includegraphics[width=4cm]{"+base_path+"/frames/frame_{:05d}.jpg}}".format(t))
    frame.update_newcommands("framesaliency","\\includegraphics[width=4cm]{"+base_path+"/saliency_map/frame_{:05d}.png}}".format(t))

    frame.update_newcommands("gpsmap","\\includegraphics[width=1.0cm]{"+base_path+"/gps_map/map_{:05d}.png}}".format(t))

    frame.update_newcommands("layera","\\includegraphics[width=1.3cm]{"+base_path+"/saliency_aux/frame_{:05d}_layer_0.png}}".format(t))
    frame.update_newcommands("layerb","\\includegraphics[width=1.1.cm]{"+base_path+"/saliency_aux/frame_{:05d}_layer_1.png}}".format(t))
    frame.update_newcommands("layerc","\\includegraphics[width=0.9cm]{"+base_path+"/saliency_aux/frame_{:05d}_layer_2.png}}".format(t))
    for i,c in enumerate(['a','b','c','d','e','f','g','h']):
        frame.update_newcommands("featimg{:}".format(c),"\\includegraphics[width=0.9cm]{"+base_path+"/saliency_aux/frame_{:05d}_feat_{:d}.png}}".format(t,i))

    frame.update_color("outputcol",v_to_color(activation_output[t,1],0,-20,20))

    if(automode[t]>0.5):
        frame.clear_tikz_style("automodecolor")
        frame.update_tikz_style("automodecolor","autogreen",None)
        frame.update_newcommands("automode","Autonomous")
    else:
        frame.clear_tikz_style("automodecolor")
        frame.update_tikz_style("automodecolor","autored",None)
        frame.update_newcommands("automode","Manual")
    # Set opacity of synapses based on sigmoid activation
    # for i in range(inter_synapse_param.shape[0]):
    #     v_pre = activation_neuron[t,int(inter_synapse_param[i,0])+1]
    #     mu = inter_synapse_param[i,3]
    #     sigma = inter_synapse_param[i,2]
    #     frame.update_tikz_style("syn{:d}x{:d}".format(int(inter_synapse_param[i,0]),int(inter_synapse_param[i,1])),"opacity","{:0.2f}".format(activity_to_opacity(v_pre,mu,sigma)))
    # for i in range(sensory_synapse_param.shape[0]):
    #     v_pre = activation_sensory[t,int(sensory_synapse_param[i,0])+1]
    #     mu = sensory_synapse_param[i,3]
    #     sigma = sensory_synapse_param[i,2]
    #     frame.update_tikz_style("sensyn{:d}x{:d}".format(int(sensory_synapse_param[i,0]),int(sensory_synapse_param[i,1])),"opacity","{:0.2f}".format(activity_to_opacity(v_pre,mu,sigma)))

    # Leading column is time -> skip on index
    for i in range(1,neuron_params.shape[0]):
        v = activation_neuron[t,i+1]
        center = neuron_center[i+1]
        low = neuron_low[i+1]
        high = neuron_high[i+1]
        frame.update_color("neur{:d}".format(i),v_to_color(v,center,low,high))
    frame.update_color("neur{:d}".format(0),v_to_color(activation_neuron[t,i+1],0,-3,3))

    for i in range(activation_sensory.shape[1]-1):
        v = activation_sensory[t,i+1]
        center = sensory_center[i+1]
        low = sensory_low[i+1]
        high = sensory_high[i+1]
        frame.update_color("feat{:d}".format(i),v_to_color(v,center,low,high))
    
    arc = np.clip(activation_output[t,1]/40,-1,1)
    frame.update_tikz_style("outputrot","rotate","{:0.2f}".format(arc*80))

    # if(np.abs(arc)<0.01):
    #     frame.update_newcommands("steerarc","\\draw[thick,red] (3.5,-1.5) -- +(0,1);")
    # elif(arc<0):
    #     deg = -arc*
    #     frame.update_newcommands("steerarc","\\draw[thick,red] (3.5,-1.5) -- +(0,1);")


    frame.render()
    amount_done = 1+t
    amount_due = N-amount_done
    time_done = time.time()-start_time
    time_per_frame = time_done/amount_done
    time_due = time_per_frame*amount_due

    mins = time_due//60
    hours = int(mins//60)
    mins = int(mins%60)
    seconds = int(time_due % 60)
    print(" [done] ({:0.1f}s/frame, ETA: {:02d}:{:02d}:{:02d} h:m:s)".format(
        time_per_frame, hours,mins,seconds,
    ))

os.makedirs("rendered",exist_ok=True)
frame.to_mp4(os.path.join("rendered","{}.mp4".format(trace)))