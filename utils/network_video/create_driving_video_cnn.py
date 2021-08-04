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

# python3 create_driving_video_lstm.py --trace lstm_test1_var0_2019-07-31-10-57-00
trace = args.trace

title = ""
if("var01" in trace):
    title += "under $\\sigma^2$=0.1 pertubation"
elif("var02" in trace):
    title += "under $\\sigma^2$=0.2 pertubation"
elif("var03" in trace):
    title += "under $\\sigma^2$=0.3 pertubation"

frame = TexAnimation("driving_cnn/template.tex",output_dir=os.path.join("sequences","seq_{}".format(trace)))
frame.update_newcommands("titletext","{}".format(title))

automode_path = os.path.join("/home/mathias/dev/ros_analysis/automode_export/",trace+".csv")
base_path = os.path.join("/home/mathias/dev/autonomous_driving/exported_replays/",trace)

# 1st column: Id, then values
activation_output = np.loadtxt(os.path.join(base_path,"output.csv"),delimiter=";",skiprows=0)
activation_neuron = np.loadtxt(os.path.join(base_path,"sensory.csv"),delimiter=";",skiprows=0)

automode = np.loadtxt(automode_path)[:,1]


neuron_center = np.mean(activation_neuron,axis=0)
# neuron_center = np.zeros(activation_neuron.shape[1])
neuron_high = neuron_center + np.std(activation_neuron,axis=0)
neuron_low = neuron_center - np.std(activation_neuron,axis=0)


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
    frame.update_newcommands("layerb","\\includegraphics[width=1.15.cm]{"+base_path+"/saliency_aux/frame_{:05d}_layer_1.png}}".format(t))
    frame.update_newcommands("layerc","\\includegraphics[width=1.0cm]{"+base_path+"/saliency_aux/frame_{:05d}_layer_2.png}}".format(t))
    frame.update_newcommands("layerd","\\includegraphics[width=0.9cm]{"+base_path+"/saliency_aux/frame_{:05d}_layer_3.png}}".format(t))
    # for i,c in enumerate(['a','b','c','d','e','f','g','h']):
    #     frame.update_newcommands("featimg{:}".format(c),"\\includegraphics[width=0.9cm]{"+base_path+"/saliency_aux/frame_{:05d}_feat_{:d}.png}}".format(t,i))

    frame.update_color("outputcol",v_to_color(activation_output[t,1],0,-20,20))

    if(automode[t]>0.5):
        frame.clear_tikz_style("automodecolor")
        frame.update_tikz_style("automodecolor","autogreen",None)
        frame.update_newcommands("automode","Autonomous")
    else:
        frame.clear_tikz_style("automodecolor")
        frame.update_tikz_style("automodecolor","autored",None)
        frame.update_newcommands("automode","Manual")

    # Leading column is time -> skip on index
    for i in range(activation_neuron.shape[1]-1):
        v = activation_neuron[t,i+1]
        center = neuron_center[i+1]
        low = neuron_low[i+1]
        high = neuron_high[i+1]
        frame.update_color("neur{:d}".format(i),v_to_color(v,center,low,high))

    # for i in range(activation_sensory.shape[1]-1):
    #     v = activation_sensory[t,i+1]
    #     center = sensory_center[i+1]
    #     low = sensory_low[i+1]
    #     high = sensory_high[i+1]
    #     frame.update_color("feat{:d}".format(i),v_to_color(v,center,low,high))
    
    arc = np.clip(activation_output[t,1]/40,-1,1)
    frame.update_tikz_style("outputrot","rotate","{:0.2f}".format(arc*80))


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

# os.makedirs("rendered",exist_ok=True)
# frame.to_mp4(os.path.join("rendered","{}.mp4".format(trace)))