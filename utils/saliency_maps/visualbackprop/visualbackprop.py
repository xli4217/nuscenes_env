import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch

FEAT_KEEP = 5  # Feature Maps to show
FEAT_SIZE = 250  # Size of feature maps to show
FEAT_MAPS_DIR = 'feat_maps'  # dir. to save feat maps
VBP_DIR = 'VBP_results'  # dir. to save VBP results
OVERLAY_DIR = "overlay"  # dir. to save overlay results

layers = []
maps = []
hooks = []

def add_hook(model, layer_name, func, hooks, layers, maps):
    '''
    Add a hook function in the layers you specified.
    Hook will be called during forward propagate at the layer you specified.

    :param net: The model you defined
    :param layer_name: Specify which layer you want to hook, currently you can hook 'all', 'maxpool', 'relu'
    :param func: Specify which hook function you want to hook while forward propagate
    :return: this function will return the model that hooked the function you specified in specific layer
    '''
    if layer_name == 'all':
        i = 0
        for m in model.modules():
            type_name = str(type(m)).replace("<'", '').replace("'>", '').split('.')[-1]
            name = 'features' + '-' + str(i) + '-' + type_name
            hook = m.register_forward_hook(func)
            hooks.append(hook)
            i += 1
        return model
    
    if layer_name == 'relu':
        i = 0
        for m in model.modules():
            if isinstance(m, torch.nn.ReLU):
                type_name = str(type(m)).replace("<'", '').replace("'>", '').split('.')[-1]
                name = 'features' + '-' + str(i) + '-' + type_name
                hook = m.register_forward_hook(func)
                layers.append((name, m))
                hooks.append(hook)
                i += 1
        return model
    
def save_feature_maps(self, inputs, outputs):
    # The hook function that saves feature maps while forward propagate
    # print((self.__class__.__name__, type(outputs.data)))
    # layers.append(self.__class__.__name__)
    maps.append(outputs.data)
    
def normalize_gamma(image, gamma=1.0):
    # normalize data for display
    image = (image - image.min()) / (image.max() - image.min())
    invGamma = 1.0 / gamma
    image = (image ** invGamma)  * 255
    return cv2.UMat(image)

def visual_feature(self, input, output):
    # The hook function that show you the feature maps while forward propagate
    vis_square(output.data[0,:])

    
def visualbackprop(layers,maps):

    '''
    :param layers: the saved layers
    :param maps: the saved maps
    :return: return the final mask
    '''

    num_layers = len(layers)
    avgs = []
    mask = None
    ups  = []

    upSample = nn.Upsample(scale_factor=2)

    num_layers = len(maps)
    for n in range(num_layers-1,0,-1):
        cur_layer=layers[n]
        if cur_layer in ["MaxPool2d", "ReLU"]:
            ##########################
            # Get and set attributes #
            ##########################
            cur_map = maps[n-1]  # input at this layer

            ###########################################
            # Average filters and multiply pixel-wise #
            ###########################################

            # Average filters
            avg = cur_map.mean(dim=1)
            avg = avg.unsqueeze(0)
#             print('avg', avg.shape)
            avgs.append(avg)

            if mask is not None:
#                 print('mask', mask.shape)
                if mask.shape != avg.shape:
                    mask = upSample(mask).data
                if mask.shape != avg.shape:
                    mask = mask[:,:, :mask.shape[2]-1, :mask.shape[3]-1]
#                 print('mask', mask.shape)
                mask = mask * avg
            else:
                mask = avg

            # upsampling : see http://pytorch.org/docs/nn.html#convtranspose2d
            weight = torch.ones(1, 1, 3, 3)
            up = torch.nn.functional.conv_transpose2d(mask, weight, stride=1, padding=1)
            mask = up.data
            ups.append(mask)

    return ups

def plotFeatMaps(layers,maps):

    '''
    :param layers: the saved layers
    :param maps: the saved maps
    :return: top feat. maps of relu layers
    '''
    print("~~~~~~plotFeatMaps~~~~~~~~~~~")
    num_layers = len(layers)
    feat_collection = []
    # Show top FEAT_KEEP feature maps (after ReLU) starting from bottom layers
    for n in range(num_layers):
        cur_layer=layers[n][1]
        if type(cur_layer) in [torch.nn.MaxPool2d, torch.nn.ReLU]:
            print(type(cur_layer))
            ##########################
            # Get and set attributes #
            ##########################
            relu = maps[n-1]
            print("map:", relu.shape)

            ###########################################
            # Sort Feat Maps based on energy of F.M. #
            ###########################################
            feat_energy = []
            # Get energy of each channel
            for channel_n in range(relu.shape[1]):
                feat_energy.append(np.sum(relu[0][channel_n].numpy()))
            feat_energy = np.array(feat_energy)
            # Sort energy
            feat_rank = np.argsort(feat_energy)[::-1]

            # Empty background
            back_len = int(math.ceil(math.sqrt(FEAT_SIZE * FEAT_SIZE * FEAT_KEEP * 2)))
            feat = np.zeros((back_len, back_len))
            col = 0
            row = 0
            for feat_n in range(FEAT_KEEP):
                if col*FEAT_SIZE + FEAT_SIZE < back_len:
                    feat[row*FEAT_SIZE:row*FEAT_SIZE + FEAT_SIZE, col*FEAT_SIZE:col*FEAT_SIZE + FEAT_SIZE] =\
                        cv2.resize(normalize_gamma(relu[0][feat_rank[feat_n]].numpy(), 0.1).get(), (FEAT_SIZE,FEAT_SIZE))
                    col = col + 1
                else:
                    row = row + 1
                    col = 0
                    feat[row*FEAT_SIZE:row*FEAT_SIZE + FEAT_SIZE, col*FEAT_SIZE:col*FEAT_SIZE + FEAT_SIZE] =\
                        cv2.resize(normalize_gamma(relu[0][feat_rank[feat_n]].numpy(), 0.1).get(), (FEAT_SIZE,FEAT_SIZE))
                    col = col + 1

            feat_collection.append(feat)

    return feat_collection

# Show VBP Result
def show_VBP(plot, image):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    image = image.numpy()

    # normalize data for display
    data = (image - image.min()) / (image.max() - image.min())
    data = data[0,0,:,:]
    data = cv2.resize(data, (224,224))
    data = (data*255).astype("uint8")
    # cv2.imwrite(label, data)
#     cv2.imshow(label, data)
    plot.imshow(data)


# Save VBP Result
def save_VBP(label, image):
    image = image.numpy()

    # normalize data for display
    data = (image - image.min()) / (image.max() - image.min())
    data = data[0,0,:,:]
    data = cv2.resize(data, (224,224))
    data = (data*255).astype("uint8")
    # cv2.imwrite(label, data)
    cv2.imwrite(label, data)


def overlay(image, mask):
    # normalize data for display
    print(mask.shape)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    print(mask.shape)
    mask = mask[0,0,:,:,:]
    print(mask.shape)
#     mask = cv2.resize(mask, (224,224))
    mask = (mask*255).astype("uint8")
#     assert image.shape == mask.shape, "image %r and mask %r must be of same shape" % (image.shape, mask.shape)
    # if image[:,:,2] + mask > 255:
        # image[:,:,2] = image[:,:,2] + mask
    # else:
    image[:,:,2] = cv2.add(image[:,:,2], mask)

    return image
