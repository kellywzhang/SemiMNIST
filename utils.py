import torch
import random
import random
import scipy.ndimage
import numpy as np
from torch.autograd import Variable

def to_gpu(var, cuda):
    if cuda:
        return var.cuda()
    return var

def parse_layers_from_file(param_file):
    f = open(param_file, 'r')

    layers = []
    for line in f:
        if len(line) > 1 and line[0] != "#":
            raw_layer = line[:-1].lower().split(",")
            raw_layer = [x.strip() for x in raw_layer]
            raw_layer[1] = int(raw_layer[1])
            raw_layer[2] = int(raw_layer[2])
            raw_layer[3] = True if raw_layer[3] == "true" else False
            raw_layer[4] = "" if raw_layer[4] == "none" else raw_layer[4]

            layers.append(raw_layer)

    print("Parsed Layers:")
    print(layers)
    return layers

# Crops the center of the image
def crop_center(img,cropx,cropy):
    if img.ndim == 2:
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[starty:starty+cropy,startx:startx+cropx]
    elif img.ndim == 3:
        b,y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[:,starty:starty+cropy,startx:startx+cropx]

# Take a random crop of the image
def crop_random(img,cropx,cropy):
    # takes numpy input
    if img.ndim == 2:
        x1 = random.randint(0, img.shape[0] - cropx)
        y1 = random.randint(0, img.shape[1] - cropy)
        return img[x1:x1+cropx,y1:y1+cropy]

# Image data augmentation
def augment(image):
    npImg = image.numpy().squeeze(1)
    # rotate image by maximum of 25 degrees clock- or counter-clockwise
    rotation = [random.randrange(-25,25) for i in range(len(image))]
    rotatedImg = [scipy.ndimage.interpolation.rotate(im, rotation[i], axes=(0,1)) for i, im in enumerate(npImg)]
    # crop image to 28x28 as rotation increases size
    rotatedImgCentered = [crop_center(im, 28, 28) for im in rotatedImg]
    # pad image by 3 pixels on each edge (-0.42421296 background color)
    paddedImg = [np.pad(im, 3, 'constant',constant_values=-0.42421296) for im in rotatedImgCentered]
    # randomly crop from padded image
    cropped = np.array([crop_random(im, 28, 28) for im in paddedImg])
    return torch.FloatTensor(cropped).unsqueeze(1)
