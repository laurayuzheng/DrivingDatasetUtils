# Arda Mavi
import os, csv
import numpy as np
from os import listdir
import pandas as pd
import cv2
import math

data_frame = []
DATA_TYPE = "test"
TRAINVAL_SPLIT = 0.8
ROOT_DIR = os.path.join("/scratch/lyzheng", "umd-final-steering")
DATA_DIR = os.path.join(ROOT_DIR, "datasets_h5", "honda")
IMG_DIR = os.path.join(DATA_DIR, "IMGS", "valHonda100k")
LABEL_FILEPATH = os.path.join(DATA_DIR, "labelsHonda100k_val.csv")
NUM_CAMERAS = 1
# NPZ_SAVE_DIR = os.path.join(DATA_DIR, DATA_TYPE)

# if not os.path.exists(NPZ_SAVE_DIR):
#     os.makedirs(NPZ_SAVE_DIR)

print("Label file: " + LABEL_FILEPATH)

with open(LABEL_FILEPATH, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        newrow = []
        for item in row: 
            # newitem = item.replace('Desktop\\track1data', os.path.join(ROOT_DIR, 'datasets_h5', 'udacity')).replace('\\', '/')
            newitem = item.replace('\\', '/')
            newrow.append(newitem)
        # print(newrow)
        data_frame.append(newrow)
df = np.array(data_frame)
df = np.transpose(df)

img_len = len(df[0])

# This block splits training set into train and val sets. 
val_threshold = None
if DATA_TYPE == "train":
    val_threshold = math.floor(img_len * TRAINVAL_SPLIT)
# img_len = 100


def fix_name(val):

    if NUM_CAMERAS == 3: 
        val = "/".split(val)
        val = val[-1]

    img_path = os.path.join(IMG_DIR,val)
    print("img path: " + img_path)
    return img_path

savedir = os.path.join(DATA_DIR, DATA_TYPE)

for i in range(img_len):

    if DATA_TYPE == "train" and i > val_threshold:
        savedir = os.path.join(DATA_DIR, "val")
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    x  = list()
    y = list()
    # print(fix_name(df[0][i]))

    
    img_center = cv2.imread(fix_name(df[0][i]))
    img_center = list(cv2.resize(img_center, (210,140)))

    # Udacity will have 3 camera angles; we make use of these here
    if NUM_CAMERAS == 3:
        img_left = cv2.imread(fix_name(df[1][i]))
        img_right = cv2.imread(fix_name(df[2][i]))
        
        img_left = list(cv2.resize(img_left, (210,140)))
        img_right = list(cv2.resize(img_right, (210,140)))

        img_new = [img_left]*2 + [img_center]*4 + [img_right]*2

    # If there is only 1 camera angle, we simply duplicate that angle 8 times.
    # Note that this means there will be no temporal component to learn. 
    else:
    #     print(np.array(img_center).shape)
    #     img_new = [img_center]*8
        img_new = [img_center]

    # img_center is just a regular 3 channel image shape now


    # print(np.array(img_old).shape)
    # print(np.array(img_new).shape)
    # break
    val = float(df[3][i])
    x.append(img_new)
    y.append([val])
    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)
    print(y)

    print(os.path.join(savedir, str(i)+'.npz'))
    # break
    np.savez(os.path.join(savedir, str(i)+'.npz'), x=x, y=y)

