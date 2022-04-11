from __future__ import print_function
 
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt # Import matplotlib functionality
import sys # Enables the passing of arguments
import glob
import os
import csv, os, sys
import random
import json
import open3d as o3
import matplotlib.pylab as pt

def get_steering_id_from_timestamp(frame_timestamp, steer_timestamps):

    #binary search
    s = 0
    t = len(steer_timestamps)-1
    m = (int)((s+t)/2)
    while t - s > 1:
        if frame_timestamp < steer_timestamps[m]:
            t = m
        else:
            s = m
        m = (int)((s+t)/2)

    if abs(frame_timestamp - steer_timestamps[s]) < abs(frame_timestamp - steer_timestamps[t]):
        steer_idx = s
    else:
        steer_idx = t

    # print('==========================')
    # print(frame_timestamp)
    # print(steer_timestamps[steer_idx-1])
    # print(steer_timestamps[steer_idx])
    # print(steer_timestamps[steer_idx+1])
    return steer_idx

def process_img(data_root, save_dir, trainval_split=0.1, test_split=0.1):

    folders = ["20190401_121727"]
    seq_count = 0
    num_frames = 32 # create sequence of 8

    for folder in folders:
        bus_signal_file = os.path.join(data_root, "camera_lidar", folder, "bus", folder.replace("_","") + "_bus_signals" + ".json")
        with open(bus_signal_file) as f:
            bus_signal = json.load(f)
        steer_labels = bus_signal['steering_angle_calculated']['values']
        sign = bus_signal['steering_angle_calculated_sign']['values']

        # DATASET_NAME = "AudiT"

        # if not os.path.exists("train"+DATASET_NAME):
        #     os.mkdir("train"+DATASET_NAME)
        # if not os.path.exists("val"+DATASET_NAME):
        #     os.mkdir("val"+DATASET_NAME)

        # labels = open("labels"+DATASET_NAME+".csv", "w")

        image_folder = os.path.join(data_root, "camera_lidar/" + folder + "/camera/cam_front_center/")
        json_list = sorted(glob.glob( image_folder + "/*.json"))
        print(json_list)
        test_split_idx = int(len(json_list)/num_frames * (1-test_split)) - 1 # index to start saving to test
        val_split_idx = int(test_split_idx * (1-trainval_split)) # index to start saving to val
        print("Total sequences in folder: ", int(len(json_list) / num_frames))
        print(test_split_idx, val_split_idx)

        # STEP = 5
        frame_count = 0 # keeps track of frames saved
        
        x = [] 
        y = [] 
        seq_count_folder = 0

        for frame_idx in range(len(json_list)):
            # if frame_idx % STEP != 0:
            #     continue
            
            if frame_count % num_frames == 0: 
                x = [] 
                y = [] 

            # print(frame_idx)

            with open(json_list[frame_idx]) as f:
                image_info = json.load(f)

            image_timestamp = image_info['cam_tstamp']

            steer_labels = np.array(steer_labels)

            steer_idx = get_steering_id_from_timestamp(image_timestamp, steer_labels[:,0])
            angle = float(steer_labels[steer_idx][1])
            if sign[steer_idx][1] == 0:
                angle = -angle

            image_name = image_info['image_png']

            # output_line = ','.join([image_name,"","",str(angle)])
            # labels.write(output_line + '\n')

            frame = cv2.imread(image_folder + image_name)
            frame = cv2.resize(frame, (224, 224))
            # cv2.imwrite("train"+DATASET_NAME+"/"+image_name, frame)
            x.append(frame)
            y.append([angle])

            if len(y) % num_frames == 0: 
                x = np.array(x)
                y = np.array(y) 
                print(x.shape, y.shape)
                subfolder = "train"
                if seq_count_folder > test_split_idx:
                    subfolder = "test"
                elif seq_count_folder > val_split_idx:
                    subfolder = "val"

                print(os.path.join(save_dir, subfolder, str(seq_count)+'.npz'))
                np.savez(os.path.join(save_dir, subfolder, str(seq_count)+'.npz'), x=x, y=y)
                seq_count += 1
                seq_count_folder += 1
            

            # frame = cv2.putText(frame, str(angle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
            # break

            frame_count += 1

    # labels.close()

if __name__ == '__main__':
    # process_img_seg()
    data_root = "/scratch/vroom/data/audi_raw/audi_munich"
    save_dir = "/scratch/vroom/data/audi_32"
    process_img(data_root, save_dir)
    # generate_depthmap()