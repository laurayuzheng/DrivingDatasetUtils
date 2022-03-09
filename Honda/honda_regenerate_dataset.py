from __future__ import print_function
 
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt # Import matplotlib functionality
import sys # Enables the passing of arguments
import glob
import os
import csv, os, sys
import random

NVIDIA_STEERING_RATIO = 15.06

def get_steering_id_from_frame_id(frame_idx, frame_timestamps, steer_labels):
    steer_idx = int(frame_idx / 30.0 * 100.0) + 800
    frame_timestamp = frame_timestamps[frame_idx][0]

    #binary search
    s = 0
    t = len(steer_labels)-1
    m = (int)((s+t)/2)
    while t - s > 1:
        if frame_timestamp < steer_labels[m][0]:
            t = m
        else:
            s = m
        m = (int)((s+t)/2)

    if abs(frame_timestamp - steer_labels[s][0]) < abs(frame_timestamp - steer_labels[t][0]):
        steer_idx = s
    else:
        steer_idx = t

    return steer_idx

if __name__ == '__main__':
    TOTAL_FRAME_NEEDED = 110000
    SKIP_FRAME_NUMBER = 4000
    STEP = 5

    video_folder_list = [
    # "C:/data/Honda/release_2019_07_08/2017_04_11_ITS1/201704110943/camera/center",
    #                      "C:/data/Honda/release_2019_07_08/2017_04_13_ITS1/201704130952/camera/center",
    #                      "C:/data/Honda/release_2019_07_08/2017_04_13_ITS1/201704131012/camera/center",
    #                      "C:/data/Honda/release_2019_07_08/2017_04_13_ITS1/201704131020/camera/center",
    #                      "C:/data/Honda/release_2019_07_08/2017_04_14_ITS1/201704140944/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_04_14_ITS1/201704141725/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_06_06_ITS1/201706061309/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_06_06_ITS1/201706061647/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_06_07_ITS1/201706070945/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_06_07_ITS1/201706071658/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_06_08_ITS1/201706081335/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_06_08_ITS1/201706081626/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_06_08_ITS1/201706081707/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_06_13_ITS1/201706130952/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_20_ITS1/201709200946/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_20_ITS1/201709201221/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_20_ITS1/201709201605/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_20_ITS1/201709201700/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_21_ITS1/201709210940/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_21_ITS1/201709211444/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_21_ITS1/201709211547/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_22_ITS1/201709220932/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_22_ITS1/201709221037/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_22_ITS1/201709221238/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_22_ITS1/201709221313/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_09_22_ITS1/201709221435/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_10_03_ITS1/201710031645/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_10_04_ITS1/201710041209/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_10_04_ITS1/201710041448/camera/center",
                         "C:/data/Honda/release_2019_07_08/2017_10_06_ITS1/201710060950/camera/center"]

    video_name_list = glob.glob("release_2019_07_08/*/*/camera/center/*.mp4")

    #video_name_list = []
    #video_name_list.append(video_name_list_0[1])
    #video_name_list.append(video_name_list_0[0])
    #print(len(video_name_list))

    video_name_list = []
    for video_folder in video_folder_list:
        video_name_list_1 = glob.glob(video_folder + "/*.mp4")
        video_name_list = video_name_list + video_name_list_1

    print(video_name_list)

    useful_idx = 0

    DATASET_NAME = "Honda100k_large"

    if not os.path.exists("train"+DATASET_NAME):
        os.mkdir("train"+DATASET_NAME)
    if not os.path.exists("val"+DATASET_NAME):
        os.mkdir("val"+DATASET_NAME)

    statistics_array = np.zeros(73)

    labels_train = open("labels"+DATASET_NAME+"_train.csv", "w")
    labels_val = open("labels"+DATASET_NAME+"_val.csv", "w")

    for video_id in range(len(video_name_list)):
        #if video_id < 5:
        #    continue
        video_name = video_name_list[video_id]

        cap = cv2.VideoCapture(video_name)
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        steer_label_file = os.path.dirname(video_name).replace("camera\\center", "general\\csv") + "/steer.csv"
        steer_label_file = os.path.dirname(video_name).replace("camera/center", "general\\csv") + "/steer.csv"

        if not os.path.exists(steer_label_file):
            sys.exit('Error: the label file is missing： ' + steer_label_file)
        
        with open(steer_label_file, newline='') as f:
            steer_labels = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
        steer_labels = steer_labels[1:len(steer_labels)]


        frame_file_name = os.path.dirname(video_name).replace("release_2019_07_08", "release_2019_07_25") + "/png_timestamp.csv"

        if not os.path.exists(frame_file_name):
            sys.exit('Error: the frame file is missing： ' + frame_file_name)
        
        with open(frame_file_name, newline='') as f:
            frame_timestamps = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
        frame_timestamps = frame_timestamps[1:len(steer_labels)]

        #print(steer_labels[0])
        #print(frame_timestamps[0])

        for i in range(len(steer_labels)):
            steer_labels[i][0] = (float)(steer_labels[i][0])
        for i in range(len(frame_timestamps)):
            frame_timestamps[i][0] = (float)(frame_timestamps[i][0])

        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #steer_idx = 44716
        #print(steer_labels[steer_idx])
        #print(steer_labels[steer_idx][2])
        scene_type = '0'
        frame_idx = -1
        while(cap.isOpened()):
            frame_idx += 1

            ret, frame = cap.read()
            if ret == False:
                break
            if frame_idx < SKIP_FRAME_NUMBER:
                continue
            if frame_idx > total_frame - SKIP_FRAME_NUMBER:
                break

            if frame_idx % STEP != 0:
                continue

            '''
            cv2.imshow('Frame',frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'): # quit
                break
            elif key & 0xFF == ord('p'): # pause
                while (key & 0xFF != ord('r')): # resume
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('0'): # useless data
                        scene_type = '0'
                        print('set scene_type to ', scene_type)
                    elif key & 0xFF == ord('1'): # left only
                        scene_type = '1'
                        print('set scene_type to ', scene_type)
                    elif key & 0xFF == ord('2'): # forward only
                        scene_type = '2'
                        print('set scene_type to ', scene_type)
                    elif key & 0xFF == ord('3'): # right only
                        scene_type = '3'
                        print('set scene_type to ', scene_type)
                    elif key & 0xFF == ord('4'): # left, multiple directions
                        scene_type = '4'
                        print('set scene_type to ', scene_type)
                    elif key & 0xFF == ord('5'): # forward, multiple directions
                        scene_type = '5'
                        print('set scene_type to ', scene_type)
                    elif key & 0xFF == ord('6'): # right, multiple directions
                        scene_type = '6'
                        print('set scene_type to ', scene_type)
                    elif key & 0xFF == ord('7'): # not sure, 1 direction
                        scene_type = '7'
                        print('set scene_type to ', scene_type)
                    elif key & 0xFF == ord('8'): # not sure, multiple directions
                        scene_type = '8'
                        print('set scene_type to ', scene_type)
                    elif key & 0xFF == ord('s'): # not sure, multiple directions
                        print(statistics_array)
            '''

            image_name = str(useful_idx)+".jpg"
            steer_idx = get_steering_id_from_frame_id(frame_idx, frame_timestamps, steer_labels)
            #steer_idx = int(frame_idx / 30.0 * 100.0) + 800
            if (steer_idx >= len(steer_labels)):
                break
            angle = float(steer_labels[steer_idx][2])

            
            output_line = ','.join([image_name,"","",str(angle),scene_type])
            #print(output_line)

            if frame_idx % STEP == 0:
                angle_id = int((angle+360) / 10 + 0.5)
                angle_id = np.clip(angle_id, 0, 72)
                # if angle_id == 36:
                #     if random.randrange(16) != 1:
                #         continue
                # elif angle_id == 35 or angle_id == 37:
                #     if random.randrange(4) != 1:
                #         continue
                # elif angle_id == 34 or angle_id == 38:
                #     if random.randrange(2) != 1:
                #         continue

                # frame = cv2.resize(frame, (455, 256))
                # frame = cv2.resize(frame, (200, 66)) # TODO
                cv2.imwrite("train"+DATASET_NAME+"/"+str(useful_idx)+".jpg", frame)
                useful_idx += 1
                labels_train.write(output_line + '\n')
                statistics_array[angle_id] += 1

            # if frame_idx % (STEP * 10) == (random.randrange(STEP*10-1)+1):
            #     #frame = cv2.resize(frame, (455, 256))
            #     frame = cv2.resize(frame, (200, 66))
            #     cv2.imwrite("val"+DATASET_NAME+"/"+str(useful_idx)+".jpg", frame)
            #     useful_idx += 1
            #     labels_val.write(output_line + '\n')
            

            if frame_idx % STEP == 0:
                #print("video_id: ", video_id)
                #print("frame_idx: ", frame_idx)
                #print("steer_idx: ", steer_idx)
                print("video_id ", video_id, "  frame_idx ", frame_idx, "  useful_idx ", useful_idx, "  scene_type ", scene_type, "  angle ", str(angle), "  angle_id ", str(angle_id-36))
            #print(steer_labels[steer_idx])

            #if frame_idx > 300:
            #    break

            if useful_idx % 1000 == 0:
                print(statistics_array)

            if useful_idx >= TOTAL_FRAME_NEEDED:
                break

        cap.release()
        cv2.destroyAllWindows()
        if useful_idx >= TOTAL_FRAME_NEEDED:
            break

    labels_train.close()
    labels_val.close()
