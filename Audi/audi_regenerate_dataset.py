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

def process_img():

    bus_signal_file = "camera_lidar/20180810_150607/bus/20180810150607_bus_signals.json"
    with open(bus_signal_file) as f:
        bus_signal = json.load(f)
    steer_labels = bus_signal['steering_angle_calculated']['values']
    sign = bus_signal['steering_angle_calculated_sign']['values']

    DATASET_NAME = "AdL"

    if not os.path.exists("train"+DATASET_NAME):
        os.mkdir("train"+DATASET_NAME)
    if not os.path.exists("val"+DATASET_NAME):
        os.mkdir("val"+DATASET_NAME)

    labels = open("labels"+DATASET_NAME+"_ori.csv", "w")

    image_folder = "camera_lidar/20180810_150607/camera/cam_front_center/"
    json_list = glob.glob( image_folder + "/*.json")

    STEP = 5

    for frame_idx in range(len(json_list)):
        if frame_idx % STEP != 0:
            continue

        print(frame_idx)

        with open(json_list[frame_idx]) as f:
            image_info = json.load(f)

        image_timestamp = image_info['cam_tstamp']

        steer_labels = np.array(steer_labels)

        steer_idx = get_steering_id_from_timestamp(image_timestamp, steer_labels[:,0])
        angle = float(steer_labels[steer_idx][1])
        if sign[steer_idx][1] == 0:
            angle = -angle

        image_name = image_info['image_png']

        output_line = ','.join([image_name,"","",str(angle)])

        labels.write(output_line + '\n')

        frame = cv2.imread(image_folder + image_name)
        # frame = cv2.resize(frame, (200, 66))
        # frame = cv2.resize(frame, (455, 256))
        frame = cv2.resize(frame, (1280, 720))
        cv2.imwrite("train"+DATASET_NAME+"/"+image_name, frame)

        frame = cv2.putText(frame, str(angle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    labels.close()


def process_img_seg():
    main_folder = "camera_lidar_semantic/"
    # DATASET_NAME = "Audi1"
    # subfolder_list = ["20180807_145028", "20180810_142822", "20180925_101535", "20180925_112730", "20180925_124435", "20180925_135056", 
    # "20181008_095521", "20181016_082154", "20181016_125231", "20181107_132300"]

    # DATASET_NAME = "Audi2"
    # subfolder_list = ["20181107_132730", "20181107_133258", "20181107_133445", "20181108_084007", "20181108_091945", "20181108_103155", 
    # "20181108_123750", "20181108_141609", "20181204_135952", "20181204_154421", "20181204_170238"]

    # DATASET_NAME = "Audi3"
    # subfolder_list = ["20180925_112730", "20180925_124435", "20180925_135056", "20181008_095521", "20181016_082154", "20181016_125231",
    # "20181107_132300", "20181107_132730", "20181107_133258"]

    # DATASET_NAME = "Audi4"
    # subfolder_list = ["20181107_133445", "20181108_084007", "20181108_091945", "20181108_103155", "20181108_123750", "20181108_141609", 
    # "20181204_135952", "20181204_154421", "20181204_170238"]


    # DATASET_NAME = "Audi5train"
    # subfolder_list = ["20180925_112730", "20180925_124435", "20181008_095521", "20181016_082154", "20181016_125231", "20181107_132730", "20181108_141609"]

    # DATASET_NAME = "Audi5val"
    # subfolder_list = ["20180925_135056","20181107_132300"] # test

    DATASET_NAME = "Audi6train"
    subfolder_list = ["20181108_084007", "20181108_091945", "20181108_103155", "20181108_123750", "20181204_154421", "20181204_170238"]

    # DATASET_NAME = "Audi6val"
    # subfolder_list = ["20181107_133445", "20181204_135952"] # test

    if not os.path.exists("train"+DATASET_NAME):
        os.mkdir("train"+DATASET_NAME)
    if not os.path.exists("val"+DATASET_NAME):
        os.mkdir("val"+DATASET_NAME)

    if not os.path.exists("train"+DATASET_NAME+"segall"):
        os.mkdir("train"+DATASET_NAME+"segall")
    if not os.path.exists("val"+DATASET_NAME+"segall"):
        os.mkdir("val"+DATASET_NAME+"segall")
    # if not os.path.exists("train"+DATASET_NAME+"seg1"):
    #     os.mkdir("train"+DATASET_NAME+"seg1")
    # if not os.path.exists("val"+DATASET_NAME+"seg1"):
    #     os.mkdir("val"+DATASET_NAME+"seg1")

    labels_train = open("labels"+DATASET_NAME+"_train.csv", "w")
    labels_val = open("labels"+DATASET_NAME+"_val.csv", "w")
    labels_segall_train = open("labels"+DATASET_NAME+"segall_train.csv", "w")
    labels_segall_val = open("labels"+DATASET_NAME+"segall_val.csv", "w")
    # labels_seg1_train = open("labels"+DATASET_NAME+"seg1_train.csv", "w")
    # labels_seg1_val = open("labels"+DATASET_NAME+"seg1_val.csv", "w")

    STEP = 1
    global_frame_idx = 0
    total_eff_count = 0
    # VAL_STEP = 10           # general
    VAL_STEP = 1000000000   # training only
    # VAL_STEP = 1            # valid only

    for folder in subfolder_list:
        # bus_signal_file = "camera_lidar/20180810_150607/bus/20180810150607_bus_signals.json"
        bus_str = folder.replace('_', '')
        bus_signal_file = main_folder + folder + "/bus/" + bus_str + "_bus_signals.json"
        with open(bus_signal_file) as f:
            bus_signal = json.load(f)

        # image_folder = "camera_lidar_semantic/20180810_150607/camera/cam_front_center/"
        image_folder = main_folder + folder + "/camera/cam_front_center/"
        seg_folder = main_folder + folder + "/label/cam_front_center/"

        for frame in bus_signal:

            if not ("frontcenter" in frame['frame_name']):
                continue

            global_frame_idx += 1
            
            if frame['flexray'].get('steering_angle_calculated') == None:
                print(frame['frame_name'], ' no steering angle!!')
                continue

            if global_frame_idx % STEP != 0:
                continue

            print(global_frame_idx, " ", frame['frame_name'])
            total_eff_count += 1

            steer_labels = frame['flexray']['steering_angle_calculated']
            sign = frame['flexray']['steering_angle_calculated_sign']
            image_timestamp = frame['timestamp']

            steer_idx = get_steering_id_from_timestamp(image_timestamp, steer_labels['timestamps'])
            angle = float(steer_labels['values'][steer_idx])
            if sign['values'][steer_idx] < 0.5:
                angle = -angle

            image_name = frame['frame_name'].replace('.json', '.png')
            if not os.path.exists(image_folder + image_name):
                print(image_folder + image_name, ' not exists!!!!')

            seg_name = image_name.replace('camera', 'label')
            if not os.path.exists(seg_folder + seg_name):
                print(seg_folder + seg_name, ' not exists!!!!')

            seg_road_name = seg_name.replace('label', 'road')

            output_line = ','.join([image_name,"","",str(angle)])
            output_line_segall = ','.join([seg_name,"","",str(angle)])
            # output_line_seg1 = ','.join([seg_road_name,"","",str(angle)])



            frame = cv2.imread(image_folder + image_name)
            # frame = cv2.resize(frame, (200, 66))
            # frame = cv2.resize(frame, (455, 256))
            frame = cv2.resize(frame, (768, 512))

            # segmentation_all = cv2.imread(seg_folder + seg_name)
            # segmentation_all = cv2.resize(segmentation_all, (200, 66))
            # segmentation_all = cv2.resize(segmentation_all, (455, 256))

            # lower_black = np.array([255,0,255], dtype = "uint16")
            # upper_black = np.array([255,0,255], dtype = "uint16")
            # segmentation_road = cv2.inRange(segmentation_all, lower_black, upper_black)

            if total_eff_count % VAL_STEP == 0:
                labels_val.write(output_line + '\n')
                labels_segall_val.write(output_line_segall + '\n')
                cv2.imwrite("val"+DATASET_NAME+"/"+image_name, frame)
                # cv2.imwrite("val"+DATASET_NAME+"segall/"+seg_name, segmentation_all)
                # labels_seg1_val.write(output_line_seg1 + '\n')
                # cv2.imwrite("val"+DATASET_NAME+"seg1/"+seg_road_name, segmentation_road)
            else:
                labels_train.write(output_line + '\n')
                labels_segall_train.write(output_line_segall + '\n')
                cv2.imwrite("train"+DATASET_NAME+"/"+image_name, frame)
                # cv2.imwrite("train"+DATASET_NAME+"segall/"+seg_name, segmentation_all)
                # labels_seg1_train.write(output_line_seg1 + '\n')
                # cv2.imwrite("train"+DATASET_NAME+"seg1/"+seg_road_name, segmentation_road)

            # for debug
            # frame = cv2.putText(frame, str(angle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)

    labels_train.close()
    labels_val.close()
    labels_segall_train.close()
    labels_segall_val.close()
    # labels_seg1_train.close()
    # labels_seg1_val.close()


# Create array of RGB colour values from the given array of reflectance values
def colours_from_reflectances(reflectances):
    return np.stack([reflectances, reflectances, reflectances], axis=1)

def create_open3d_pc(lidar, cam_image=None):
    # create open3d point cloud
    pcd = o3.geometry.PointCloud()
    
    # assign point coordinates
    pcd.points = o3.utility.Vector3dVector(lidar['pcloud_points'])
    
    # assign colours
    if cam_image is None:
        median_reflectance = np.median(lidar['pcloud_attr.reflectance'])
        colours = colours_from_reflectances(lidar['pcloud_attr.reflectance']) / (median_reflectance * 5)
        
        # clip colours for visualisation on a white background
        colours = np.clip(colours, 0, 0.75)
    else:
        rows = (lidar['pcloud_attr.row'] + 0.5).astype(np.int)
        cols = (lidar['pcloud_attr.col'] + 0.5).astype(np.int)
        colours = cam_image[rows, cols, :] / 255.0
        
    pcd.colors = o3.utility.Vector3dVector(colours)
    
    return pcd

def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                        'camera_' + \
                        file_name_image[2] + '_' + \
                        file_name_image[3] + '.png'
    return file_name_image

def undistort_image(image, config, cam_name):
    if cam_name in ['front_left', 'front_center', \
                    'front_right', 'side_left', \
                    'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = \
                  np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = \
                  np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = \
                  np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']
        
        if (lens == 'Fisheye'):
            return cv2.fisheye.undistortImage(image, intr_mat_dist,\
                                      D=dist_parms, Knew=intr_mat_undist)
        elif (lens == 'Telecam'):
            return cv2.undistort(image, intr_mat_dist, \
                      distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image

def plot_lidar_id_vs_delat_t(image_info, lidar):
    timestamps_lidar = lidar['timestamp']
    timestamp_camera = image_info['cam_tstamp']
    time_diff_in_sec = (timestamps_lidar - timestamp_camera) / (1e6)
    lidar_ids = lidar ['lidar_id']
    pt.fig = pt.figure(figsize=(15, 5))
    pt.plot(time_diff_in_sec, lidar_ids, 'go', ms=2)
    pt.grid(True)
    ticks = np.arange(len(image_info['lidar_ids'].keys()))
    ticks_name = []
    for key in ticks:
        ticks_name.append(image_info['lidar_ids'][str(key)])
    pt.yticks(ticks, tuple(ticks_name))
    pt.ylabel('LiDAR sensor')
    pt.xlabel('delta t in sec')
    pt.title(image_info['cam_name'])
    pt.show()
    
def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    image = np.copy(image_orig)
    
    # get rows and cols
    rows = (lidar['pcloud_attr.row'] + 0.5).astype(np.int)
    cols = (lidar['pcloud_attr.col'] + 0.5).astype(np.int)
  
    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['pcloud_attr.distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['pcloud_attr.distance'])

    # get distances
    distances = lidar['pcloud_attr.distance']  
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                        np.sqrt(pixel_opacity), 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = \
                (1. - pixel_opacity) * \
                np.multiply(image[pixel_rows, pixel_cols, :], \
                colours[i]) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)

def generate_depthmap():
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10))

    bus_signal_file = "camera_lidar/20180810_150607/bus/20180810150607_bus_signals.json"
    with open(bus_signal_file) as f:
        bus_signal = json.load(f)
    steer_labels = bus_signal['steering_angle_calculated']['values']
    sign = bus_signal['steering_angle_calculated_sign']['values']

    with open ('camera_lidar/cams_lidars.json', 'r') as f:
        config = json.load(f)

    file_names_lidar = sorted(glob.glob('camera_lidar/20180810_150607/lidar/cam_front_center/*.npz'))

    for file_name_lidar in file_names_lidar:
        file_name_lidar = file_name_lidar.replace('\\', '/')

        lidar_front_center = np.load(file_name_lidar)
        print(list(lidar_front_center.keys()))
        rows = (lidar_front_center['pcloud_attr.row'] + 0.5).astype(np.int)
        cols = (lidar_front_center['pcloud_attr.col'] + 0.5).astype(np.int)
        points = lidar_front_center['pcloud_points']
        depthmap = np.zeros((1208,1920))
        print('rows ', rows.shape)
        print('cols ', cols.shape)
        print('points ', points.shape)
        print('distance ', lidar_front_center['pcloud_attr.distance'].shape)
        print('distance ', lidar_front_center['pcloud_attr.distance'])
        print('depth ', lidar_front_center['pcloud_attr.depth'].shape)
        print('depth ', lidar_front_center['pcloud_attr.depth'])
        print(rows)
        print(lidar_front_center['pcloud_attr.timestamp'].shape)
        print(lidar_front_center['pcloud_attr.boundary'].shape)
        print(lidar_front_center['pcloud_attr.boundary'])

        print(np.max(rows))
        print(np.max(cols))

        depth_array = lidar_front_center['pcloud_attr.depth']
        for i in range(len(depth_array)):
            # print(rows[i], ' ', cols[i], ' ', points[i])
            depthmap[rows[i],cols[i]] = depth_array[i]
            # cv2.circle(imgDepth, (int(points2D[i][0]), int(points2D[i][1])), radius=1, color=(v,v,v))

        depthmap = cv2.dilate(depthmap, kernel1)
        # depthmap = cv2.resize(depthmap, (1920//2, 1208//2))

        # cv2.imshow('depthmap ', depthmap/10.0)
        # cv2.waitKey(0)
        # pcd_front_center = create_open3d_pc(lidar_front_center)
        # o3.visualization.draw_geometries([pcd_front_center])

        seq_name = file_name_lidar.split('/')[2]
        file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)
        file_name_image = 'camera_lidar/20180810_150607/camera/cam_front_center/' + file_name_image
        image_front_center = cv2.imread(file_name_image)
        image_front_center = cv2.cvtColor(image_front_center, cv2.COLOR_BGR2RGB)

        pt.fig = pt.figure(figsize=(15, 15))

        # display image from front center camera
        pt.imshow(image_front_center)
        pt.axis('off')
        pt.title('front center 1')

        # undist_image_front_center = undistort_image(image_front_center, config, 'front_center')

        # pt.fig = pt.figure(figsize=(15, 15))
        # pt.imshow(undist_image_front_center)
        # pt.axis('off')
        # pt.title('front center 2')


        file_name_image_info = file_name_image.replace(".png", ".json")

        def read_image_info(file_name):
            with open(file_name, 'r') as f:
                image_info = json.load(f)
                
            return image_info

        image_info_front_center = read_image_info(file_name_image_info)  
        print(image_info_front_center)

        image = map_lidar_points_onto_image(image_front_center, lidar_front_center)
        pt.fig = pt.figure(figsize=(20, 20))
        pt.imshow(image)
        pt.axis('off')


        plt.show()


        break

if __name__ == '__main__':
    # process_img_seg()
    # process_img()
    generate_depthmap()