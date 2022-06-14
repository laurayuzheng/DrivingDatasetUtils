# https://colab.research.google.com/github/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb

import os
import glob
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

## Path variables 
RAW_DATA_DIR = "/scratch/DrivingDatasetUtils/Waymo/data"
NUM_FRAMES = 8
IMG_SHAPE = (224, 224)

def show_camera_image(camera_image, camera_labels, layout, cmap=None):
    """Show a camera image and the given camera labels."""

    ax = plt.subplot(*layout)

    # Draw the camera labels.
    for camera_labels in camera_labels:
    # Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != camera_image.name:
            continue

        # Iterate over the individual labels.
        for label in camera_labels.labels:
            # Draw the object bounding box.
            ax.add_patch(patches.Rectangle(
            xy=(label.box.center_x - 0.5 * label.box.length,
                label.box.center_y - 0.5 * label.box.width),
            width=label.box.length,
            height=label.box.width,
            linewidth=1,
            edgecolor='red',
            facecolor='none'))

    # Show the camera image.
    plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
    plt.grid(False)
    plt.axis('off')
    plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
    plt.show()
    # plt.savefig('data/valWaymo/'+str(id)+'.png')

def generate_waymo_npz(data_dir: str, subfolder: str, output_dir: str):
    ''' Generates stacked NPZ files for use with 3d Transformers. 
        Processes images into stacks of [num_frames] images. 
        Saved x's have shape ([num_frames], 224, 224, 3). 
        Saved y's have shape ([num_frames], 1)

        Parameters: 
        data_dir (str): Path containing data (parent of train, val, test folders).
        subfolder (str): Subset of data, e.g. one of ["train", "val", "test"]
        output_dir (str): Similar to data_dir, and is the path to save .npz files to.
    '''

    # Create output path if not exists
    if not os.path.exists(os.path.join(output_dir, subfolder)):
        os.makedirs(os.path.join(output_dir, subfolder))

    # Define some variables for sake of naming files 
    seq_count = 0 
    t_sample_rate = 5
    t_sample_count = 0

    # Each sequence should consist of 8 images 
    num_frames = NUM_FRAMES

    # Get list of all tfrecord files
    tfrecord_list = glob.glob(os.path.join(data_dir, subfolder, "*.tfrecord"))
    
    # Define empty stacks 
    x = [] 
    y = [] 

    for tfrecord in tfrecord_list: 
        print("Processing tfrecord: ", tfrecord)
        dataset = tf.data.TFRecordDataset(tfrecord, compression_type='')
        last_context = -1 

        for i, data in enumerate(dataset): 
            frame = open_dataset.Frame() 
            frame.ParseFromString(bytearray(data.numpy()))
            
            current_context = open_dataset.Context.name
            if current_context != last_context:
                x = [] 
                y = [] 
                t_sample_count = 0

            # (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            for j, image in enumerate(frame.images):
                # For debugging
                # show_camera_image(image, frame.camera_labels, [3, 3, j+1])

                # Only want front image views 
                # print(open_dataset.CameraName.Name.Name(image.name).lower())
                if "front" != open_dataset.CameraName.Name.Name(image.name).lower():
                    continue

                if t_sample_count % t_sample_rate != 0: 
                    t_sample_count += 1
                    continue 
                
                # Reshape to (224, 224)
                show_camera_image(image, frame.camera_labels, [3, 3, j+1])
                img = cv2.resize(np.array(tf.image.decode_jpeg(image.image)), IMG_SHAPE)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                steering_angle = image.velocity.w_z*180/math.pi 
                
                # Add data to stack
                x.append(img)
                y.append([steering_angle])
                t_sample_count += 1
                last_context = open_dataset.Context.name

                if len(y) % num_frames == 0: 
                    x = np.array(x)
                    y = np.array(y)
                    # print("X shape: ", x.shape)
                    # print("Y shape: ", y.shape)

                    npz_save_path = os.path.join(output_dir, subfolder, str(seq_count)+'.npz')
                    print("Saving data: ", npz_save_path)
                    np.savez(npz_save_path, x=x, y=y)
                    seq_count += 1

                    # Empty the stacks
                    x = []
                    y = []
    
def generate_waymo_npz_single(data_dir: str, subfolder: str, output_dir: str):
    ''' Generates stacked NPZ files for use with 3d Transformers. 
        Processes images into stacks of [num_frames] images. 
        Saved x's have shape (224, 224, 3). 
        Saved y's have shape (1,)

        Parameters: 
        data_dir (str): Path containing data (parent of train, val, test folders).
        subfolder (str): Subset of data, e.g. one of ["train", "val", "test"]
        output_dir (str): Similar to data_dir, and is the path to save .npz files to.
    '''

    # Create output path if not exists
    if not os.path.exists(os.path.join(output_dir, subfolder)):
        os.makedirs(os.path.join(output_dir, subfolder))

    # Define some variables for sake of naming files 
    seq_count = 0 

    # Each sequence should consist of 8 images 
    num_frames = NUM_FRAMES

    # Get list of all tfrecord files
    tfrecord_list = glob.glob(os.path.join(data_dir, subfolder, "*.tfrecord"))
    
    # Define empty stacks 
    x = [] 
    y = [] 

    for tfrecord in tfrecord_list: 
        print("Processing tfrecord: ", tfrecord)
        dataset = tf.data.TFRecordDataset(tfrecord, compression_type='')
        
        for i, data in enumerate(dataset): 
            frame = open_dataset.Frame() 
            frame.ParseFromString(bytearray(data.numpy()))
            # (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            for j, image in enumerate(frame.images):
                # For debugging
                # show_camera_image(image, frame.camera_labels, [3, 3, j+1])

                # Only want front image views 
                if "front" != open_dataset.CameraName.Name.Name(image.name).lower():
                    continue
                
                # Reshape to (224, 224)
                img = cv2.resize(np.array(tf.image.decode_jpeg(image.image)), IMG_SHAPE)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                steering_angle = image.velocity.w_z*180/math.pi 
                
                # Add data to stack
                x = np.array(img)
                y = np.array([steering_angle])
                # print("X shape: ", x.shape)
                # print("Y shape: ", y.shape)

                npz_save_path = os.path.join(output_dir, subfolder, str(seq_count)+'.npz')
                print("Saving data: ", npz_save_path)
                np.savez(npz_save_path, x=x, y=y)
                seq_count += 1


if __name__ == "__main__":
    subfolders = ["train", "val", "test"] # use for loop later
    # subfolders = ["test"]
    output_dir = "/scratch/vroom/data/Waymo8_Fixed"

    for folder in subfolders:
        generate_waymo_npz(RAW_DATA_DIR, folder, output_dir)

    # for folder in subfolders:
    #     generate_waymo_npz_single(RAW_DATA_DIR, folder, output_dir)
