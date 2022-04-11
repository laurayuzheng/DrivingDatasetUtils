import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import cv2
import math

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

# def show_camera_image(camera_image, camera_labels, id, layout, cmap=None):
# 	"""Show a camera image and the given camera labels."""

# 	ax = plt.subplot(*layout)

# 	# Draw the camera labels.
# 	for camera_labels in frame.camera_labels:
# 	# Ignore camera labels that do not correspond to this camera.
# 		if camera_labels.name != camera_image.name:
# 			continue

# 		# Iterate over the individual labels.
# 		for label in camera_labels.labels:
# 			# Draw the object bounding box.
# 			ax.add_patch(patches.Rectangle(
# 			xy=(label.box.center_x - 0.5 * label.box.length,
# 				label.box.center_y - 0.5 * label.box.width),
# 			width=label.box.length,
# 			height=label.box.width,
# 			linewidth=1,
# 			edgecolor='red',
# 			facecolor='none'))

# 	# Show the camera image.
# 	plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
# 	plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
# 	plt.grid(False)
# 	plt.axis('off')
# 	plt.savefig('data/valWaymo/'+str(id)+'.png')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='batch val test')
    parser.add_argument('--gpu_id', required=False, metavar="gpu_id", help='gpu id (0/1)')
    parser.add_argument('--start_id', '-sid', required=False, type=int, default=0, help='start_id.')
    args = parser.parse_args()

    data_folder = "data/val/"
    tffiles = glob.glob(data_folder+"*.tfrecord")
    # FILENAME = "/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"

    # f_label_out_global = open("data/val/labelsWaymo_val.csv", "a")

    result = cv2.VideoWriter('filename.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         30, (640, 480))

    # frame_id_global = args.start_id
    cnt = 0
    for tffile in tffiles:
        cnt += 1
        # f_start_id = open("start_id.txt", "r")
        # frame_id_global = int(next(f_start_id))
        # f_start_id.close()
        if not (cnt == 1 or cnt == 3 or cnt == 5):
            continue

        frame_id_global = 0

        dataset = tf.data.TFRecordDataset(tffile, compression_type='')

        label_file_1 = tffile.replace(".tfrecord", ".csv")
        f_label_out_1 = open(label_file_1, "w")

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            for index, image in enumerate(frame.images):
                # show_camera_image(image, frame.camera_labels, frame_id_global, [3, 3, index+1])
                if open_dataset.CameraName.Name.Name(image.name) != "FRONT":
                    continue

                # img = cv2.resize(np.array(tf.image.decode_jpeg(image.image)), (480, 320))
                img = cv2.resize(np.array(tf.image.decode_jpeg(image.image)), (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite('data/valWaymoM/'+str(frame_id_global)+'.jpg', img)
                result.write(img)

                # img = cv2.resize(img, (200, 66))
                # cv2.imwrite('data/valWaymo/'+str(frame_id_global)+'.jpg', img)

                one_record = str(frame_id_global)+".jpg,,," + str(image.velocity.w_z*180/math.pi)
                print(one_record)
                # f_label_out_1.write(one_record+"\n")
                # f_label_out_global.write(one_record+"\n")

                frame_id_global += 1
            # break				

        f_label_out_1.close()

        print('frame_id_global ', str(frame_id_global))
        # f_start_id = open("start_id.txt", "w")
        # f_start_id.write(str(frame_id_global))
        # f_start_id.close()

        # if cnt >= 3:
        # 	break

    result.release()

    # f_label_out_global.close()

