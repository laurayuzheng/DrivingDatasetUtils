import os, glob 
import csv
import cv2
import numpy as np

import os, glob 
import csv
import cv2
import numpy as np

TOTAL_FRAMES = 63825
STEERING_RATIO = 15.06

def generate_npz(raw_datadir, savedir, num_frames=8, trainval_split=0.1, test_split=0.1):

    saved_npz_count = 0 
    total_npz = TOTAL_FRAMES // num_frames
    test_split_idx = int(total_npz * (1-test_split)) - 1
    val_split_idx = int(test_split_idx * (1-trainval_split))

    os.makedirs(os.path.join(savedir, "train"), exist_ok=True)
    os.makedirs(os.path.join(savedir, "test"), exist_ok=True)
    os.makedirs(os.path.join(savedir, "val"), exist_ok=True)

    # seq_list = glob.glob(os.path.join(raw_datadir, "*", ""))
    csv_file = os.path.join(raw_datadir, "data.txt")

    # label_df = pd.read_csv(csv_file, header=0)
    # print(label_df)

    frame_count = 0 
    x = [] 
    y = [] 
    with open(csv_file, "r") as f:
        heading = next(f)
        reader = csv.reader(f, delimiter=",")
        
        for row in reader: 
            data = row[0].split(" ")
            img_path = os.path.join(raw_datadir, data[0])
            # print(img_path)

            frame = cv2.imread(img_path)
            frame = cv2.resize(frame, (224, 224))
            label = float(data[1]) / STEERING_RATIO
            # print(label)

            x.append(frame)
            y.append([label])

            if len(y) % num_frames == 0: 
                x = np.array(x)
                y = np.array(y)

                # print(x.shape, y.shape)
                subfolder = "train" 
                if saved_npz_count > test_split_idx:
                    subfolder = "test"
                elif saved_npz_count > val_split_idx:
                    subfolder = "val"
                
                npz_savepath = os.path.join(savedir, subfolder, str(saved_npz_count)+".npz")
                print("Saved %s out of %d total" % (npz_savepath, total_npz))
                np.savez(npz_savepath, x=x, y=y)
                saved_npz_count += 1

                x = [] 
                y = []
                # break
            
            frame_count += 1
        


if __name__ == "__main__":
    savedir = "/scratch/vroom/data/SullyChen8"
    raw_unprocessed_dir = "/scratch/vroom/data/SullyChenRaw/"
    print("Start")
    generate_npz(raw_unprocessed_dir, savedir)
    print("Finished generating at ", savedir)