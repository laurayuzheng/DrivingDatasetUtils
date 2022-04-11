# Waymo Open Dataset 

## Dependencies 
* Python 3.8 
* pip install waymo-open-dataset-tf-2-3-0
    - Installs Tensorflow 2.3
* conda install cudatoolkit=10.1
* pip install matplotlib
* pip install opencv-python

## Notes 
* Saves to .npz format. 
* In context of NPZ, x is the image, y is the label. 
* Saved shape of image is (num_frames, 224, 224, 3)
* Saved shape of label is (num_frames, 1)