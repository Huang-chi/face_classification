from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
# from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file

# parameters for loading data and images



def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args

# @contextmanager
# def video_capture(*args, **kwargs):
#     cap = cv2.VideoCapture(*args, **kwargs)
#     try:
#         yield cap
#     finally:
#         cap.release()
