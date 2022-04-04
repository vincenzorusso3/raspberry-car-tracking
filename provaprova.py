import os
from typing import Sequence
from urllib.request import urlretrieve

import cv2
from motpy import Detection, MultiObjectTracker, NpImage
from motpy.core import setup_logger,Box
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track
from motpy.utils import ensure_packages_installed

import argparse
import sys
import time

import numpy as np

import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils

#from object_detector import Detection

from time import time
import argparse

from iou_tracker import track_iou
from util import load_mot, save_to_csv




load_dection = []
mint = 2 
  while(True):
     # read a frame
     id = 0
     for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)  
         load_dection = np.concatenate((load_dection, np.array([[cap.get(cv2.CAP_PROP_POS_FRAMES), id, x, y, w, h, 1]])), axis = 0)
         id += 1
    minT += 1
    if minT == mint: 
        detections = load_mot(load_dection, nms_overlap_thresh=None, with_classes=False)
        tracks = track_iou(detections, 0, 0.5, 0.5, mint)
        load_dection = []
        minT = 0