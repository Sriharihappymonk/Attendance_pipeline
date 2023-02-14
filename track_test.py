import argparse
import asyncio
import subprocess as sp
import time
import ipfsApi
from db_insert import insert_db
import os
import fnmatch
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from math import ceil
from PIL import Image
from multiprocessing import Process, Queue
import glob
from nanoid import generate
from io import BytesIO
import datetime #datetime module to fetch current time when frame is detected

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from matplotlib import gridspec

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

from keras.applications.resnet import ResNet50 
#from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet import preprocess_input 
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import cv2
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from collections import Counter

#face_detection
import lmdb
# import face_lmdb
import json
import face_recognition 

#ipfs
#api = ipfsApi.Client('216.48.181.154', 5001)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'Detection') not in sys.path:
    sys.path.append(str(ROOT / 'Detection'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from Detection.models.common import DetectMultiBackend
from Detection.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from Detection.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from Detection.utils.torch_utils import select_device, time_sync
from Detection.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

 # Load model
device_track=''
devices = select_device(device_track)
model = DetectMultiBackend(WEIGHTS / 'best_nov16.pt', device=devices, dnn=False, data=None, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)  # check image size

TOLERANCE = 0.70
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'svm'

batchId = 0
temp_data = {}
member_type_dict = {"00": "known whitelist", "01": "known blacklist", "10": "unknown first time", "11": "unknown repeat"}

person_count = []
vehicle_count = []
avg_Batchcount_person =[]
avg_Batchcount_vehicel = []
track_person = []
track_vehicle = []
batch_person_id = []
detect_count = []
act_output = []
batch_person_id = []
face_did_location_store = dict()
count_person =0
known_whitelist_faces = []
known_whitelist_id = []
known_blacklist_faces = []
known_blacklist_id = []
face_did_encoding_store = dict()
track_type = []
timestamp = []
video_cid = []


#load lmdb
env = lmdb.open('/home/srihari/Face_Recognition_pipeline/lmdb/face-detection.lmdb',
                max_dbs=10, map_size=int(100e9))


# Now create subdbs for known and unknown people.
known_db = env.open_db(b'white_list')
unknown_db = env.open_db(b'black_list')

def lmdb_known():
    # Iterate each DB to show the keys are sorted:
    with env.begin() as txn:
        list1 = list(txn.cursor(db=known_db))
        
    db_count_whitelist = 0
    for key, value in list1:
        #fetch from lmdb
        with env.begin() as txn:
            re_image = txn.get(key, db=known_db)
            # Deserialization
            print("Decode JSON serialized NumPy array")
            decodedArrays = json.loads(re_image)
            
            finalNumpyArray = np.asarray(decodedArrays["array"], dtype="uint8")
            
            # Load an image
            # image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
            image = finalNumpyArray
            ratio = np.amax(image) / 256
            image = (image / ratio).astype('uint8')
            
            # Get 128-dimension face encoding
            # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
            try :
                encoding = face_recognition.face_encodings(image)[0]
            except IndexError as e  :
                print( "Error ", IndexError , e)
                continue
            
            # Append encodings and name
            known_whitelist_faces.append(encoding)
            known_whitelist_id.append(key.decode())
            db_count_whitelist += 1
            
    print(db_count_whitelist, "total whitelist person")


def lmdb_unknown():
    # Iterate each DB to show the keys are sorted:
    with env.begin() as txn:
        list2 = list(txn.cursor(db=unknown_db))
        
    db_count_blacklist = 0
    for key, value in list2:
        #fetch from lmdb
        with env.begin() as txn:
            re_image = txn.get(key, db=unknown_db)
            # Deserialization
            print("Decode JSON serialized NumPy array")
            decodedArrays = json.loads(re_image)
            
            finalNumpyArray = np.asarray(decodedArrays["array"],dtype="uint8")
            
            # Load an image
            # image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
            image = finalNumpyArray
            ratio = np.amax(image) / 256
            image = (image / ratio).astype('uint8')
            
            # Get 128-dimension face encoding
            # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
            try :
                encoding = face_recognition.face_encodings(image)[0]
            except IndexError as e  :
                print( "Error ", IndexError , e)
                continue
            
            # Append encodings and name
            known_blacklist_faces.append(encoding)
            known_blacklist_id.append(key.decode())
            db_count_blacklist += 1
    
    print(db_count_blacklist, "total blacklist person")
# def Attendance(name):
#     with open('attendance.csv','r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         # if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')

@torch.no_grad()
def run(
        source,
        queue1 = Queue(),
        yolo_weights=WEIGHTS / 'best_nov16.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        save_vid=True,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'Nats_output/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    global batchId 
    batchId = batchId +  18

    batch_info = {}
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, devices, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources


    # Run tracking
    frame_count = 0
    frame_id = 0
    batch_info[batchId]={}
   
    #model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    #print(list(dataset))
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        print(vid_cap)
        frame_id = frame_id + 1
        batch_info[batchId][frame_id]={}
        t1 = time_sync()
        im = torch.from_numpy(im).to(devices)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        #print(im)
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        #print(im)
        break



if __name__ == "__main__":
    run(source="/home/srihari/Face_Recognition_pipeline/Nats_video3-8_Trim.mp4")
    
