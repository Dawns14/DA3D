import os.path as osp

from configs.config_general import BASE_DIR, IS_UBUNTU

CALIB = [-2.54, 0.3, 0.]

if IS_UBUNTU:
    SPLITTER = '/'
else:
    SPLITTER = '\\'

DELAY_FRAME = 0

SG_NORMAL = 0
SG_START_LABELING = 1

SL_START_LABELING = 0
SL_CLICK_CENTER = 1
SL_CLICK_FRONT = 2
SL_END_LABELING = 3

PATH_SEQ = 'E:\\radar_bin_lidar_bag_files\\generated_files'

PATH_IMG_G = osp.join(BASE_DIR, 'resources', 'imgs', 'prevg.png')
PATH_IMG_L = osp.join(BASE_DIR, 'resources', 'imgs', 'prevl.png')

PATH_IMG_F = osp.join(BASE_DIR, 'resources', 'imgs', 'prevf.png')
PATH_IMG_B = osp.join(BASE_DIR, 'resources', 'imgs', 'prevb.png')

FONT = 'Times New Roman'
FONT_SIZE = 10

W_BEV = 1280
H_BEV = 800
W_CAM = 320
H_CAM = 240

BT_LEFT = 0
BT_RIGHT = 1
BT_MIDDLE = 2

RANGE_Z = [-3, 3]

LIST_CLS_NAME = [
    'Sedan',
    'Bus or Truck',
    'Motorcycle',
    'Bicycle',
    'Pedestrian',
    'Pedestrian Group',
    'Bicycle Group',
]

LIST_CLS_COLOR = [
    [0,255,0],
    [0,50,255],
    [0,0,255],
    [0,200,255],
    [255,0,0],
    [255,0,100],
    [255,200,0],
]

LIST_Z_CEN_LEN = [
    [-1.5,0.95],
    [-1.5,1.5],
    [-1.5,1.5],
    [-1.5,1.5],
    [-1.5,1.5],
    [-1.5,1.5],
    [-1.5,1.5],
]

LINE_WIDTH = 2

RANGE_Y_FRONT = [-4,4]
RANGE_Z_FRONT = [-8,8]
IMG_SIZE_YZ = [300,150]
M_PER_PIX_YZ = 16./300.

RANGE_X_FRONT = [-8,8]
RANGE_Z_FRONT = [-8,8]
IMG_SIZE_XZ = [300,300]
M_PER_PIX_XZ = 16./300.
