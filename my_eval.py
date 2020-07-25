from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import cv2
from ssd_small import build_small_ssd


# img_path = 'data/helmet_dataset/scenario1-share/JPEGImages/0318_5.jpg'
# img = cv2.imread(img_path)
# # BGR
# cv2.rectangle(img, (100, 100), (250, 180), (0, 0, 255), 2)
# cv2.putText(img, 'name:%.2f' % 0.97888, (100 + 2, 100 + 11), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255))
# cv2.imshow('test', img)
# cv2.waitKey(0)
# cv2.imwrite('./eval/imgs/result.jpg', img)

# ssd_net = build_small_ssd('test', 300, 5)
# net = torch.nn.DataParallel(ssd_net)
# net.load_state_dict(torch.load('weights/helmet_detection_net.pth'))
# torch.save(ssd_net.state_dict(), 'detection_net.pth')
















