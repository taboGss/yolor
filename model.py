import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

## Funciones auxiliares provenientes de la clase LoadImages ##
from utils2.Loaders import load_classes, LoadImages


class yolor:
    def __init__(self, weights=None, imgsz=None, cfg=None, names=None):
        # Todos los parametros que sean None son inicializados por defecto
        # como en el repo de yolor https://github.com/WongKinYiu/yolor
        if weights == None:
            self.weights = 'yolor_p6.pt'
        else:
            if os.path.isfile(weights): self.weights = weights
            else: raise NameError(f"El archivo '{weights}' no existe") 
        
        if cfg == None:
            self.cfg = 'cfg/yolor_p6.cfg'
        else:
            if os.path.isfile(cfg): self.cfg = cfg
            else: raise NameError(f"El archivo '{cfg}' no existe")
        
        if names == None:
            self.names = 'data/coco.names'
        else:
            if os.path.isfile(names): self.names = names
            else: raise NameError(f"El archivo '{names}' no existe")
    
        if imgsz == None: self.imgsz = 1280
        else: self.imgsz = imgsz

        # Parametros para yolor
        self.conf_thres = 0.4 # object confidence threshold
        self.iou_thres = 0.5 # IOU threshold for NMS
        self.classes = None # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False # class-agnostic NMS
        self.augment = False # augmented inference

        # Seleccionamos el device (cuda/cpu) 
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        with torch.no_grad(): # Especificamos a Pytorch que vamos a hacer inferencia
            self.model = Darknet(self.cfg, self.imgsz).cuda()
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        
            self.model.to(self.device).eval()
            if self.half: self.model.half()  # to FP16

            # Get names and colors
            self.names = load_classes(self.names)
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

            ## Aparentemente el modelo debe ser inicializado a traves de una img del size adecuado ##
            img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
            _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
    
    @torch.no_grad()
    def detect(self, img0):
        path, img, im0, vid_cap = LoadImages(img0, 'null', img_size=self.imgsz, auto_size=64)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (self.names[int(cls)], conf) # Add bbox to image
                plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

        return det, im0
