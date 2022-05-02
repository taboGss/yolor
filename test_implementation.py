from pkgutil import extend_path
from model import yolor
import numpy as np
import cv2

"""cap = cv2.VideoCapture('/home/tabo/Documents/Spinar/people_detection/test.mp4')
if not cap.isOpened():
    raise NameError("Error opening video stream or file")

model = yolor() # Creamos el modelo usando la implementacion de yolor

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    detections, img = model.detect(frame) # Realizamos la inferencia
    print(detections)
    
    cv2.imshow('yolor', img)
    if cv2.waitKey(25) & 0xFF == ord('q'): break"""

img  = cv2.imread('workers.jpg')
model = yolor(weights='yolor_p6.pt', 
              names='/home/tabo/Documents/Spinar/yolor2/data/coco.names')

detections, img_yolor = model.detect(img)
print(detections)

cv2.imshow("PPE detection", img_yolor)
cv2.waitKey(0)

