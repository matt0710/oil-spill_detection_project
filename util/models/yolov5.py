import torch
import cv2
import numpy as np
import pathlib

#pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

whT = 416
confThreshold = 0.7
nmsThreshold = 0.3
classNames = 'oil-spill'

path = pathlib.Path(__file__).parent
parent = path.parent
modelPath = str(parent) + "\sources\yolov5s\\best_150.pt"

def print_ground_truth(im):

    im = str(im).replace("jpg", "png")

    img = cv2.imread(im)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # change color format

    lower_blue = np.array([90, 50, 70])
    upper_blue = np.array([128, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 10:
            cv2.drawContours(mask, contour, -1, (0, 255, 0), 3)  # -1 draw all countours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow("mask", mask)


def create_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=modelPath)
    model.conf = 0.7  # NMS confidence threshold
    model.iou = 0.3  # NMS IoU threshold

    return model


def find_objects(outputs, img):
    bbox = []
    classIds = []
    confs = []

    for output in outputs:

        classId = output[5]
        confidence = output[4]
        if confidence > confThreshold:
            x1, y1, x2, y2 = int(output[0]), int(output[1]), int(output[2]), int(output[3])
            bbox.append([x1, y1, x2, y2])
            classIds.append(classId)
            confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = np.squeeze(i)
        box = bbox[i]
        x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames.upper()} {confs[i] * 100}',
                     (x_min, y_min - 10), cv2.FONT_ITALIC, 0.6, (255, 0, 255), 2)


def run_model(model, img, ground_truth_path):
    output = model(img)

    c = cv2.imread(img)
    find_objects(output.xyxy[0], c)

    cv2.imshow("image", c)
    print_ground_truth(ground_truth_path)
    cv2.waitKey(0)
