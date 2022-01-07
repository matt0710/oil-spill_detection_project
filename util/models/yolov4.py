import cv2
import numpy as np
import pathlib

path = pathlib.Path(__file__).parent
parent = path.parent

whT = 416
confThreshold = 0.7
nmsThreshold = 0.3

classNames = 'oil-spill'

modelConfiguration = str(parent) + "\sources\yolov4\\yolov4-tiny-oil_spill.cfg"
modelWeight = str(parent) + "\sources\yolov4\\yolov4-tiny-oil_spill_v5_v1.weights"


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
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net


def find_objects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            # let find highest probability value
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)  # it gives us pixel values
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = np.squeeze(i)
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {confs[i] * 100}',
                    (x, y - 10), cv2.FONT_ITALIC, 0.6, (255, 0, 255), 2)


def run_model(net, img, ground_truth_path):
    cap = cv2.imread(img)
    blob = cv2.dnn.blobFromImage(cap, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)

    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[np.squeeze(i) - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    find_objects(outputs, cap)

    cv2.imshow("Image", cap)
    print_ground_truth(ground_truth_path)
    cv2.waitKey(0)
