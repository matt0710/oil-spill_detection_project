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
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {confs[i] * 100}',
                    (x, y - 10), cv2.FONT_ITALIC, 0.6, (255, 0, 255), 2)


def run_model(net, img):
    cap = cv2.imread(img)
    blob = cv2.dnn.blobFromImage(cap, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)

    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)

    outputs = net.forward(outputNames)

    find_objects(outputs, cap)

    cv2.imshow("Image", cap)
    cv2.waitKey(0)
