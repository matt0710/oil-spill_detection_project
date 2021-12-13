import torch
import cv2
import pathlib

#pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

whT = 416
confThreshold = 0.7
nmsThreshold = 0.3
classNames = 'oil-spill'

path = pathlib.Path(__file__).parent
parent = path.parent
modelPath = str(parent) + "\sources\yolov5s\\best_150.pt"


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
        box = bbox[i]
        x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames.upper()} {confs[i] * 100}',
                     (x_min, y_min - 10), cv2.FONT_ITALIC, 0.6, (255, 0, 255), 2)


def run_model(model, img):
    output = model(img)

    c = cv2.imread(img)
    find_objects(output.xyxy[0], c)

    cv2.imshow("image", c)
    cv2.waitKey(0)
