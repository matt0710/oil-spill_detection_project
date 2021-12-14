import json
import os
import pathlib

import numpy as np
import cv2


def first_check(counter):
    if counter < 10:
        string = "img_000" + str(counter) + ".jpg"
    elif 10 <= counter < 100:
        string = "img_00" + str(counter) + ".jpg"
    elif 100 <= counter < 1000:
        string = "img_0" + str(counter) + ".jpg"
    else:
        string = "img_" + str(counter) + ".jpg"

    return string

path = pathlib.Path(__file__).parent
images = os.listdir(str(path) + '\images\ground_truth\\')


data = {}
data['info'] = []
data['licenses'] = []
data['categories'] = []
data['images'] = []
data['annotations'] = []

out_file = open("train_polygon.json", "w")

data['info'].append(
    {"year": "2021",
        "version": "1",
        "description": "Created by me",
        "contributor": "",
        "date_created": "2021-12-11T15:33:06+00:00"}
)

data['licenses'].append({
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }
)

data['categories'].append({
            "id": 0,
            "name": "oil-spill",
            "supercategory": "none"
        }
)


counter = 0
index = 100000

for image in images:
    counter += 1
    im = cv2.imread(str(path) + '\images\ground_truth\\' + image)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # change color format

    lower_blue = np.array([90, 50, 70])
    upper_blue = np.array([128, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    annotations = []
    bbox = []

    if len(contours) != 0:

        data['images'].append(
            {
                "id": counter,
                "license": 1,
                "file_name": first_check(counter),
                "height": 650,
                "width": 1250,
                "date_captured": "2021-12-21T15:33:06+00:00"
            }
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:
                x, y, w, h = cv2.boundingRect(contour)
                data['annotations'].append({
                    "id": index,
                    "image_id": counter,
                    "category_id": 0,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "segmentation": [contour.astype(float).flatten().tolist()],
                    "iscrowd": 0
                    }
                )
                index += 1

json.dump(data, out_file)
