import numpy as np
import cv2
import xml.etree.cElementTree as ET
import os
import pathlib

path = pathlib.Path(__file__).parent
images = os.listdir(str(path) + '\images\ground_truth\\')
count = 0
emptyCount = 0


def first_check(start, counter):
    filename = ET.SubElement(start, "filename")
    if counter < 10:
        filename.text = "img_000" + str(counter) + ".jpg"

    elif 10 <= counter < 100:
        filename.text = "img_00" + str(counter) + ".jpg"

    elif 100 <= counter < 1000:
        filename.text = "img_0" + str(counter) + ".jpg"

    else:
        filename.text = "img_" + str(counter) + ".jpg"

    return filename.text


def second_check(counter):

    if counter < 10:
        string = "img_000" + str(counter) + ".xml"

    elif 10 <= counter < 100:
        string = "img_00" + str(counter) + ".xml"

    elif 100 <= counter < 1000:
        string = "img_0" + str(counter) + ".xml"

    else:
        string = "img_" + str(counter) + ".xml"

    return string


for image in images:
    count += 1
    im = cv2.imread(str(path) + '\images\ground_truth\\' + image)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)# change color format

    lower_blue = np.array([90, 50, 70])
    upper_blue = np.array([128, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        emptyCount += 1

    if len(contours) != 0 or emptyCount < 70:

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "img"

        first_check(root, count)
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unknown"
        ET.SubElement(source, "annotation").text = "Unknown"
        ET.SubElement(source, "image").text = "Unknown"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "1250"
        ET.SubElement(size, "height").text = "650"
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 10:
                cv2.drawContours(im, contour, -1, (0, 255, 0), 3)  # -1 draw all countours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)

                item = ET.SubElement(root, "object")
                ET.SubElement(item, "name").text = "oil-spill"
                ET.SubElement(item, "truncated").text = "0"
                ET.SubElement(item, "occluded").text = "0"
                ET.SubElement(item, "difficult").text = "0"

                bndbox = ET.SubElement(item, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(x)
                ET.SubElement(bndbox, "ymin").text = str(y)
                ET.SubElement(bndbox, "xmax").text = str(x+w)
                ET.SubElement(bndbox, "ymax").text = str(y+h)
                attributes = ET.SubElement(item, "attributes")
                attribute = ET.SubElement(attributes, "attribute")
                ET.SubElement(attribute, "name").text = "rotation"
                ET.SubElement(attribute, "value").text = "0.0"

        os.chdir(str(path) + '\\annotation\\')
        tree = ET.ElementTree(root)
        tree.write(second_check(count))

        #cv2.imshow("image", im)
        #cv2.imshow("mask", mask)

        #cv2.waitKey(0)