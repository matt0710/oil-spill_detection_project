import os
import cv2

from util.models import yolov4
from util.models import yolov5
from util.models import yolov5s6


print("Welcome, please insert the model's name that you want to execute: ")
model_name = input()

images = os.listdir(str(os.getcwd()) + '\images\\test_images\\')

if model_name == 'yolov4':
    my_yolov4 = yolov4.create_model()
    for image in images:
        yolov4.run_model(my_yolov4, str(os.getcwd()) + '\images\\test_images\\' + image, str(os.getcwd()) + '\images\\ground_truth\\' + image)
        if 0xFF == ord("n"):
            break
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


elif model_name == 'yolov5s':
    my_yolov5 = yolov5.create_model()
    for image in images:
        yolov5.run_model(my_yolov5, str(os.getcwd()) + '\images\\test_images\\' + image, str(os.getcwd()) + '\images\\ground_truth\\' + image)
        if 0xFF == ord("n"):
            break
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

elif model_name == 'yolov5s6':
    my_yolov5s6 = yolov5s6.create_model()
    for image in images:
        yolov5s6.run_model(my_yolov5s6, str(os.getcwd()) + '\images\\test_images\\' + image, str(os.getcwd()) + '\images\\ground_truth\\' + image)
        if 0xFF == ord("n"):
            break
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

else:
    print("Usage error: the implemented models are <yolov4> <yolov5s> <yolov5s6>")

