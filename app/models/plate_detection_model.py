import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time
from torchvision import transforms
# from pathlib import Path
from PIL import Image
from IPython.display import display
import yolov5
import torchvision
from IPython.display import clear_output, display, HTML
import pytesseract

from app.objects_controller import ObjectController
from app.utils.coordinat_utils import letterbox, to_original_coordinates


class PlateDetectionModel:
    """Finds plate position on images and video"""
    def __init__(self):
        model = yolov5.load('keremberke/yolov5m-license-plate')
        model.conf = 0.25
        model.iou = 0.45
        model.agnostic = False
        model.multi_label = False
        model.max_det = 1000

        device = torch.device("cpu")
        model = model.to(device)
        _ = model.eval()
        self.model = model

    def find_boxes_on_image(self, image):
        """
        Finds a position of plate on image
        :return: The boxes for all plates as list.
        """
        image, ratio, (dw, dh) = letterbox(image, stride=64, auto=True)
        with torch.no_grad():
            results = self.model([image])
        predictions = results.pred[0]
        boxes = predictions[:, :4]
        original_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            original_box = to_original_coordinates(xmin, ymin, xmax, ymax, ratio, dw, dh)
            original_boxes.append(original_box)
        return original_boxes

    def find_boxes_on_video(self, video, step=1):
        """
        Finds a position of plate on video
        :return: The plate id, box and plate image as a tuple for each processed frame.
        """
        objects_controller = ObjectController()
        counter = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            counter += 1

            if counter == step:
                resized_frame, ratio, (dw, dh) = letterbox(frame, stride=64, auto=True)
                results = self.model([resized_frame])
                predictions = results.pred[0]
                boxes = predictions[:, :4]

                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    object_id = objects_controller.find_object_id(box)
                    objects_controller.refresh_coordinates(object_id, box)

                    original_box = to_original_coordinates(xmin, ymin, xmax, ymax, ratio, dw, dh)
                    original_xmin, original_ymin, original_xmax, original_ymax = original_box
                    box_image = frame[int(original_ymin):int(original_ymax), int(original_xmin):int(original_xmax)]
                    yield (object_id, original_box, box_image)
                counter = 0



