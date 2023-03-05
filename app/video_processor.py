from logging import Logger
from pathlib import Path

import cv2

from app.perspective_transform import PerspectiveTransform
from app.models.plate_detection_model import PlateDetectionModel
from app.plate_recognition import stack_images, text_recognition
from app.models.text_detection_model import TextDetectionModel


def process_video(video_path: Path):
    """
    Recognizes all numbers on the video
    """
    if not video_path.exists():
        raise ValueError(f"Incorrect video path: {video_path}")
    plate_detection_model = PlateDetectionModel()
    text_detection_model = TextDetectionModel()
    perspective_transformer = PerspectiveTransform(text_detection_model)

    video = cv2.VideoCapture(str(video_path))
    # assert video.isOpened()
    numbers = {}
    try:
        for object_id, box, plate_image in plate_detection_model.find_boxes_on_video(video, step=5):
            numbers.setdefault(object_id, []).append(plate_image)
    finally:
        video.release()

    text_result = []
    plate_images = []
    for id, images in numbers.items():
        stacked_image = stack_images(images)
        rotated_image = perspective_transformer.transform(stacked_image)
        text = text_recognition(rotated_image, text_detection_model)
        text_result.append(text)
        plate_images.append(stacked_image)
    return list(zip(text_result, plate_images))


