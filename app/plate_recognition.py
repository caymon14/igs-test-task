from typing import List
import numpy as np
import cv2
import pytesseract
from app.utils.image_utils import cut_box


def get_blur_value(image):
    canny = cv2.Canny(image, 50, 250)
    return np.mean(canny)


def calculate_coef(image):
    diagonal = (image.shape[0] ** 2 + image.shape[1] ** 2) ** 0.5
    size_coef = image.shape[1] / (image.shape[0] * 2.5)
    if size_coef > 1.0:
        size_coef = 1 / size_coef
    blur_coef = get_blur_value(image) ** 1 / 1000
    return (diagonal * blur_coef * size_coef)


def select_best(img_list, count=5):
    coef_list = []
    for img in img_list:
        coef_list.append(calculate_coef(img))
    sorted_coef_list = coef_list.copy()
    sorted_coef_list.sort()
    result = [img_list[coef_list.index(coef)] for coef in sorted_coef_list[-count:]]
    return result, len(result) - 1


def stack_images_ECC(image_list, best_index, resize_index=2):
    M = np.eye(3, 3, dtype=np.float32)

    best_shape = image_list[best_index].shape
    best_size = (best_shape[1]*resize_index, best_shape[0]*resize_index)
    stacked_image = cv2.resize(image_list[best_index], best_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    best_image = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2GRAY)

    for idx, image in enumerate(image_list):
        if idx != best_index:
            image = cv2.resize(image, best_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            # Estimate perspective transform
            s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), best_image, M, cv2.MOTION_HOMOGRAPHY)
            w, h, _ = image.shape
            # Align image to first image
            image = cv2.warpPerspective(image, M, (h, w))
            stacked_image += image

    stacked_image /= len(image_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    return stacked_image, best_size


def stack_images(images: List):
    selected_images, best_index = select_best(images)
    stacked_image, size = stack_images_ECC(selected_images, best_index=best_index)
    return stacked_image


def letter_image_preprocessing(letter_image):
    gray = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    white = [255, 255, 255]
    result = cv2.copyMakeBorder(invert, 5, 5, 5, 5, cv2.BORDER_CONSTANT,value=white)
    return result


def text_recognition(image, text_detection_model):
    image = image.copy()
    letter_boxes = text_detection_model.get_letter_boxes(image)
    box_images = []
    for box in letter_boxes:
        box_images.append(cut_box(image, box))

    text_result = []
    for letter_image in box_images:
        preprocessed_letter_image = letter_image_preprocessing(letter_image)
        whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = pytesseract.image_to_string(preprocessed_letter_image, lang='eng', config=f'--psm 6 -c tessedit_char_whitelist={whitelist}')
        text_result.append(result.replace("\n", "").strip())
    if text_result:
        return str.join("", text_result)
    else:
        return ""
