import cv2
import numpy as np

def add_boxes_to_image(image, boxes):
    new_image = image.copy()
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        cv2.putText(new_image, "Number Plate", (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (228, 79, 215), 2)
        cv2.rectangle(
            new_image,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            color=(228, 79, 215),
            thickness=1,
            lineType=cv2.LINE_AA
        )
    return new_image


def cut_boxes_from_image(image, boxes):
    result_images = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        plate_roi = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        result_images.append(plate_roi)
    return result_images


def wrap_perspective(image, box):
    test_box = box.copy()

    box_x_list = test_box[:, 0]
    box_y_list = test_box[:, 1]
    ih, iw = image.shape[:2]

    x_list = (box_x_list - np.min(box_x_list)) * (iw / (np.max(box_x_list) - np.min(box_x_list)))
    y_list = (box_y_list - np.min(box_y_list)) * (ih / (np.max(box_y_list) - np.min(box_y_list)))
    source_box = [[x_list[idx], y_list[idx]] for idx in range(len(test_box))]

    ih, iw = image.shape[:2]
    tar = np.float32([[0, 0], [iw, 0], [iw, ih], [0, ih]])
    M = cv2.getPerspectiveTransform(np.float32(source_box), tar)

    ih, iw = image.shape[:2]
    transformed_image = cv2.warpPerspective(image, M, (iw, ih), flags=cv2.INTER_CUBIC)
    return transformed_image, source_box


def cut_box(image, box):
    image_copy = image.copy()
    bw, bh = (int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1))

    tar = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    M = cv2.getPerspectiveTransform(np.float32(box), tar)

    transformed_image = cv2.warpPerspective(image_copy, M, (bw, bh), flags=cv2.INTER_CUBIC)
    return transformed_image


