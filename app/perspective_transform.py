from app.utils.image_utils import wrap_perspective


class PerspectiveTransform:
    """For perspective transformation and normalisation"""
    def __init__(self, text_detection_model):
        self.text_detection_model = text_detection_model

    def transform(self, image):
        boxes = self.text_detection_model.get_text_boxes(image)
        max_size = 0
        max_box = None
        for box in boxes:
            size = abs(box[0][0] - box[1][0]) * abs(box[0][1] - box[3][1])
            if size > max_size:
                max_box = box
                max_size = size
        if max_box is not None:
            rotated_image, base_box = wrap_perspective(image, max_box)
            return rotated_image
        else:
            return image


