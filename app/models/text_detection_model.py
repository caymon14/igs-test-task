from craft_text_detector import load_craftnet_model, load_refinenet_model, get_prediction


class TextDetectionModel:
    '''Text position detection'''
    def __init__(self):
        self.refine_net = load_refinenet_model(cuda=False)
        self.craft_net = load_craftnet_model(cuda=False)

    def get_text_boxes(self, image):
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=False,
            long_size=1280
        )
        boxes = prediction_result["boxes"]
        return boxes

    def get_letter_boxes(self, image):
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.9,
            link_threshold=1.0,
            low_text=0.3,
            cuda=False,
            long_size=1280
        )
        boxes = prediction_result["boxes"]
        sorted_boxes = boxes[boxes[:, 0, 0].argsort()]
        return sorted_boxes