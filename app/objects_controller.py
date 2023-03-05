

class ObjectController:
    """Implementation of a simple algorithm for tracking objects"""
    def __init__(self, border=50):
        self.objects = {}
        self.border = border

    def find_object_id(self, box, border=100):
        xmin, ymin, xmax, ymax = box
        mean_x, mean_y = (xmax + xmin) / 2, (ymax + ymin) / 2
        diagonal = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5

        if not self.objects:
            return 0
        distances = []
        ids = []
        for obj_id, (obj_x, obj_y) in self.objects.items():
            distance = (abs(mean_x - obj_x) ** 2 + abs(mean_y - obj_y) ** 2) ** 0.5
            if distance < border or distance < diagonal * 3:
                distances.append(distance)
                ids.append(obj_id)
        if distances:
            result = ids[distances.index(min(distances))]
        else:
            result = len(self.objects)
        return result

    def refresh_coordinates(self, object_id: int, box):
        xmin, ymin, xmax, ymax = box
        mean_x, mean_y = (xmax + xmin) / 2, (ymax + ymin) / 2
        self.objects[object_id] = (mean_x, mean_y)
