from unittest import result


class Global_result:
    result = []

class DetectionInfo:
    def __init__(self):
        self.bbox = []
        self.landmarks = []
        self.confidence = 0

    def set_bbox(self, index, value):
        self.bbox[index] = value

    def set_landmark(self, index, value):
        self.landmarks[index] = value

    def set_confidence(self, conf):
        self.confidence = conf