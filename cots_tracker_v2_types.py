def box_area(x0, y0, x1, y1):
    return (x1 - x0 + 1) * (y1 - y0 + 1)


class Detection():
    def __init__(self, class_id, score, x0, y0, x1, y1):
        self.class_id = class_id
        self.score = score
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def __repr__(self):
        return (f'Class {self.class_id}, score {self.score}, '
                f'box ({self.x0}, {self.y0}, {self.x1}, {self.y1})')

    def area(self):
        return box_area(self.x0, self.y0, self.x1, self.y1)

    def iou(self, other):
        overlap_x0 = max(self.x0, other.x0)
        overlap_y0 = max(self.y0, other.y0)
        overlap_x1 = min(self.x1, other.x1)
        overlap_y1 = min(self.y1, other.y1)
        if overlap_x0 < overlap_x1 and overlap_y0 < overlap_y1:
            overlap_area = box_area(overlap_x0, overlap_y0, overlap_x1,
                                    overlap_y1)
            return overlap_area / (self.area() + other.area() - overlap_area)
        else:
            return 0
        