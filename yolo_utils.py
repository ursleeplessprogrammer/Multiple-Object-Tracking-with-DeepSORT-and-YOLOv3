import cv2
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)
        self.objness = objness
        self.classes = np.asarray(classes if classes is not None else [], dtype=float)
        self.label = -1
        self.score = -1.0

    def get_label(self):
        if self.label == -1 and self.classes.size:
            self.label = int(np.argmax(self.classes))
        return self.label

    def get_score(self):
        if self.score == -1 and self.classes.size:
            self.score = float(self.classes[self.get_label()])
        return self.score


def sigmoid(x):
    x = np.clip(x, -80, 80)
    return 1.0 / (1.0 + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    netout = netout.reshape((grid_h, grid_w, 3, -1)).copy()
    boxes = []
    netout[..., :2] = sigmoid(netout[..., :2])
    netout[..., 4:] = sigmoid(netout[..., 4:])
    netout[..., 5:] *= netout[..., 4:5]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(3):
                objectness = float(netout[row, col, b, 4])
                if objectness <= obj_thresh:
                    continue
                x, y, w, h = netout[row, col, b, :4]
                x = (col + x) / grid_w
                y = (row + y) / grid_h
                w = anchors[2 * b] * np.exp(np.clip(w, -20, 20)) / net_w
                h = anchors[2 * b + 1] * np.exp(np.clip(h, -20, 20)) / net_h
                classes = netout[row, col, b, 5:]
                boxes.append(BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes))
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    scale = min(net_w / image_w, net_h / image_h)
    new_w, new_h = int(image_w * scale), int(image_h * scale)
    x_offset = (net_w - new_w) / (2.0 * net_w)
    y_offset = (net_h - new_h) / (2.0 * net_h)
    x_scale, y_scale = new_w / net_w, new_h / net_h
    for box in boxes:
        box.xmin = int((box.xmin - x_offset) / x_scale * image_w)
        box.xmax = int((box.xmax - x_offset) / x_scale * image_w)
        box.ymin = int((box.ymin - y_offset) / y_scale * image_h)
        box.ymax = int((box.ymax - y_offset) / y_scale * image_h)
        box.xmin = max(0, min(image_w - 1, box.xmin))
        box.xmax = max(0, min(image_w - 1, box.xmax))
        box.ymin = max(0, min(image_h - 1, box.ymin))
        box.ymax = max(0, min(image_h - 1, box.ymax))


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    return max(0, min(x2, x4) - max(x1, x3))


def bbox_iou(box1, box2):
    intersect = _interval_overlap((box1.xmin, box1.xmax), (box2.xmin, box2.xmax)) * _interval_overlap((box1.ymin, box1.ymax), (box2.ymin, box2.ymax))
    area1 = max(0, box1.xmax - box1.xmin) * max(0, box1.ymax - box1.ymin)
    area2 = max(0, box2.xmax - box2.xmin) * max(0, box2.ymax - box2.ymin)
    union = area1 + area2 - intersect
    return float(intersect / union) if union > 0 else 0.0


def do_nms(boxes, nms_thresh):
    if not boxes:
        return
    num_classes = len(boxes[0].classes)
    for class_id in range(num_classes):
        sorted_indices = np.argsort([-box.classes[class_id] for box in boxes])
        for i, index_i in enumerate(sorted_indices):
            if boxes[index_i].classes[class_id] <= 0:
                continue
            for index_j in sorted_indices[i + 1:]:
                if boxes[index_j].classes[class_id] > 0 and bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[class_id] = 0


def load_image_pixels(filename, shape):
    image = load_img(filename)
    width, height = image.size
    image = load_img(filename, target_size=shape)
    image = img_to_array(image).astype("float32") / 255.0
    return np.expand_dims(image, 0), width, height


def image_preprocess(image, target_size):
    target_h, target_w = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(scale * w), int(scale * h)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.full((target_h, target_w, 3), 128.0, dtype=np.float32)
    dx, dy = (target_w - new_w) // 2, (target_h - new_h) // 2
    padded[dy:dy + new_h, dx:dx + new_w] = resized
    return np.expand_dims(padded / 255.0, 0), w, h


def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = [], [], []
    for box in boxes:
        for i, label in enumerate(labels):
            if i < len(box.classes) and box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(label)
                v_scores.append(float(box.classes[i] * 100.0))
    return v_boxes, v_labels, v_scores


def draw_boxes(filename, v_boxes, v_labels, v_scores, label_map=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    label_map = label_map or {}
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()
    for box, label, score in zip(v_boxes, v_labels, v_scores):
        color = label_map.get(label, "red")
        ax.add_patch(Rectangle((box.xmin, box.ymin), box.xmax - box.xmin, box.ymax - box.ymin, fill=False, color=color))
        plt.text(box.xmin, box.ymin, f"{label} ({score:.3f})", color=color)
    plt.show()


def draw_bbox(image, bboxes, classes=None, show_label=True, show_confidence=True, Text_colors=(255, 255, 0), rectangle_colors="", tracking=False):
    import colorsys
    import random

    output = image.copy()
    classes = classes or {}
    num_classes = max(len(classes), 1)
    hsv_tuples = [(1.0 * i / num_classes, 1.0, 1.0) for i in range(num_classes)]
    colors = [tuple(int(channel * 255) for channel in colorsys.hsv_to_rgb(*hsv)) for hsv in hsv_tuples]
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    image_h, image_w = output.shape[:2]
    thickness = max(1, int(0.6 * (image_h + image_w) / 1000))
    font_scale = 0.75 * thickness
    for bbox in bboxes:
        if len(bbox) < 6:
            continue
        x1, y1, x2, y2 = np.asarray(bbox[:4], dtype=np.int32)
        score = float(bbox[4])
        class_id = int(bbox[5])
        class_name = classes.get(class_id, str(class_id))
        color = rectangle_colors if rectangle_colors != "" else colors[class_id % len(colors)]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness * 2)

        if show_label or tracking:
            if tracking:
                suffix = f" {int(score)}"
            else:
                suffix = f" {score:.2f}" if show_confidence else ""
            label = f"{class_name}{suffix}" if show_label else f"ID {int(score)}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, thickness
            )
            cv2.rectangle(
                output, (x1, y1), (x1 + text_width, y1 - text_height - baseline),
                color, thickness=cv2.FILLED,
            )
            cv2.putText(
                output, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                font_scale, Text_colors, thickness, lineType=cv2.LINE_AA,
            )
    return output
