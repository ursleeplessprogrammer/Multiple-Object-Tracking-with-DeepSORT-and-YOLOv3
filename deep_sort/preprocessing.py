import numpy as np


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=float)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    area = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    idxs = np.argsort(np.asarray(scores)) if scores is not None else np.argsort(y2)
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(int(i))
        if last == 0:
            break
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = (w * h) / np.maximum(area[idxs[:last]], 1e-12)
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0])))
    return pick
