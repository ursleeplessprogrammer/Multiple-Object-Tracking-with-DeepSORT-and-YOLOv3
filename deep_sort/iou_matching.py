import numpy as np
from . import linear_assignment


def iou(bbox, candidates):
    bbox = np.asarray(bbox)
    candidates = np.asarray(candidates)
    if candidates.size == 0:
        return np.zeros((0,), dtype=float)
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]
    tl = np.maximum(bbox_tl, candidates_tl)
    br = np.minimum(bbox_br, candidates_br)
    wh = np.maximum(0.0, br - tl)
    area_intersection = wh[:, 0] * wh[:, 1]
    area_bbox = bbox[2] * bbox[3]
    area_candidates = candidates[:, 2] * candidates[:, 3]
    return area_intersection / np.maximum(area_bbox + area_candidates - area_intersection, 1e-12)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
    return cost_matrix
