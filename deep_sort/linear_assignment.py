import numpy as np
from scipy.optimize import linear_sum_assignment
from . import kalman_filter

INFTY_COST = 1e5


def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    if not track_indices or not detection_indices:
        return [], list(track_indices), list(detection_indices)
    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix = np.asarray(cost_matrix)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_detections = [], [], []
    assigned_rows, assigned_cols = set(row_indices), set(col_indices)
    for row, col in zip(row_indices, col_indices):
        track_idx, detection_idx = track_indices[row], detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    unmatched_tracks.extend(track_indices[i] for i in range(len(track_indices)) if i not in assigned_rows)
    unmatched_detections.extend(detection_indices[i] for i in range(len(detection_indices)) if i not in assigned_cols)
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    unmatched_detections = list(detection_indices)
    matches = []
    for level in range(cascade_depth):
        if not unmatched_detections:
            break
        track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if not track_indices_l:
            continue
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections, track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - {k for k, _ in matches})
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices, gated_cost=INFTY_COST, only_position=False):
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        distances = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, distances > gating_threshold] = gated_cost
    return cost_matrix
