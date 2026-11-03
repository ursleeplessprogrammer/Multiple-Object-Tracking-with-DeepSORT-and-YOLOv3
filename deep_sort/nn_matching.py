import numpy as np


def _pdist(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * np.dot(a, b.T), 0.0)


def _cosine_distance(a, b, data_is_normalized=False):
    a, b = np.asarray(a), np.asarray(b)
    if not data_is_normalized:
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return 1.0 - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    distances = _pdist(x, y)
    return np.maximum(0.0, np.min(distances, axis=0)) if distances.shape[0] else np.full((len(y),), np.inf)


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return np.min(distances, axis=0) if distances.shape[0] else np.full((len(y),), np.inf)


class NearestNeighborDistanceMetric:
    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._distance = _nn_euclidean_distance
        elif metric == "cosine":
            self._distance = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; choose 'euclidean' or 'cosine'.")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {target: self.samples.get(target, []) for target in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            samples = self.samples.get(target, [])
            if samples:
                cost_matrix[i, :] = self._distance(np.asarray(samples), np.asarray(features))
            else:
                cost_matrix[i, :] = np.inf
        return cost_matrix
