import os
from pathlib import Path

import cv2
import numpy as np


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    mot_dir = Path(mot_dir)
    detection_dir = Path(detection_dir) if detection_dir is not None else mot_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sequence_dir in sorted(path for path in mot_dir.iterdir() if path.is_dir()):
        sequence = sequence_dir.name
        image_dir = sequence_dir / "img1"
        detection_file = detection_dir / sequence / "det" / "det.txt"
        if not detection_file.exists():
            continue
        image_filenames = {
            int(Path(filename).stem): image_dir / filename
            for filename in os.listdir(image_dir)
            if Path(filename).stem.isdigit()
        }
        detections_in = np.loadtxt(detection_file, delimiter=",")
        detections_in = np.atleast_2d(detections_in)
        detections_out = []
        frame_indices = detections_in[:, 0].astype(int)
        for frame_idx in range(int(frame_indices.min()), int(frame_indices.max()) + 1):
            rows = detections_in[frame_indices == frame_idx]
            image_path = image_filenames.get(frame_idx)
            if image_path is None:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            features = encoder(image, rows[:, 2:6].copy())
            detections_out.extend(np.r_[row, feature] for row, feature in zip(rows, features))
        np.save(output_dir / f"{sequence}.npy", np.asarray(detections_out), allow_pickle=False)
