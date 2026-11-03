import argparse
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import (
    ANCHORS, CLASS_THRESHOLD, COCO_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH,
    KERAS_MODEL_FILE, REID_MODEL_FILE, TRACK_CLASSES,
    TRACKER_MAX_AGE, TRACKER_MAX_COSINE_DISTANCE, TRACKER_MAX_IOU_DISTANCE,
    TRACKER_N_INIT, TRACKER_NN_BUDGET,
)
from deep_sort import Detection, NearestNeighborDistanceMetric, Tracker, create_box_encoder
from yolo_utils import correct_yolo_boxes, decode_netout, do_nms, draw_bbox, get_boxes, image_preprocess


def detect_frame(model, frame, threshold=CLASS_THRESHOLD):
    image_data, image_w, image_h = image_preprocess(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    predictions = model.predict(image_data, verbose=0)
    boxes = []
    for i, prediction in enumerate(predictions):
        boxes.extend(decode_netout(prediction[0], ANCHORS[i], threshold, IMAGE_HEIGHT, IMAGE_WIDTH))
    correct_yolo_boxes(boxes, image_h, image_w, IMAGE_HEIGHT, IMAGE_WIDTH)
    do_nms(boxes, 0.3)
    v_boxes, v_labels, v_scores = get_boxes(boxes, COCO_CLASSES, threshold)
    return [
        (box, label, score)
        for box, label, score in zip(v_boxes, v_labels, v_scores)
        if label in TRACK_CLASSES
    ]


def object_tracking(video_path, output_path, model_path=KERAS_MODEL_FILE,
                    encoder_path=REID_MODEL_FILE, display=True):
    model = load_model(str(model_path), compile=False)
    encoder = create_box_encoder(str(encoder_path), batch_size=1)
    metric = NearestNeighborDistanceMetric("cosine", TRACKER_MAX_COSINE_DISTANCE, TRACKER_NN_BUDGET)
    tracker = Tracker(metric, TRACKER_MAX_IOU_DISTANCE, TRACKER_MAX_AGE, TRACKER_N_INIT)

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    frame_times, total_times = [], []
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frame_start = time.perf_counter()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections_raw = detect_frame(model, rgb_frame)
            detect_end = time.perf_counter()

            boxes, names, scores = [], [], []
            for box, label, score in detections_raw:
                x1, y1, x2, y2 = box.xmin, box.ymin, box.xmax, box.ymax
                boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
                names.append(label)
                scores.append(score)
            boxes = np.asarray(boxes, dtype=float).reshape((-1, 4))
            names = np.asarray(names)
            scores = np.asarray(scores, dtype=float)
            features = encoder(frame, boxes) if len(boxes) else np.empty((0, 0), dtype=np.float32)
            current_detections = [
                Detection(box, score, class_name, feature)
                for box, score, class_name, feature in zip(boxes, scores, names, features)
            ]

            tracker.predict()
            tracker.update(current_detections)

            tracked_bboxes = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 5:
                    continue
                x1, y1, x2, y2 = track.to_tlbr()
                class_name = track.get_class()
                class_id = COCO_CLASSES.index(class_name) if class_name in COCO_CLASSES else -1
                tracked_bboxes.append([x1, y1, x2, y2, track.track_id, class_id])

            output_frame = draw_bbox(
                frame, tracked_bboxes, classes={i: name for i, name in enumerate(COCO_CLASSES)},
                show_label=True, show_confidence=False, rectangle_colors=(0, 255, 0), tracking=True,
            )
            finished = time.perf_counter()
            frame_times.append(detect_end - frame_start)
            total_times.append(finished - frame_start)
            frame_times, total_times = frame_times[-20:], total_times[-20:]
            detection_ms = 1000.0 * sum(frame_times) / len(frame_times)
            detection_fps = 1000.0 / detection_ms if detection_ms else 0.0
            total_fps = len(total_times) / sum(total_times) if sum(total_times) else 0.0
            cv2.putText(output_frame, f"Time: {detection_ms:.1f} ms | Detection FPS: {detection_fps:.1f} | Total FPS: {total_fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 2)
            writer.write(output_frame)
            if display:
                cv2.imshow("YOLOv3 + DeepSORT", output_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        writer.release()
        video.release()
        if display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--model", default=str(KERAS_MODEL_FILE))
    parser.add_argument("--encoder", default=str(REID_MODEL_FILE))
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    object_tracking(args.video, args.output, args.model, args.encoder, not args.no_display)


if __name__ == "__main__":
    main()
