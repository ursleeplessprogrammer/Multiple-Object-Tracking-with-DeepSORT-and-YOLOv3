import argparse
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import ANCHORS, CLASS_THRESHOLD, COCO_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, KERAS_MODEL_FILE, NMS_THRESHOLD
from yolo_utils import correct_yolo_boxes, decode_netout, do_nms, draw_bbox, get_boxes, image_preprocess


def detect_frame(model, frame, threshold=CLASS_THRESHOLD, nms_threshold=NMS_THRESHOLD):
    image_data, image_w, image_h = image_preprocess(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    predictions = model.predict(image_data, verbose=0)
    boxes = []
    for i, prediction in enumerate(predictions):
        boxes.extend(decode_netout(prediction[0], ANCHORS[i], threshold, IMAGE_HEIGHT, IMAGE_WIDTH))
    correct_yolo_boxes(boxes, image_h, image_w, IMAGE_HEIGHT, IMAGE_WIDTH)
    do_nms(boxes, nms_threshold)
    v_boxes, v_labels, v_scores = get_boxes(boxes, COCO_CLASSES, threshold)
    detections = []
    for box, label, score in zip(v_boxes, v_labels, v_scores):
        class_id = COCO_CLASSES.index(label)
        detections.append((
            int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax),
            class_id, float(score / 100.0),
        ))
    return detections


def object_detection(video_path, output_path, model_path=KERAS_MODEL_FILE, display=True):
    model = load_model(str(model_path), compile=False)
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    frame_times = []
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            start = time.perf_counter()
            detections = detect_frame(model, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for x1, y1, x2, y2, class_id, confidence in detections:
                label = f"{COCO_CLASSES[class_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elapsed = time.perf_counter() - start
            frame_times.append(elapsed)
            frame_times = frame_times[-20:]
            avg_ms = 1000.0 * sum(frame_times) / len(frame_times)
            current_fps = 1000.0 / avg_ms if avg_ms else 0.0
            cv2.putText(frame, f"Time: {avg_ms:.1f} ms | FPS: {current_fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            writer.write(frame)
            if display:
                cv2.imshow("YOLOv3 Detection", frame)
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
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    object_detection(args.video, args.output, args.model, not args.no_display)


if __name__ == "__main__":
    main()
