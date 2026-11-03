from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_FILE = ROOT / "yolov3.weights"
KERAS_MODEL_FILE = ROOT / "model.h5"
REID_MODEL_FILE = ROOT / "model_reid.pb"

IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416
CLASS_THRESHOLD = 0.7
NMS_THRESHOLD = 0.3

ANCHORS = [
    [116, 90, 156, 198, 373, 326],
    [30, 61, 62, 45, 59, 119],
    [10, 13, 16, 30, 33, 23],
]

TRACKER_MAX_COSINE_DISTANCE = 0.7
TRACKER_MAX_EUCLIDEAN_DISTANCE = 0.7
TRACKER_MAX_IOU_DISTANCE = 0.7
TRACKER_NN_BUDGET = None
TRACKER_MAX_AGE = 30
TRACKER_N_INIT = 3

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
NUM_CLASS = {i: name for i, name in enumerate(COCO_CLASSES)}
CLASS_TO_ID = {name: i for i, name in NUM_CLASS.items()}

LABEL_MAP = {
    "person": "blue", "bicycle": "red", "car": "yellow", "truck": "green",
    "motorbike": "white", "aeroplane": "white", "bus": "white",
    "train": "white", "boat": "white",
}

TRACK_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"
}
