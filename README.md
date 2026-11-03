# Multiple-Object-Tracking-with-DeepSORT-and-YOLOv3

<p align="center">
    <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Framework-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
    <img src="https://img.shields.io/badge/API-Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
    <img src="https://img.shields.io/badge/Computer%20Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
    <img src="https://img.shields.io/badge/Object%20Detection-YOLOv3-00FFFF?style=for-the-badge&logoColor=black" />
    <img src="https://img.shields.io/badge/Object%20Tracking-DeepSORT-2E8B57?style=for-the-badge&logoColor=white" />
    <img src="https://img.shields.io/badge/Library-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
    <img src="https://img.shields.io/badge/Library-SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" />
    <img src="https://img.shields.io/badge/IDE-Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white" />
</p>

Deep Learning Final Project that implements the **DeepSORT algorithm** and combines it with **YOLOv3** for the Multiple Object Tracking (MOT) problem.

This repository contains an implementation of a deep-learning-based object detection and tracking system. It covers a standard YOLOv3 detection workflow and a tracking workflow that combines YOLOv3 with DeepSORT. The detector identifies objects in each video frame, while DeepSORT associates detections across frames to maintain object identities.

---

## 📌 Overview

This project was developed as a final project for a Deep Learning course. Its primary goal is to build a system that can detect multiple objects in a video and, when tracking is enabled, follow those objects consistently across consecutive frames.

The project uses the **KITTI Road Segmentation dataset** as its road-scene dataset context. The report discusses vehicle-mounted camera imagery and road environments relevant to traffic understanding and autonomous driving.

Two workflows are provided:

1. **YOLOv3 Object Detection** — detects and classifies objects in video frames.
2. **YOLOv3 + DeepSORT Object Tracking** — uses the detector's bounding boxes and class predictions, then associates detections over time to assign persistent track IDs.

## 🚀 Key Features

### Object Detection
- Implemented using **YOLOv3 (You Only Look Once v3)** with a Darknet-53 backbone.
- Uses multi-scale detection to identify objects at different sizes.
- Decodes bounding boxes, objectness scores, and class probabilities from the detector output.
- Applies confidence filtering and Non-Maximum Suppression (NMS) to remove duplicate detections.

### Backbone Modification
- The report describes adding **three additional layers** to the YOLOv3 backbone.
- The modification is intended to improve the extraction of abstract and complex image features.
- Both the original and modified YOLOv3 architecture builders are included in `yolov3_model.py`.

### Object Tracking
DeepSORT is integrated with YOLOv3 and utilizes:
- **Kalman Filter:** predicts the future position and state of tracked objects.
- **Hungarian Algorithm:** performs data association between detections and existing tracks.
- **Deep Association Metric:** compares visual appearance descriptors using cosine distance.
- **IoU Matching:** uses bounding-box overlap as a geometric matching measure.
- **Track Management:** creates, updates, confirms, and removes tracks according to their state and age.

### Video Processing
- Processes video frame by frame.
- Supports detection-only and detection-plus-tracking workflows.
- Draws bounding boxes, class labels, and (for tracking) track IDs on the output video.

## 📊 Methodology

The overall process is divided into the following stages.

### 1. Dataset and Annotation Preparation
- Reads image and annotation information from a COCO-style JSON structure.
- Converts annotation information into YOLO-format label files.
- Generates a dataset YAML configuration containing the image paths and class names.
- The report's example configuration uses the KITTI image directory. Update paths and dataset splits to match the local dataset.

### 2. Image Pre-processing
- Resizes input frames to **416 × 416**.
- Normalizes image pixel values before inference.
- Retains the original frame dimensions so predicted bounding boxes can be mapped back to the source image.

### 3. Object Detection with YOLOv3
For each frame:
1. The preprocessed image is passed to YOLOv3.
2. The model predicts bounding boxes, objectness scores, and class probabilities at multiple scales.
3. The output is decoded into bounding-box coordinates and class predictions.
4. Coordinates are corrected to the original frame dimensions.
5. Confidence filtering and NMS are applied to remove low-confidence and overlapping duplicate detections.

### 4. Object Tracking with DeepSORT
For the tracking workflow:
1. The selected YOLOv3 detections are converted into DeepSORT detection objects.
2. Image crops corresponding to the detections are passed to the appearance encoder.
3. The encoder generates visual feature descriptors.
4. The Kalman Filter predicts the positions of existing tracks.
5. DeepSORT associates new detections with existing tracks using appearance and motion information, with IoU-based matching as an additional matching stage.
6. Track states are updated, new tracks are initiated when needed, and old tracks are removed according to tracker settings.

---

## 📊 Performance Analysis & Key Findings

The report presents a 30-epoch training experiment and qualitative observations from video testing. The following results and findings are based on the report.

### 1. Detection Capabilities

- **Multi-Object Detection:** The model detects multiple road-scene object classes, including cars, buses, trucks, motorbikes, and people.
- **Distance Sensitivity:** Distant and small objects can be detected, but performance decreases when an object's size becomes too small relative to the input resolution.
- **Training Impact:** The report shows improved detection metrics over the course of training. Before training, the model had difficulty placing bounding boxes accurately and missed some visible targets.

#### Training Metrics

The following values are transcribed from the 30-epoch experiment:

| Epoch | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---:|---:|---:|---:|---:|
| 1 | 0.65078 | 0.56897 | 0.556883 | 0.35136 |
| 5 | 0.63708 | 0.52929 | 0.457347 | 0.26334 |
| 10 | 0.68648 | 0.574357 | 0.54893 | 0.319465 |
| 15 | 0.76076 | 0.64454 | 0.64036 | 0.38829 |
| 20 | 0.80482 | 0.69247 | 0.69017 | 0.43023 |
| 25 | 0.85193 | 0.76885 | 0.74187 | 0.47931 |
| 30 | 0.89287 | 0.80248 | 0.77916 | 0.53138 |

These are object-detection metrics. The report does not provide standard multi-object tracking metrics such as MOTA, IDF1, or HOTA, so the values above should not be interpreted as a quantitative evaluation of tracking identity consistency.

### 2. Environmental & Hardware Challenges

- **Lighting Conditions (Day vs. Night):** The report observes that detection remains possible at night, but the detection range is shorter than in daylight. Headlight glare and low-contrast environments can reduce detection reliability.
- **Occlusion:** Tracking consistency is challenged when objects are partially or fully hidden. If an object disappears for a period, DeepSORT may assign a new ID when it reappears.
- **Crowded Scenes:** When visually similar objects (for example, two motorbikes with similar appearance) move close together, distinguishing them can be difficult.
- **Camera Point of View (POV):** The report observes better results with forward-facing dashcam footage than with rear-facing footage. In some rear-facing footage, bounding boxes may fall outside the actual vehicle boundaries.

### 3. Error Analysis (Loss Factors)

The report discusses the following detection error factors:

- **Localization Error:** Occurs when a predicted bounding box does not align accurately with the ground-truth object. This can be affected by object scale and the detector's prediction grid.
- **Classification Loss:** Misclassification can occur in visually complex scenes, such as a car being classified as a truck.
- **Objectness Loss:** The model may have difficulty deciding whether a predicted bounding box contains an object, especially for very small or overlapping targets.

### 4. Tracking Observations

DeepSORT adds persistent identities to detections and can help maintain object identities as objects move through the scene or cross paths. Its performance is still affected by long occlusions, similar-looking objects, detector errors, and changes in visual appearance.

The report's tracking evaluation is qualitative. No numerical tracking benchmark is stated, so claims of a measured improvement in tracking accuracy or real-time speed should not be inferred from the detection metrics alone.

---

## 📁 Files

The repository is organized by responsibility so that model construction, detection utilities, dataset preparation, video processing, and tracking logic are kept separate.

### Main Python files

| File | Description |
|---|---|
| `build_model.py` | Builds the original or modified YOLOv3 model, loads Darknet weights, and saves the resulting Keras model. |
| `config.py` | Stores image dimensions, anchors, class names, thresholds, tracking classes, and DeepSORT configuration. |
| `dataset.py` | Parses COCO-style annotation JSON and prepares YOLO-format labels and dataset configuration. |
| `train_dataset.py` | Runs the dataset preparation and training-command workflow. The external training script must be available separately. |
| `video_detection.py` | Runs YOLOv3 video detection without persistent tracking IDs. |
| `video_tracking.py` | Runs YOLOv3 detection with DeepSORT tracking and writes the annotated output video. |
| `weights.py` | Reads Darknet weights and includes a helper for downloading the YOLOv3 weights file. |
| `yolov3_model.py` | Defines the original and modified YOLOv3 Keras architectures. |
| `yolo_utils.py` | Provides image preprocessing, output decoding, bounding-box correction, IoU, NMS, and drawing utilities. |

### DeepSORT package

| File | Description |
|---|---|
| `deep_sort/detection.py` | Represents detections with bounding boxes, confidence values, classes, and appearance features. |
| `deep_sort/encoder.py` | Loads the TensorFlow frozen-graph encoder and extracts appearance descriptors. |
| `deep_sort/generate_detections.py` | Extracts image patches and generates appearance features for detections. |
| `deep_sort/iou_matching.py` | Calculates IoU-based matching costs. |
| `deep_sort/kalman_filter.py` | Implements the Kalman Filter used for track prediction and state updates. |
| `deep_sort/linear_assignment.py` | Implements assignment and matching-cascade utilities. |
| `deep_sort/nn_matching.py` | Calculates appearance distances and manages feature samples for tracks. |
| `deep_sort/preprocessing.py` | Provides detection-level non-maximum suppression utilities. |
| `deep_sort/track.py` | Defines an individual track and its lifecycle/state. |
| `deep_sort/tracker.py` | Coordinates prediction, detection association, track updates, and track creation/removal. |

### Repository layout

```text
Multiple-Object-Tracking-with-DeepSORT-and-YOLOv3/
│
├── build_model.py
├── config.py
├── dataset.py
├── train_dataset.py
├── video_detection.py
├── video_tracking.py
├── weights.py
├── yolov3_model.py
├── yolo_utils.py
├── requirements.txt
├── README.md
│
└── deep_sort/
    ├── __init__.py
    ├── detection.py
    ├── encoder.py
    ├── generate_detections.py
    ├── iou_matching.py
    ├── kalman_filter.py
    ├── linear_assignment.py
    ├── nn_matching.py
    ├── preprocessing.py
    ├── track.py
    └── tracker.py
```

## Files

- `yolov3_model.py`: original and modified YOLOv3 Keras architectures.
- `yolo_utils.py`: bounding boxes, YOLO output decoding, preprocessing, NMS, and drawing.
- `weights.py`: Darknet weights loading and download helper.
- `dataset.py`: annotation JSON parsing and dataset YAML generation.
- `train_dataset.py`: dataset configuration and training command wrapper.
- `video_detection.py`: YOLOv3 video detection without tracking.
- `video_tracking.py`: modified YOLOv3 detection with DeepSORT tracking.
- `deep_sort/`: DeepSORT detection encoder, Kalman filter, matching, track management, and preprocessing.


# 🧠 Model Files and Weights
The source code does not include the model binaries, dataset, or sample videos. Prepare the required assets before running the scripts.
| File | Purpose |
|---|---|
| `yolov3.weights` | Darknet pretrained weights used to initialize the YOLOv3 architecture. |
| `model.h5` | Keras YOLOv3 detector used by the video detection and tracking scripts. Use the trained detector you intend to evaluate. |
| `model_reid.pb` | TensorFlow frozen graph used by DeepSORT to extract appearance/re-identification features. |

**Important**: model.h5 and model_reid.pb serve different purposes and are not interchangeable. model.h5 performs object detection, while model_reid.pb extracts appearance descriptors for DeepSORT. The report's code does not include these binary assets; they must be obtained or generated separately.

Place the files in the project root or provide their paths through command-line arguments.

## 🛠️ Requirements
- Python 3.x (Python 3.9–3.11 is recommended for the listed TensorFlow version)
- TensorFlow 2.15.x / Keras
- OpenCV
- NumPy
- Scipy
- Matplotlib
The dependency versions are defined in requirements.txt. Compatibility with other TensorFlow or Python versions has not been verified.


## ⚙️ Configuration

Common settings are centralized in `config.py`.

| Setting | Default | Description |
|---|---:|---|
| `IMAGE_WIDTH` | `416` | Detector input width. |
| `IMAGE_HEIGHT` | `416` | Detector input height. |
| `CLASS_THRESHOLD` | `0.7` | Confidence threshold used when filtering detections. |
| `NMS_THRESHOLD` | `0.3` | Non-Maximum Suppression threshold. |
| `TRACKER_MAX_COSINE_DISTANCE` | `0.7` | Appearance-distance threshold for matching. |
| `TRACKER_MAX_IOU_DISTANCE` | `0.7` | IoU matching distance threshold. |
| `TRACKER_MAX_AGE` | `30` | Maximum age of a track without a successful update. |
| `TRACKER_N_INIT` | `3` | Number of hits required to confirm a track. |


## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Build the original YOLOv3 model

```bash
python build_model.py --download --weights yolov3.weights --output model.h5
```

## Build the modified YOLOv3 model

```bash
python build_model.py --modified --weights yolov3.weights --output model.h5
```

## Run video detection

```bash
python video_detection.py input.mp4 output_detection.mp4 --model model.h5
```

## Run YOLOv3 + DeepSORT

Place the trained Keras detector at `model.h5` and the frozen TensorFlow re-identification graph at `model_reid.pb`, or pass their paths explicitly:

```bash
python video_tracking.py input.mp4 output_tracking.mp4 --model model.h5 --encoder model_reid.pb
```

## ⚠️ Limitations and Reproducibility Notes
- The dataset, pretrained Darknet weights, trained Keras detector, DeepSORT re-identification graph, and sample videos are not bundled with the source code.
- Dataset paths shown in the report are environment-specific. Update them to match your local directory structure.
- The training command relies on an external YOLO training script that is not included in this repository.
- The DeepSORT encoder requires a compatible TensorFlow frozen graph. A YOLOv3 .h5 model cannot replace it.
- Video processing speed depends on the input resolution, video, hardware, and model files. Real-time performance should be benchmarked on the target machine.
- The Python files are an organized reconstruction of the code shown in the report. Some formatting, indentation, and incomplete snippets were repaired to make the modules usable.
- Passing Python syntax checks does not guarantee successful inference; the required model assets and compatible runtime environment are still necessary.

  
## 🏁 Conclusion
Based on the experiments, the combination of YOLOv3 and DeepSORT provides several advantages over standalone detection:

1. Accurate Detection and Tracking:
   While YOLOv3 is excellent at rapid detection, DeepSORT ensures that object identities are maintained consistently across frames, resulting in stable tracking.

2. Superior Object Identification:
   DeepSORT can distinguish between similar-looking objects and handle complex scenarios like partial occlusions or objects crossing paths by using unique feature descriptors.

3. Optimized Processing Speed:
   Despite adding a tracking layer to the workflow, the system maintains a high processing speed, making it suitable for real-time applications.

4. Wide Applicability:
   This hybrid approach is highly effective for surveillance systems, traffic analysis, and autonomous vehicle navigation.


## 📚 References
- Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv:1804.02767.
- Wojke, N., Bewley, A., & Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. 2017 IEEE International Conference on Image Processing (ICIP).
- KITTI Vision Benchmark Suite. KITTI Road/Lane Detection and Road Segmentation Benchmark.
- Aqqa, M., et al. (2019). Understanding How Video Quality Affects Object Detection Algorithms.
- Gao, J., et al. (2020). Deep Convolutional Neural Networks for Image-Based Detection.

---
## 📄 License

Please refer to the repository for applicable licensing information.

## 👨‍💻 Author

**ursleeplessprogrammer**

