# Multiple-Object-Tracking-with-DeepSORT-and-YOLOv3
Deep Learning Final Project that implemented DeepSORT algorithm and combined it with YOLOv3 for Multiple Object Tracking problem.

This repository contains the implementation of a Deep Learning project focused on real-time object detection and tracking. The project evaluates the performance of a standard YOLOv3 model and compares it with a Modified YOLOv3 integrated with DeepSORT (Deep Learning-based Object Tracking).

## 📌 About the Project
The primary goal of this project is to create a robust system capable of not only detecting objects in a frame but also tracking them consistently across multiple frames in a video sequence. We utilized the KITTI-Road-Segmentation dataset for theory and training context, focusing on urban environments.

## 🚀 Key Features
Object Detection: Implemented using YOLOv3 (You Only Look Once v3) with a Darknet-53 backbone.
Backbone Modification: Added 3 additional layers to the YOLOv3 backbone to enhance the extraction of abstract and complex features, improving detection reliability without causing overfitting.
Object Tracking: Integrated DeepSORT, which utilizes:
Kalman Filter: To predict the future position of objects.
Hungarian Algorithm: For data association and matching.
Deep Association Metric: Using cosine distance to identify objects based on unique visual features.

## 📊 Methodology
Pre-processing: Image resizing (416x416), normalization, and padding.
Detection: YOLOv3 generates bounding boxes and class probabilities.
Tracking (DeepSORT):
Feature extraction from detected bounding boxes.
Motion prediction via Kalman Filter.
Matching detections to existing tracks using appearance descriptors and IoU (Intersection over Union).

## 🏁 Conclusion
Based on the experiments, the combination of YOLOv3 and DeepSORT provides several advantages over standalone detection:

Accurate Detection and Tracking: 
While YOLOv3 is excellent at rapid detection, DeepSORT ensures that object identities are maintained consistently across frames, resulting in stable tracking.

Superior Object Identification: 
DeepSORT can distinguish between similar-looking objects and handle complex scenarios like partial occlusions or objects crossing paths by using unique feature descriptors.

Optimized Processing Speed: 
Despite adding a tracking layer to the workflow, the system maintains a high processing speed, making it suitable for real-time applications.
Wide Applicability: This hybrid approach is highly effective for surveillance systems, traffic analysis, and autonomous vehicle navigation.

## 🛠️ Requirements
- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
