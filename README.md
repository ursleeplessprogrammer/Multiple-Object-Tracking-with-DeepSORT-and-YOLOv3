# Multiple-Object-Tracking-with-DeepSORT-and-YOLOv3
Deep Learning Final Project that implemented DeepSORT algorithm and combined it with YOLOv3 for Multiple Object Tracking problem.

This repository contains the implementation of a Deep Learning project focused on real-time object detection and tracking. The project evaluates the performance of a standard YOLOv3 model and compares it with a Modified YOLOv3 integrated with DeepSORT (Deep Learning-based Object Tracking).

## 📌 About the Project
This project was developed as final project for Deep Leaning Courses. 
The primary goal of this project is to create a robust system capable of not only detecting objects in a frame but also tracking them consistently across multiple frames in a video sequence. 
I utilized the KITTI-Road-Segmentation dataset for theory and training context, focusing on urban environments.

## 🚀 Key Features
- Object Detection: Implemented using YOLOv3 (You Only Look Once v3) with a Darknet-53 backbone.

- Backbone Modification: Added 3 additional layers to the YOLOv3 backbone to enhance the extraction of abstract and complex features, improving detection reliability without causing overfitting.

- Object Tracking: Integrated DeepSORT, which utilizes:
  - Kalman Filter: To predict the future position of objects.
  - Hungarian Algorithm: For data association and matching.
  - Deep Association Metric: Using cosine distance to identify objects based on unique visual features.

## 📊 Methodology
1. Pre-processing: Image resizing (416x416), normalization, and padding.
   
2. Detection: YOLOv3 generates bounding boxes and class probabilities.
   
3. Tracking (DeepSORT):
- Feature extraction from detected bounding boxes.
- Motion prediction via Kalman Filter.
- Matching detections to existing tracks using appearance descriptors and IoU (Intersection over Union).

## 📊 Performance Analysis & Key Findings
Based on testing across 30 diverse video sequences, we identified several critical factors affecting model performance:

1. Detection Capabilities
- Multi-Object Detection: The model successfully detects multiple object classes (cars, buses, trucks, motorbikes, and people) simultaneously in high-density traffic.
- Distance Sensitivity: The model can detect distant objects (small scale), but performance drops if the object size is too minute for the input resolution.
- Training Impact: There is a significant accuracy boost after the training process. Before training, the model struggled with precise bounding box placement and missed several obvious targets.

2. Environmental & Hardware Challenges
- Lighting Conditions (Day vs. Night): The model remains functional at night, but the detection range is shorter compared to daylight. Glare from headlights and low-contrast environments can degrade performance.
- Occlusion: Tracking consistency is challenged when objects are partially or fully hidden by others. If an object disappears briefly, DeepSORT may occasionally assign a new ID upon its reappearance.
- Crowded Scenes: When similar objects (e.g., two motorbikes of the same color) move closely together, the model sometimes struggles to differentiate them.
- Camera Point of View (POV): Forward-facing dashcams yield much better results than rear-facing ones. Rear-facing cameras often result in "sunken" bounding boxes that fall outside the actual vehicle range.

3. Error Analysis (Loss Factors)
- Localization Error: Occurs when the predicted bounding box does not align perfectly with the ground truth, often due to varying object scales or grid cell limitations.
- Classification Loss: Occasional misclassification (e.g., a car being labeled as a truck) occurs in complex visual scenes.
- Objectness Loss: High loss is observed when the model fails to determine if a bounding box actually contains an object, particularly with very small or overlapping targets.

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

## 🛠️ Requirements
- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
