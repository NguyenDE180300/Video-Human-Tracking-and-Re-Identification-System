# Advanced Person Tracking Notebooks: DeepSORT vs. Re-ID

This repository contains two Jupyter Notebooks that implement and compare two powerful methods for real-time person tracking in videos. Both methods use a YOLO (You Only Look Once) model for initial object detection.

The primary goal is to provide a hands-on comparison between a motion-based tracking algorithm (DeepSORT) and an appearance-based re-identification (Re-ID) system.

---

## Notebook Descriptions

### 1. `trainingDeepsort.ipynb`: Person Tracking with YOLO + DeepSORT

> **Note:** This notebook implements tracking using a pre-existing DeepSORT algorithm; it does not train the DeepSORT model itself.

- **Core Technologies:** YOLO + DeepSORT  
- **Methodology:**
  1. **Detection:** A YOLO model detects all persons in each frame.
  2. **Tracking:** DeepSORT uses a **Kalman Filter** and a small CNN for appearance matching to assign consistent IDs.
- **Key Characteristics:**
  - Fast and computationally efficient.
  - Susceptible to ID switches in crowded or occluded scenes.

---

### 2. `TrainingYoloCNN.ipynb`: Person Tracking with YOLO + CNN-based Re-identification

> **Note:** This notebook implements a re-identification pipeline using pre-trained YOLO and ResNet models. It does not train these models from scratch.

- **Core Technologies:** YOLO + ResNet  
- **Methodology:**
  1. **Detection:** YOLO detects all persons in each frame.
  2. **Feature Extraction:** Detected regions are passed through **ResNet** to generate a feature vector (appearance embedding).
  3. **Re-identification:** Uses **cosine similarity** to match appearance with known identities.
- **Key Characteristics:**
  - Robust to occlusion and re-appearance.
  - Slower due to heavy feature extraction per person per frame.

---

## Project Structure
```
your-project-folder/
├── trainingDeepsort.ipynb
├── TrainingYoloCNN.ipynb
├── requirements.txt
├── yolov8n.pt # Downloaded YOLOv8 weights
├── yolo12s.pt # Or other YOLO model weights
├── videos/
│ └── input_video.mp4 # Your input video files
└── outputs/
├── output_deepsort.mp4 # From DeepSORT notebook
└── output_reid.mp4 # From Re-ID notebook
```
---

## Requirements

Install dependencies:
```
pip install -r requirements.txt
```
Models and video files must be downloaded or placed manually in their corresponding folders.

## Authors & License
Created by Hoang Trung Nguyen — for educational and research purposes.
Feel free to fork and adapt.
