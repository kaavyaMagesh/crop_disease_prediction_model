Crop Disease Detection using Computer Vision
Overview

This project implements a computer vision–based crop disease detection system using deep learning. The model classifies crop leaf images into healthy and diseased categories and identifies the specific disease type. The solution enables early disease detection, reduces crop loss, and is suitable for deployment in real-world agricultural environments.

The model is designed to be lightweight and deployable on mobile devices, making it appropriate for low-connectivity and resource-constrained settings.

Objectives

Detect crop diseases from leaf images captured via mobile cameras

Classify multiple crop–disease combinations

Enable fast and reliable inference for field usage

Integrate with a farmer advisory mobile application

Model Architecture

Base model: MobileNetV2 (transfer learning)

Framework: TensorFlow / Keras

Methodology:

Initialized with ImageNet pretrained weights

Custom classification head added

Selective fine-tuning of deeper layers

MobileNetV2 was chosen due to its low latency, small memory footprint, and suitability for edge and mobile deployment.

Dataset

The model was trained using publicly available, labeled plant disease image datasets containing multiple crop species and disease classes. These datasets include field-like variations such as lighting changes, background noise, and partial leaf damage.

Primary sources:

Kaggle multi-class crop disease image datasets

Open-source plant disease image repositories

Data Preprocessing and Augmentation

Images resized to 224 × 224 pixels

Pixel normalization to [0,1] range

Data augmentation techniques:

Random rotations

Horizontal and vertical flips

Zoom and brightness adjustments

These steps improve generalization to real-world farm conditions.

Training Details

Loss function: Categorical Cross-Entropy

Optimizer: Adam

Batch size: 32

Epochs: 20–30

Evaluation metrics:

Accuracy

Precision

Recall

Confusion matrix

Results

Achieved approximately 90% validation accuracy (varies by disease class)

Robust performance under varying illumination and background conditions

Suitable for real-time inference on mobile devices

Inference Pipeline

Farmer captures a crop leaf image using a mobile device

Image is resized and normalized

Image is passed through the trained CNN model

Model outputs disease class and confidence score

Output is mapped to disease description and treatment recommendations

Deployment

Model converted to TensorFlow Lite format

Integrated into a Flutter-based mobile application

Supports offline inference for low-connectivity regions

Integration in Smart India Hackathon Solution

This crop disease detection module integrates with:

Crop recommendation engine

Weather and humidity data sources

Sustainability and advisory modules

The combined multi-modal system enables early disease warnings by correlating image-based detection with environmental conditions.

Future Improvements

Region-specific disease fine-tuning

Disease severity estimation (mild, moderate, severe)

Multi-leaf and whole-plant analysis

Federated learning for decentralized model improvement

License

This project is intended for academic, research, and hackathon use.
