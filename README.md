# Brand-Model Classification and Bolt Detection in Pre-Disassembly Inspection of EV Battery Packs
1. Description
This repository contains the official implementation of the research paper: "Mamba-CSDS: A Robust Dual-Stream Visual Sensing System for Automated EV Battery Disassembly under Industrial Constraints".
The project addresses the critical challenge of automated lithium-ion battery (LIB) recycling. We propose an improved Mamba-CSDS model that concurrently handles two heterogeneous tasks within a single framework:
•	Global Task: Brand and model identification of battery modules.
•	Local Task: High-precision detection of fastening bolts.
By leveraging a backbone and a Dual-Stream head architecture, the model overcomes the limitations of traditional CNNs in capturing long-range dependencies and avoids the quadratic computational complexity of Transformers.

2. Dataset Information
The experiments were validated on a purpose-built industrial dataset captured in an authentic recycling warehouse environment to ensure high ecological validity.
•	Data Acquisition: To simulate the fixed-viewpoint sensing typical of automated industrial lines, we collected 10,720 high-resolution images using an iPhone 12 stabilized on a tripod at standardized heights and angles. This approach ensures image stability while capturing "in-the-wild" variability in illumination and viewpoint.
•	Classification Scope: The dataset encompasses four major manufacturers and seven distinct battery pack models:
  •	BYD: 113ADC, 113ADG
  •	CATL: 113ADH, 113ADS
  •	Gotion: 113AA2, 113ADF
  •	Sunwoda: 113ADR
•	Detection Scope: The dataset includes 14 fine-grained categories of fastening bolts, each annotated with high-precision bounding boxes for localization tasks.
•	Data Partitioning: The data is divided into a training set (8,576 images) and a testing set (2,146 images) following a rigorous 80:20 split.

3. Code Information
The framework is developed with Python 3.10 and PyTorch 2.1.2，and is built upon the OpenMMLab (MMDetection & MMEngine) ecosystem.
•	Main Entry: cocotrain8.py is the customized training script used to execute the multi-task learning pipeline.
•	Backbone: ImprovedMambaBackbone implementing the Cross-Scan mechanism to capture spatial information from four directions.
•	Neck: FPN (Feature Pyramid Network) optimized to enhance feature representation for small object (bolt) detection.
•	Heads:
- CascadeRoIHead: For robust bolt bounding box regression and localization.
- Shared2FCBrandHead: A customized classification branch incorporating State-conditioned FiLM and Prototype-assisted Fusion.
•	Data Pipeline: CustomDetDataPreprocessor manages the simultaneous loading and normalization of image tensors and multi-task ground truths.

4. Requirements
To install the necessary dependencies, ensure you have a CUDA-enabled environment:
•torch >= 1.12.0
•mmcv == 2.0.0
•mmdet == 3.0.0
•mmengine == 0.7.1
•mamba-ssm

5. Usage Instructions
5.1 Environment Setup
Follow the MMDetection Installation Guide to set up the base environment and install the required mamba-ssm dependencies.
5.2 Training
# Navigate to the project directory
cd object_detection
# Start the multi-task training process
python cocotrain8.py --config configs/mamba_vision/temp.py
5.3 Evaluation
Our evaluation suite provides a comprehensive analysis beyond standard accuracy metrics, focusing on industrial reliability:
•Standardized Metrics: Computes COCO mAP ([.50:.95]), AP50, and AR@100 for the detection task, alongside Top-1/Top-5 Accuracy and Macro F1-Score for the classification task.
•Interference Diagnosis: * Cross-Task Confusion Mapping: The system analyzes the "strongest confusers" for both tasks, identifying which brand models or bolt types are most likely to be misidentified.
•Context-Aware Bolt Statistics: Calculates bolt detection confusion matrices specifically filtered by battery brand to evaluate the impact of different background textures on bolt recognition.
•Visual Error Auditing: * Spatial Confusion Linking: For misclassified bolts, the system generates visualizations linking the Prediction to the Ground Truth to diagnose localization vs. classification errors.
•Automated Gallery Generation: Automatically synthesizes a grid-view gallery of the most significant interference cases for qualitative failure analysis.

6. Citations
If you use this code or the dataset in your research, please cite our work:
@article{mamba_csds_2026,
title={Mamba-CSDS: A Robust Dual-Stream Visual Sensing System for Automated EV Battery Disassembly under Industrial Constraints},
author={Guanfu Liao,Mohd Nashrul Mohd Zubir,Hwa Jen Yap, Mohammed A. H. Ali and Wen Tong Chong},
journal={PeerJ Computer Science},
year={2026}
}

7. License
This project is released under the MIT License. For access to the full  dataset for research purposes, please contact the corresponding author.
