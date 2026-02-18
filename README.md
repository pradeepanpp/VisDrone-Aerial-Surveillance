<div align="center">

# Real-Time Aerial Surveillance Pipeline (VisDrone)
### *Optimized Object Detection for Small-Scale Aerial Targets*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-05998b.svg)](https://fastapi.tiangolo.com/)


</div>

## 📌 Project Abstract
This repository implements an end-to-end MLOps pipeline for aerial object detection, focusing on the identification of ground vehicles from high-altitude drone imagery. I utilized a **Faster R-CNN (ResNet-50 FPN)** architecture, specifically optimized to address the "Small Object Problem" inherent in aerial perspectives.

The system was validated on a representative subset of the **VisDrone-DET Dataset**. By mathematically realigning the **Anchor Generator scales to 16px**, I improved the model's sensitivity to tiny targets that occupy a minimal pixel footprint. The final system is implemented via a high-performance **FastAPI** backend, providing an interactive interface for real-time forensic target analysis.

## 🖥️ Live Inference Proof
The dashboard provides real-time tactical overlays with bounding boxes and confidence scores for detected vehicles.

<div align="center">
  <img width="1920" height="974" alt="image01" src="https://github.com/user-attachments/assets/d6e1883e-74df-447d-93a2-797ae04cbcfc" /
  <br>
  <em>Fig 1: Automated API documentation showing the /predict endpoint for real-time target ingestion.</em>
</div>

## 🔍 Qualitative Performance Analysis

<div align="center">
  <img width="1920" height="980" alt="image02" src="https://github.com/user-attachments/assets/b6ded9e9-efb8-45bd-904f-965300ea82b6" />

  <br>
  <em>Fig 2: Tactical visualization of the Anchor-Optimized Faster R-CNN resolving multiple small-scale vehicles in a high-clutter urban environment with high confidence scores.</em>
</div>



## ⚙️ Modular MLOps Workflow
To ensure scientific reproducibility, the project maintains a strictly decoupled architecture:
- **Data Engine**: Custom parser for raw VisDrone telemetry with a built-in vehicle heuristic filter.
- **Architectural Layer**: Parameterized RPN (Region Proposal Network) with custom anchor scales.
- **Deployment**: Production-ready **FastAPI** service with automated Swagger documentation.

## 🚀 Execution Guide

## 1. Environment Setup

### Create a dedicated environment
conda create -n visdrone python=3.12 -y

conda activate visdrone

### Install requirements
pip install -r requirements.txt

pip install -e .


## 2. Production API Launch
Start the FastAPI server for local testing:

python main.py

Access the interactive API dashboard at http://localhost:8080/docs
