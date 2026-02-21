<div align="center">

# Aerial Object Detection Pipeline (VisDrone)
### *Optimized Object Detection for Small-Scale Aerial Targets*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-05998b.svg)](https://fastapi.tiangolo.com/)


</div>

## 📌 Project Abstract
This project implements a modular object detection pipeline for drone imagery (VisDrone-DET subset) using **Faster R-CNN (ResNet-50 FPN)**.  
To improve detection of **small objects**, I tuned the **RPN anchor scales** (including a 16px scale) and exposed the model through a **FastAPI inference service** with a `/predict` endpoint and Swagger UI.

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


## 2. API Launch
Start the FastAPI server for local testing:

python main.py

Access the interactive API dashboard at http://localhost:8080/docs
