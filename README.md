# Explainable AI for Diabetic Retinopathy Screening in Rural India

An AI-powered and explainable diabetic retinopathy screening system designed to assist in the early detection and analysis of diabetic eye disease, with a focus on improving accessibility in rural and resource-constrained healthcare environments.

The system combines deep learning-based retinal image classification with explainability, retinal analysis, image-quality assessment, medical knowledge, hospital assistance, multilingual support, and automated report generation.

---

## 📌 Overview

Diabetic Retinopathy (DR) is a diabetes-related eye disease that can cause vision loss if it is not detected and treated early.

In many rural and remote areas, access to ophthalmologists and specialized retinal screening facilities is limited. This project aims to provide an AI-assisted screening system that can analyze retinal fundus images and provide an understandable assessment to support healthcare professionals.

The key focus of the project is not only **prediction**, but also **explainability**.

The system attempts to answer:

- What is the predicted severity of diabetic retinopathy?
- How confident is the model?
- Which regions of the retinal image influenced the prediction?
- Is the uploaded image of sufficient quality?
- What does the result mean in simple medical terms?
- Where can the patient find appropriate healthcare support?

---

## 🎯 Objectives

- Detect diabetic retinopathy from retinal fundus images.
- Classify retinal images into different severity levels.
- Provide explainable AI outputs using visual explanations.
- Assess retinal image quality before analysis.
- Analyze important retinal features.
- Provide medical information in an understandable format.
- Generate an automated screening report.
- Assist users in finding nearby hospitals or healthcare facilities.
- Support multilingual communication.
- Build a system that can assist screening in rural and resource-constrained areas.

---

## 🧠 AI Model

The project uses a deep learning-based image classification approach.

### Model

**ResNet18**

ResNet18 is a convolutional neural network architecture based on residual learning.

The model can learn visual patterns from retinal fundus images and use these features to classify diabetic retinopathy severity.

### Transfer Learning

A pretrained ResNet18 model can be fine-tuned on retinal images instead of training a deep neural network completely from scratch.

This helps reduce:

- Training time
- Computational requirements
- Required training data
- Model convergence time

---

## 🔍 Explainable AI

A major component of this project is **Explainable AI (XAI)**.

Deep learning models can provide predictions without clearly explaining why a particular prediction was made.

To address this issue, the project uses explainability techniques such as **Grad-CAM**.

### Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) generates a heatmap showing the regions of an image that contributed to the model's prediction.

For retinal images, this can help visualize areas that the model considered important during classification.

### Why Explainability Matters

Explainability can help:

- Healthcare professionals understand AI predictions.
- Identify whether the model is focusing on meaningful retinal regions.
- Increase transparency.
- Detect potentially incorrect predictions.
- Improve trust in AI-assisted screening systems.

---

## 🩺 Retinal Analysis

The project contains a dedicated retinal analysis component.

`retinal_analysis.py`

This component can be used to perform analysis of retinal images and extract relevant information that can support the screening process.

---

## 🖼️ Image Quality Assessment

Before performing AI-based analysis, image quality is an important consideration.

`image_quality.py`

This module is responsible for assessing whether an uploaded retinal image is suitable for analysis.

Poor-quality images may contain:

- Blur
- Insufficient illumination
- Poor contrast
- Unclear retinal structures
- Other imaging artifacts

Checking image quality can help reduce unreliable predictions caused by unsuitable input images.

---

## 🧪 Model Training

`train.py`

This file handles the training process of the deep learning model.

The general workflow is:

```text
Retinal Dataset
      ↓
Image Loading
      ↓
Preprocessing
      ↓
Data Augmentation
      ↓
ResNet18
      ↓
Model Training
      ↓
Trained Model
