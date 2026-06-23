# 🩺 Diabetic Retinopathy Severity Detection using Deep Learning

## 📌 Overview

Diabetic Retinopathy (DR) is a diabetes-related eye disease that can lead to vision impairment and blindness if not detected early. Manual screening of retinal fundus images is time-consuming and requires expert ophthalmologists.

This project presents an AI-powered system that automatically classifies retinal fundus images into different stages of Diabetic Retinopathy using Deep Learning techniques. The system utilizes a pre-trained DenseNet121 model and provides severity prediction through a user-friendly Streamlit web application.

---

## 🎯 Objectives

- Detect Diabetic Retinopathy from retinal fundus images.
- Classify images into different severity levels.
- Improve image quality using preprocessing techniques.
- Build an end-to-end deployable application.
- Provide model confidence scores and probability distributions.
- Improve interpretability using Grad-CAM visualizations.

---

## 📂 Dataset

The project uses retinal fundus images categorized into five classes:

| Class | Description |
|---------|------------|
| No_DR | No Diabetic Retinopathy |
| Mild | Mild Diabetic Retinopathy |
| Moderate | Moderate Diabetic Retinopathy |
| Severe | Severe Diabetic Retinopathy |
| Proliferative_DR | Proliferative Diabetic Retinopathy |

Dataset Source:

- APTOS Blindness Detection Dataset
- Diabetic Retinopathy 224x224 Gaussian Filtered(Kaggle)


---

## 🖼️ Image Preprocessing

Retinal fundus images often contain black borders and varying illumination conditions that can affect model performance. To improve image quality and focus the model on relevant retinal features, the following preprocessing techniques were applied:

### 1. Black Border Removal (`crop_black`)

Many retinal images contain large black regions surrounding the retina. These regions do not contain useful clinical information and may introduce noise during training.

Benefits:
- Removes unnecessary background information.
- Focuses the model on retinal structures.
- Improves feature extraction.

### 2. CLAHE Enhancement (`apply_clahe`)

Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied to improve local image contrast.

Benefits:
- Enhances blood vessels and retinal lesions.
- Improves visibility of subtle abnormalities.
- Helps the model learn discriminative features more effectively.


### 3. DenseNet Preprocessing

Images were processed using:

```python
tensorflow.keras.applications.densenet.preprocess_input()
```

to prepare them for the pre-trained DenseNet121 architecture.

## 🧠 Deep Learning Architecture

### Transfer Learning

The project uses:

**DenseNet121**

pre-trained on ImageNet.

Advantages:

- Efficient feature reuse.
- Lower parameter count.
- Strong performance on medical imaging tasks.

---


## ⚖️ Class Imbalance Handling

Medical datasets often contain more healthy images than diseased images.

Class weights were computed and applied during training:

```python
class_weight=class_weights
```

Benefits:

- Reduces bias toward majority classes.
- Improves minority class recognition.

---

## 📊 Model Evaluation

The model was evaluated using:

### Accuracy

Measures overall prediction correctness.

### Precision

Measures prediction quality.

### Recall

Measures ability to detect positive cases.

### F1 Score

Balances Precision and Recall.

### Confusion Matrix

Provides class-wise performance analysis.

---

## 📉 Training Performance

Best Model Performance:

| Metric | Value |
|----------|----------|
| Accuracy | ~76% |
| Weighted F1 Score | ~0.75 |
| Macro F1 Score | ~0.63 |

The model demonstrated strong performance on:

- No_DR
- Moderate DR

while minority classes such as:

- Severe DR
- Proliferative DR

remain challenging due to class imbalance.

---

## 🔥 Grad-CAM Explainability

To improve model interpretability, Grad-CAM was integrated.

### Purpose

Visualize retinal regions that influence model predictions.

Benefits:

- Increased transparency.
- Better understanding of model decisions.
- Useful for medical AI applications.

Output:

- Original Image
- Prediction
- Grad-CAM Heatmap Overlay

---

## 🌐 Streamlit Web Application

The project includes a Streamlit-based interface.

### Features

✅ Upload retinal image

✅ Automated preprocessing

✅ Severity prediction

✅ Confidence score

✅ Class probabilities

✅ Medical interpretation report

✅ Grad-CAM visualization

---

## 🚀 Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Streamlit App

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 💻 Technologies Used

### Deep Learning

- TensorFlow
- Keras

### Computer Vision

- OpenCV

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib

### Deployment

- Streamlit

---


## 🔮 Future Enhancements

- EfficientNet-based architecture
- Ensemble models
- Higher-resolution retinal analysis
- Multi-stage DR grading
- Cloud deployment
- Doctor recommendation system
- Automated report generation

---
