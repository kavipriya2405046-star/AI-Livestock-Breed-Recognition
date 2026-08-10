# 🐄🐃 Livestock Breed Recognition using Swin Transformer + CBAM
## 📌 Project Overview
This project focuses on AI-based livestock breed image classification using Deep Learning, specifically **Swin Transformer** combined with the **Convolutional Block Attention Module (CBAM)**.
The project contains separate experiments and models for livestock breed classification, including cattle, buffalo, and other livestock breed datasets.
The main objective is to build an image-based system that can learn breed-specific visual features and predict the corresponding livestock breed.

## 🎯 Objectives

- Develop an AI-based livestock breed classification system.
- Classify livestock breeds from input images.
- Explore Swin Transformer for image feature extraction.
- Improve feature representation using CBAM attention.
- Train and evaluate models using breed-specific image datasets.
- Experiment with different training and fine-tuning strategies.

## 🧠 Technologies Used

- Python
- Google Colab
- PyTorch
- Torchvision
- Timm
- Swin Transformer
- CBAM
- Scikit-learn
- Matplotlib
- Seaborn

## 🏗️ Model Architecture

The main architecture used in the project is:

**Swin Transformer + CBAM**

### General Workflow

```text
Input Livestock Image
        ↓
Image Preprocessing
        ↓
Swin Transformer
        ↓
CBAM Attention
        ↓
Feature Extraction
        ↓
Classification Head
        ↓
Predicted Breed
```

## 📂 Repository Structure

```text
AI-Livestock-Breed-Recognition/
│
├── README.md
├── buffalo_stc.ipynb
├── cow_model_st.ipynb
└── swintransformer.ipynb
```

## 📓 Project Notebooks

### 🐃 1. Buffalo Breed Classification

**File:** `buffalo_stc.ipynb`

This notebook focuses on buffalo breed classification using a Swin Transformer based deep learning model with CBAM attention.

The notebook includes:

* Dataset preparation
* Image preprocessing
* Data augmentation
* Swin Transformer
* CBAM attention
* Model training
* Fine-tuning
* Validation
* Classification report
* Confusion matrix

The buffalo dataset contains **17 breed classes** and **8,653 images**.

### 🐄 2. Cow Breed Classification

**File:** `cow_model_st.ipynb`

This notebook implements a **Swin-Tiny + CBAM** architecture for cow breed classification.

### Model Configuration

* Number of Classes: **12**
* Image Size: **224 × 224**
* Batch Size: **32**
* Number of Epochs: **50**
* Learning Rate: **1e-4**
* Weight Decay: **1e-2**

The CBAM module is used within the Swin Transformer pipeline to improve feature attention.

### 🐑 3. Swin Transformer Livestock Classification

**File:** `swintransformer.ipynb`

This notebook contains another livestock breed classification experiment using Swin Transformer.

The dataset contains **7 classes**:

* Barbari
* Goat
* Harri
* Naeimi
* Najdi
* Roman
* Sawakni

The dataset is divided into:
Training   → 70%
Validation → 15%
Testing    → 15%

## ⚙️ Dataset and Preprocessing

The notebooks perform image preprocessing and augmentation before training.

The preprocessing includes:

* Image resizing
* Random cropping
* Horizontal flipping
* Vertical flipping
* Rotation
* Color augmentation
* Normalization
* Random erasing in selected experiments

The main model experiments use **224 × 224** image inputs.

## 🧪 Model Training and Evaluation

The models are evaluated using:

* Training Accuracy
* Validation Accuracy
* Test Accuracy
* Loss
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

## 🛠️ Installation

Install the required libraries:

```bash
pip install timm torch torchvision tqdm matplotlib seaborn scikit-learn
```

## 🚀 How to Run

### Step 1: Open Google Colab

Open any of the `.ipynb` files using Google Colab.

### Step 2: Prepare the Dataset

The notebooks use datasets stored in Google Drive.

Update the dataset path according to your Google Drive location.

### Step 3: Install Dependencies

Run the required installation commands.

### Step 4: Run the Notebook

Execute the cells in sequence:
 Dataset Preparation
        ↓
 Data Preprocessing
        ↓
 Data Augmentation
        ↓
  Model Creation
        ↓
  Model Training
        ↓
   Validation
        ↓
     Testing
        ↓
Performance Evaluation

## 📊 Results

The notebooks contain the training and evaluation outputs for the different livestock classification experiments.

Performance is analyzed using accuracy, loss, precision, recall, F1-score, classification reports, and confusion matrices.

The complete experimental results are available inside the respective Jupyter notebooks.

## 🔑 Key Features

* AI-based livestock breed classification
* Swin Transformer based image classification
* CBAM attention mechanism
* Multiple livestock breed datasets
* Image preprocessing and augmentation
* Model training and fine-tuning
* Classification report
* Confusion matrix analysis
* GPU-based training using Google Colab

## 🔮 Future Enhancements

* Add more livestock species and breed classes.
* Develop a web-based livestock breed recognition application.
* Add image upload functionality.
* Add real-time camera-based prediction.
* Provide detailed information about the predicted breed.
* Deploy the trained models as an online application.
* Integrate multiple livestock models into a single system.

## 👥 Project Team

This project is developed as a team-based academic project.

### Contributors

* Kavipriya T.
* Add your other team members here

## 📜 License

This project is developed for academic and research purposes.

## ⭐ Acknowledgement

This project explores modern deep learning techniques, particularly Swin Transformer and CBAM, for livestock breed image classification.

