# Image Classification using Logistic Regression and SVM with Feature Engineering

## Overview
This project focuses on image classification using **Logistic Regression** and **Support Vector Machine (SVM)** models. The dataset consists of grayscale images of various objects, which are preprocessed and used to train machine learning models. **Feature engineering** techniques, such as **Histogram of Oriented Gradients (HOG)**, are employed to improve classification accuracy.

## Project Workflow
1. **Load and Visualize the Dataset**
   - Convert images to grayscale.
   - Resize images to a uniform size **(100x100 pixels)**.
   - Flatten images for model training.
   - Display sample images from the dataset.

2. **Train Logistic Regression on Raw Pixel Values**
   - Use raw pixel values as features.
   - Train a logistic regression model.
   - Evaluate performance using accuracy metrics.

3. **Feature Engineering with HOG (Histogram of Oriented Gradients)**
   - Extract **HOG features** to enhance classification performance.
   - Compare results with models trained on raw pixel values.

4. **Train Logistic Regression on HOG Features**
   - Use **HOG-transformed data** for training.
   - Assess the impact of feature extraction on model accuracy.

5. **Train an SVM Classifier**
   - Experiment with **Support Vector Machines (SVM)** for classification.
   - Compare performance against Logistic Regression.

## Implementation

### Dataset Preparation
- Images are loaded from the **`dataset/`** directory.
- Classes include:
  - **Cars**
  - **Cricket Ball**
  - **Ice Cream**
- Images are preprocessed (**grayscale conversion and resizing**).
- The dataset is split into **training and testing sets (80/20 split).**

### Model Training
#### Logistic Regression
- **Without HOG Features:** Trained directly on raw pixel values.
- **With HOG Features:** Trained on extracted HOG descriptors.

#### Support Vector Machine (SVM)
- **Without HOG Features:** SVM trained on raw pixel values.
- **With HOG Features:** SVM trained on extracted HOG descriptors.

### Model Evaluation
- Training and test accuracy scores are calculated.
- **Confusion matrices** and **heatmaps** are used to visualize performance.
- Sample predictions are displayed.

## Model Performance Comparison

| **Model**                                  | **Feature Extractor (HOG)** | **Training Accuracy (%)** | **Test Accuracy (%)** |
|--------------------------------------------|----------------------------|--------------------------|-----------------------|
| Logistic Regression (Raw Data) - Run 1    | ❌ No                      | 100.00                   | 79.80                 |
| Logistic Regression (HOG Features) - Run 1 | ✅ Yes                     | 100.00                   | 96.97                 |
| Logistic Regression (Raw Data) - Run 2    | ❌ No                      | 97.34                    | 92.93                 |
| Logistic Regression (HOG Features) - Run 2 | ✅ Yes                     | 99.87                    | 97.47                 |

## Observations
1. **Feature extraction using HOG significantly improves test accuracy.**  
   - Without HOG, test accuracy is lower (**79.80% and 92.93%**) compared to models with HOG (**96.97% and 97.47%**).

2. **Raw data models tend to overfit.**  
   - Models without HOG achieve **100% and 97.34%** training accuracy but struggle with generalization.

## Saving the Best Model
The best-performing model (**SVM with HOG features**) is saved as **`best_model.pkl`** for future inference.

```python
import joblib
best_model = SVMClassifier
joblib.dump(best_model, "best_model.pkl")
