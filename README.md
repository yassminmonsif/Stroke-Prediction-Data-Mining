# Stroke Prediction Using Data Mining Techniques

## 1. Project Overview

This project develops a machine learning-based system for predicting stroke occurrence using patient demographic, medical, and lifestyle information.

The project applies data mining and machine learning techniques to identify patterns associated with stroke and build a predictive system that can estimate the probability of stroke for a new patient.

> **Important:** This project is for educational and research purposes. The prediction system is not a medical diagnostic tool and should not be used to make clinical decisions.

---

## 2. Problem Statement

Stroke is a serious medical condition, and identifying patients who may be at higher risk can support early investigation and preventive strategies.

The objective of this project is to use patient information such as age, hypertension, heart disease, glucose level, BMI, smoking status, and other demographic factors to build a machine learning classification model that predicts whether a patient belongs to the stroke or no-stroke class.

### Main Objective

Build and evaluate multiple data mining models for stroke prediction and develop a simple prediction interface for new patient data.

---

## 3. Dataset

The project uses the **Stroke Prediction Dataset** from Kaggle.

**Source:** Kaggle – Stroke Prediction Dataset  
**Dataset URL:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

The dataset contains **5,110 patient records** and includes demographic, medical, and lifestyle attributes.

### Features

| Feature | Description |
|---|---|
| gender | Patient gender |
| age | Patient age |
| hypertension | Whether the patient has hypertension (0/1) |
| heart_disease | Whether the patient has heart disease (0/1) |
| ever_married | Whether the patient has ever been married |
| work_type | Type of employment |
| Residence_type | Urban or Rural residence |
| avg_glucose_level | Average glucose level |
| bmi | Body Mass Index |
| smoking_status | Smoking category |
| stroke | Target variable: 0 = No Stroke, 1 = Stroke |

The original `id` field was removed because it is only an identifier and does not provide useful predictive information.

---

## 4. Dataset Characteristics

The target variable is highly imbalanced:

- No Stroke: **4,861 records (95.13%)**
- Stroke: **249 records (4.87%)**

Because of this imbalance, accuracy alone is not sufficient for evaluating the models.

The project therefore considers:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## 5. Exploratory Data Analysis (EDA)

Several exploratory analyses were performed to understand relationships between patient characteristics and stroke status.

### Main observations

- The stroke group generally contains older patients than the no-stroke group.
- Stroke rate differs between hypertension groups.
- Stroke rate differs between heart disease groups.
- Average glucose level shows different distributions between stroke and no-stroke groups.
- BMI distributions show substantial overlap between the two classes.
- Stroke rates vary across smoking-status categories.

These observations describe associations in the dataset and do not establish causal relationships.

---

## 6. Data Preprocessing

The following preprocessing steps were performed:

### 6.1 Missing Values

The `bmi` feature contained:

- **201 missing values**
- **3.93% of the dataset**

Missing BMI values were replaced using the **median BMI (28.1)**.

Median imputation was selected because BMI contains outliers and the median is less sensitive to extreme values than the mean.

### 6.2 Duplicate Records

No duplicate rows were found.

### 6.3 ID Removal

The `id` column was removed because it is an identifier rather than a meaningful predictive feature.

### 6.4 Encoding

Categorical variables were transformed using **One-Hot Encoding**.

### 6.5 Scaling

Numerical features were standardized using **StandardScaler**.

### 6.6 Train/Test Split

The dataset was divided into:

- 80% training data
- 20% testing data

Stratified splitting was used to preserve the class distribution.

### 6.7 Handling Class Imbalance

Because only 4.87% of records belong to the stroke class, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

SMOTE was applied **only to the training set** to avoid data leakage.

---

## 7. Machine Learning Algorithms

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. XGBoost

The models were evaluated both before and after feature selection.

---

## 8. Feature Selection

After preprocessing and one-hot encoding, the dataset contained **21 processed features**.

`SelectKBest` with `mutual_info_classif` was used to select the most informative features.

The number of features was reduced from:

**21 → 10 selected features**

### Selected Features

- age
- bmi
- ever_married_No
- ever_married_Yes
- work_type_Private
- smoking_status_Unknown
- smoking_status_formerly smoked
- smoking_status_never smoked
- hypertension
- heart_disease

Feature importance analysis using tree-based models also showed that **age** was one of the most influential features in the prediction task.

---

## 9. Final Model

The final experiment used:

**Logistic Regression + 10 Selected Features**

The final model was evaluated on the original unseen test set.

### Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 74.85% |
| Precision | 14.43% |
| Recall | 84.00% |
| F1-Score | 24.45% |
| ROC-AUC | 0.842 |

### Final Confusion Matrix

The final model produced:

- True Negatives (TN): **723**
- False Positives (FP): **249**
- False Negatives (FN): **8**
- True Positives (TP): **42**

The model correctly identified **42 of the 50 stroke cases** in the test set, corresponding to a recall of **84%**.

At the same time, it produced **249 false positives**, which is an important limitation and should be considered when interpreting the model.

---

## 10. Prediction System

A simple interactive prediction interface was developed using **Python and ipywidgets**.

The user can enter:

- Gender
- Age
- Hypertension
- Heart disease
- Marital status
- Work type
- Residence type
- Average glucose level
- BMI
- Smoking status

The system then returns:

- Predicted class
- Estimated stroke probability

Example output:

```text
Prediction: NO STROKE PREDICTED
Estimated Stroke Probability: XX.XX%
```

The probability is a machine-learning estimate based on patterns learned from the dataset.

---

## 11. Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn / SMOTE
- XGBoost
- ipywidgets

---

## 12. Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Encoding + Scaling
   ↓
SMOTE on Training Data
   ↓
Train Multiple Models
   ↓
Feature Importance
   ↓
Feature Selection
   ↓
Model Re-training
   ↓
Model Evaluation
   ↓
Final Stroke Prediction System
```

---

## 13. Limitations

The project has several limitations:

1. The dataset is highly imbalanced.
2. The model generates a relatively high number of false positives.
3. The dataset is not sufficient for clinical diagnosis.
4. The model has been evaluated using this dataset's train/test split and should be tested on independent external data before any real-world use.
5. Model performance can depend on preprocessing choices, feature selection, class-imbalance handling, and hyperparameters.

---

## 14. Future Work

Future improvements may include:

1. Hyperparameter optimization using Grid Search or Randomized Search.
2. Cross-validation for more robust performance estimation.
3. Testing additional ensemble and machine learning algorithms.
4. Exploring alternative class-imbalance techniques.
5. Additional feature engineering.
6. Testing the model on independent external datasets.
7. Developing a web-based prediction application.
8. Adding explainable AI techniques for individual predictions.
9. Further analysis of false positives and false negatives.
10. Clinical validation using appropriately validated medical datasets.

---

## 15. Repository Structure

```text
Stroke-Prediction-Data-Mining/
│
├── README.md
├── stroke_prediction.ipynb
├── dataset/
│   └── healthcare-dataset-stroke-data.csv
│
├── images/
│   ├── age_vs_stroke.png
│   ├── glucose_vs_stroke.png
│   ├── bmi_vs_stroke.png
│   ├── hypertension_stroke_rate.png
│   ├── heart_disease_stroke_rate.png
│   ├── smoking_stroke_rate.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
└── requirements.txt
```

---

## 16. How to Run

### Option 1: Google Colab

1. Open the notebook in Google Colab.
2. Upload the dataset.
3. Run the notebook cells from top to bottom.
4. Use the prediction interface to enter patient information.

### Option 2: Local Python Environment

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost ipywidgets
```

Then open:

```text
stroke_prediction.ipynb
```

and run the notebook.

---

## 17. Conclusion

This project demonstrates a complete data mining workflow for stroke prediction, including exploratory data analysis, data cleaning, preprocessing, class-imbalance handling, feature selection, machine learning model comparison, and an interactive prediction interface.

The final Logistic Regression model using the selected features achieved a **ROC-AUC of 0.842** and a **Recall of 84%** on the test set.

The results demonstrate the potential of machine learning to identify patterns associated with stroke in the selected dataset, while also highlighting the importance of considering false positives, class imbalance, and appropriate evaluation metrics.

**This system is an educational predictive model and is not intended to replace professional medical diagnosis or clinical assessment.**
