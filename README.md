# Loan Approval Classification – Machine Learning

## Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a loan application will be approved or not. The problem is formulated as a supervised binary classification task using applicant demographic, financial, and credit-related attributes.

---

## Dataset Description

**Dataset Name:** Loan Approval Classification Dataset  
**Source:** Kaggle  
**URL:** https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data 

**Overview:**  
This dataset is a synthetic version inspired by the original Credit Risk dataset on Kaggle and enriched with additional variables based on Financial Risk for Loan Approval data. SMOTENC was used to simulate new data points to enlarge the instances. The dataset is structured for both categorical and continuous features.


**Key Features Include:**

- Applicant age  
- Annual income  
- Employment experience  
- Loan amount  
- Interest rate  
- Loan intent  
- Credit score  
- Previous loan defaults  
- Home ownership  

**Target Variable:**

- `loan_status`  
  - 1 → Loan Approved  
  - 0 → Loan Rejected  

---

## Models Implemented

The following classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest Classifier (Ensemble)  
6. XGBoost Classifier (Ensemble)  

---

## Evaluation Metrics

Each model was evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|------|
| Logistic Regression | 0.896778 | 0.951399 | 0.777893 | 0.7495 | 0.763433 | 0.697646 |
| Decision Tree | 0.898222 | 0.851179 | 0.773461 | 0.7665 | 0.769965 | 0.704634 |
| kNN | 0.895222 | 0.926789 | 0.780968 | 0.7345 | 0.757021 | 0.690840 |
| Naive Bayes | 0.811889 | 0.789287 | 0.675830 | 0.2950	 | 0.410721 | 0.357603 |
| Random Forest | 0.927667 | 0.973792 | 0.887867 | 0.7720 | 0.825889 | 0.783560 |
| XGBoost | 0.929333 | 0.975807 | 0.891954 | 0.7760 | 0.829947 | 0.788653 |

---

## Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Logistic Regression provided a strong baseline with stable performance. |
| Decision Tree | Decision Tree captured non-linear relationships but showed signs of overfitting. |
| kNN | kNN performance depended heavily on feature scaling. |
| Naive Bayes |  Naive Bayes achieved very high recall but lower precision due to independence assumptions. |
| Random Forest | Random Forest improved generalization through ensemble learning. |
| XGBoost  | XGBoost achieved strong predictive performance due to boosting and regularization.

---
