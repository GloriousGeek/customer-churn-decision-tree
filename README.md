# Customer Churn Prediction using Decision Trees

This project applies a Decision Tree classifier to predict customer churn using a public banking dataset. The model was fine-tuned using GridSearchCV and evaluated with key classification metrics. The goal is to help banks proactively identify customers likely to leave, using a transparent and interpretable ML model.

---

## Problem Statement

Customer retention is a critical challenge in banking. With low switching costs and high competition, predicting customer churn is essential for designing retention strategies. This project builds a Decision Tree model to classify customers as likely to churn or not, based on demographic and account-related features.

---

## Dataset

- **Source**: [Kaggle – Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- **Type**: Binary classification
- **Target**: `churn` (0 = stay, 1 = leave)
- **Size**: ~10,000 entries

---

## Tools & Libraries

- Python (Pandas, NumPy)
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib / Seaborn

---

## Project Workflow

1. **Data Cleaning**: Dropped irrelevant columns and checked for missing values.
2. **Encoding**: Applied dummy encoding to categorical features (`gender`, `country`).
3. **Balancing**: Used SMOTE to handle class imbalance.
4. **Modeling**: Built a base Decision Tree and fine-tuned it using GridSearchCV.
5. **Evaluation**: Compared base and fine-tuned models using Accuracy, F1 Score, and ROC AUC.

---

## Results

| Model        | Accuracy | F1 Score | ROC AUC |
|--------------|----------|----------|---------|
| Base Model   | 77.0%    | 0.77     | 0.73    |
| Fine-Tuned   | 82.0%    | 0.82     | 0.84    |

The fine-tuned model demonstrates strong generalization and balanced performance. It’s interpretable and suitable for real-world banking applications.

---

## How to Run

1. Clone the repo
2. Install dependencies  
