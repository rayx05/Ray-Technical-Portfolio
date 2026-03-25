# Depression Prediction Using Machine Learning

Predicting depression diagnosis in students using survey data and supervised 
machine learning models.

## Problem Statement

Mental health screening at scale is challenging — this project explores whether 
student depression can be predicted from observable behavioral and academic 
factors using machine learning, without requiring clinical diagnosis.

## Dataset

- **Source:** Student Depression Dataset (Kaggle)
- **Size:** 27,901 records, filtered to 27,837 student observations
- **Features:** Academic pressure, CGPA, sleep duration, dietary habits, 
financial stress, suicidal thoughts history, family mental health history, 
and more
- **Target:** Binary depression diagnosis (0 = No, 1 = Yes)

## Methods

**Data Preprocessing**
- Filtered dataset to student population only
- Converted categorical variables to numeric 
  (sleep duration, dietary habits, gender, suicidal thoughts)
- Dropped high-cardinality features (city, degree) to reduce noise
- Handled missing values via row removal

**Models Evaluated**
- Decision Tree Classifier (depth-optimized via GridSearchCV)
- Naive Bayes
- Logistic Regression
- Linear SVC
- Linear Regression (for comparison)

**Validation**
- Train/test split (75/25)
- 5-fold cross-validation for all classifiers

## Key Results

| Model | Accuracy |
|---|---|
| Decision Tree (depth=7) | 82.8% |
| Cross-validated Decision Tree | 84.6% |
| Linear Regression (R²) | 0.517 |

**Top predictive features (Decision Tree):**
Suicidal thoughts history, academic pressure, and financial stress emerged 
as the strongest predictors of depression diagnosis.

## Tools

Python, pandas, scikit-learn, matplotlib, seaborn, Google Colab

## How to Run

Open `Depression_Prediction_Machine_Learning.ipynb` in Jupyter or Google Colab.
Update the dataset path if running locally.
